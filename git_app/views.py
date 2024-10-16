import time
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
import git
import datetime
import os
from git_app.prompts import base_prompt
import google.generativeai as genai
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from dotenv import load_dotenv
from django.contrib.auth.models import User
from git_app.models import GitHubRepo, Project, Report, CronJob
import schedule
import threading
import subprocess
import logging

logger = logging.getLogger(__name__)


load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

llm = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)


# Function to run the schedule in a separate thread
def run_scheduler():
    while True:
        now = datetime.datetime.now()

        # Calculate the next Monday 12 AM
        next_monday = now + datetime.timedelta((0 - now.weekday()) % 7)  # 0 represents Monday
        next_monday_midnight = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)

        # Calculate the time difference in seconds
        time_until_monday_midnight = (next_monday_midnight - now).total_seconds()

        # Print the time we're sleeping for (until next Monday 12 AM)
        logger.info(f"Sleeping for {time_until_monday_midnight / (60 * 60):.2f} hours until next Monday 12 AM")

        # Sleep until next Monday 12 AM
        time.sleep(time_until_monday_midnight)

        # Run the scheduled tasks
        logger.info("Running scheduled tasks")
        schedule.run_all()

# CronJobs in Background
def start_jobs():
    try:
        jobs = CronJob.objects.all()
        for job in jobs:
            wsgi_request = HttpRequest()
            wsgi_request.POST = {
                'username': job.username,
                'repo_name': job.repo_name,
                'contributor': job.contributor,
                'token': job.token,
                'emails': job.emails
            }
            response = get_weekly_report(wsgi_request)
            if response.status_code == 200: 
                logger.info(f"Successfully Analysed {job.repo_name} and Sent Report") 
            else: 
                logger.info(f"Something went Wrong while Analysing {job.repo_name}")
                
        
    except Exception as e:
        logger.info(f"Error: {e}")

schedule.every().monday.at("00:00").do(start_jobs)
# schedule.every(3).minutes.do(start_jobs)
logger.info("CronJobs Created")
# Start a new thread for the scheduler
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True  # Daemonize thread so it exits when main program exits
scheduler_thread.start()



def clone_repo_and_get_commits(repo_url, dest_folder):
    content = ""
    # dest_folder = "./repo/" + repo_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(dest_folder):
        try:
            git.Repo.clone_from(repo_url, dest_folder)
            # subprocess.run(['git', 'clone', repo_url, dest_folder])
            logger.info(f"Repository cloned to {dest_folder}")
        except Exception as e:
            logger.info(f"Error cloning repository: {e}")
            content = f"Error cloning repository: {e}"
    else:
        logger.info(f"Repository already exists at {dest_folder}. Pulling latest changes...")
        try:
            # Initialize GitPython Repo object
            logger.info(f"Initializing Repo : {dest_folder}")
            repo = git.Repo(dest_folder)
            repo.remotes.origin.pull()
        except Exception as e:
            logger.info(f"Error: Pulling {e}")

    
    repo = git.Repo(dest_folder)

    # Get the commits from the last week
    last_week = datetime.datetime.now() - datetime.timedelta(weeks=1)
    commits = list(repo.iter_commits(since=last_week.isoformat()))

    # Print the commit details
    if not commits:
        logger.info("No commits found in the last week")
        content = "<h2>No commits found in the last week</h2>"
    else:
        content = commit_diff(commits)
    return content

def commit_diff(commits):
    content = ""
    for commit in commits:
        logger.info("Commmit")
        logger.info(f"Commit: {commit.hexsha}")
        logger.info(f"Author: {commit.author.name}")
        logger.info(f"Date: {commit.committed_datetime}")
        logger.info(f"Message: {commit.message}")
        logger.info("\n" + "-"*60 + "\n")
        # logger.info("Changes:")
        
        # Iterating over all files in the commit
        for item in commit.tree.traverse():
            if isinstance(item, git.objects.blob.Blob):
                file_path = item.path
                blob = commit.tree / file_path
                file_contents = blob.data_stream.read()
                content += f"\n\n--- {file_path} ---\n\n"
                content += f"```{file_contents}```"


        # Parent commits
        parent_shas = [parent.hexsha for parent in commit.parents]
        logger.info(f"Parent Commits: {', '.join(parent_shas)}")
        content += f"Parent Commits: {', '.join(parent_shas)} <br>"
        # Commit stats
        stats = commit.stats.total
        # content += str(stats)
        logger.info(f"Stats: {stats}")
        # commits_changes = f"""Commit: {commit.hexsha}\n Author: {commit.author.name}\nDate: {commit.committed_datetime}\nMessage: {commit.message}\n
                # Parent Commits: {', '.join(parent_shas)}\nStats: {stats}"""

        # Diff with parent
        if commit.parents:
            diffs = commit.diff(commit.parents[0])
            for diff in diffs:
                content += f"<br> Changed Files: <br> --- {diff.a_path} ---"
                logger.info("Difference:")
                logger.info(f"File: {diff.a_path}")
                logger.info(f"New file: {diff.new_file}")
                logger.info(f"Deleted file: {diff.deleted_file}")
                logger.info(f"Renamed file: {diff.renamed_file}")
                # logger.info(f"Changes:\n{diff.diff}")

                if diff.diff:
                    logger.info(diff.diff.decode('utf-8'))#, language='diff')

            logger.info("\n" + "-"*60 + "\n")
        # logger.info(f"Content: \n{content}")
    with open('output.txt', 'w') as f:
        f.write(content)
    return content
    

# Function to traverse all files and write their contents to a single file
def traverse_and_copy(src_folder, output_file):
    # Define unwanted file extensions or patterns
    unwanted_extensions = [
        '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.exe', '.bin', 
        '.lock', '.generators', '.yml', '.scss', '.css', '.html', '.erb',
        '.sample', '.rake', '.haml']
    unwanted_files = ['LICENSE', 'README.md', '.dockerignore',  'manifest.js', 'exclude', 'repositories']
    logger.info("Copying the files")
    logger.info(f"Skipping Extensions {unwanted_extensions} and Files {unwanted_files}.")
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
        for root, _, files in os.walk(src_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if ((os.path.splitext(file)[1].lower() in unwanted_extensions) or 
                    (file in unwanted_files) or 
                    (is_binary(file_path))):
                    continue
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    outfile.write(f"--- {file_path} ---\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")

def detect_framework(project_dir):
    FRAMEWORK_FILES = {
        'Django': ['manage.py', 'settings.py', 'urls.py'],
        'Flask': ['app.py'],
        'React': ['package.json', 'src/App.js', 'public/index.html'],
        'Vue.js': ['package.json', 'src/App.vue', 'vue.config.js'],
        'Angular': ['package.json', 'angular.json', 'src/main.ts'],
        'Express.js': ['app.js', 'package.json', 'server.js'],
        'Ruby on Rails': ['Gemfile', 'config/routes.rb', 'db/migrate'],
        'Laravel': ['artisan', 'composer.json', 'routes/web.php'],
        'Symfony': ['composer.json', 'bin/console', 'config/services.yaml'],
        'Spring Boot': ['pom.xml', 'src/main/resources/application.properties'],
        'ASP.NET Core': ['Program.cs', 'Startup.cs', 'appsettings.json'],
        'Gin': ['main.go', 'go.mod'],
        'Echo': ['main.go', 'go.mod'],
        'Next.js': ['package.json', 'pages', 'next.config.js'],
        'Nuxt.js': ['package.json', 'pages', 'nuxt.config.js'],
        'Bootstrap': ['index.html', 'package.json'],
        'Tailwind CSS': ['tailwind.config.js', 'package.json'],
        'Foundation': ['index.html', 'package.json'],
        'React Native': ['package.json', 'App.js', 'android', 'ios'],
        'Flutter': ['pubspec.yaml', 'lib', 'android', 'ios'],
        'Qt': ['.pro', 'CMakeLists.txt'],
        'Boost': ['CMakeLists.txt']
    }

    detected_frameworks = []

    for framework, files in FRAMEWORK_FILES.items():
        # Check if any of the framework-specific files exist
        for file in files:
            if os.path.exists(os.path.join(project_dir, file)):
                detected_frameworks.append(framework)
                break  # Found the framework, move to next

    return detected_frameworks
                
# Function to check if the file is binary
def is_binary(file_path):
    try:
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(1024), b''):
                if b'\0' in chunk:
                    return True
    except Exception as e:
        logger.info(f"Could not read {file_path} to check if it's binary: {e}")
    return False


def analyze_repo(params, output_file):
    try:
        username = params['username']
        repo_name = params['repo_name']
        contributor = params['contributor']
        token = params['token']

        content = ''

        with open(output_file, 'r', encoding='utf-8') as file:
            content = file.read()

        prompts = []
            
        total_tokens = llm.count_tokens(content).total_tokens
        logger.info(f"Total Tokens: {total_tokens}")
        if total_tokens > 1000000:
            chunk_size = 1000000
            chunks = [content[i:i+chunk_size] if i+chunk_size <= len(content) else content[i:] for i in range(0, len(content), chunk_size)]
            prompts = [base_prompt.replace("context_here", chunk) for chunk in chunks]
            logger.info(f"Divided into {len(prompts)} prompts")
        else:
            prompts.append(base_prompt.replace("context_here", content))
            # logger.info(f"Prompt: {llm.count_tokens(prompts[-1])}")
        reports = []
        for prompt in prompts:
            logger.info("Generating Response")
            response = llm.generate_content(prompt)
            report = response.text
            reports.append(report)
            logger.info(f"Output: {llm.count_tokens(response.text)}")
            logger.info("Wait for 30 seconds")
            time.sleep(30)  # Wait for 30 seconds before generating the next report
        if len(reports) > 1:
            combined_report = "\n\n".join(reports)
            response = llm.generate_content(f"Combine these Contents and Generate a single one in HTML format ```content : {combined_report}```")
            reports.append(response.text)
            logger.info(f"Combined Output: {llm.count_tokens(response.text)}")
        logger.info("Responses generated successfully")
        return "\n\n".join(prompts), reports[-1]
    except Exception as e:
        logger.info(f"Error: {e}")

# Send Email using Brevo
def send_brevo_mail(subject, html_content, emails):
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    # subject = "My Subject"
    # html_content = "<html><body><h1>This is my first transactional email </h1></body></html>"
    to = None
    if isinstance(emails, str):
        to = [{"email":email.strip(), "name":email.split("@")[0]} for email in emails.split(',')]
    logger.info(f"Number of emails: {len(to)}")
    
    # # Create a list of dictionaries for the 'to' parameter
    # to = [{"email": email, "name": email.split("@")[0]} for email in emails]
    # to = emails
    cc = [{"email":"mdkhaisars118@gmail.com", "name":"Mohammed Khaisar"}]
    bcc = [{"email":"mdkhaisars118@gmail.com", "name":"Mohammed Khaisar"}]
    sender = {"name":"Mohammed Khaisar", "email":"khaisar@betacraft.io"}
    headers = {"Some-Custom-Name":"unique-id-12154"}
    params = {"parameter":"My param value","subject":"New Subject"}
    # for email in emails:
    # to = [{"email":email, "name":email.split("@")[0]}]
    logger.info(f"To: {to}")
    try:
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(to=to, cc=cc, bcc=bcc, headers=headers, html_content=html_content, sender=sender, subject=subject)
        api_response = api_instance.send_transac_email(send_smtp_email)
        logger.info(f"Email sent successfully: {api_response}")
        return True, "Email sent successfully"
    except Exception as e:
        logger.info(f"Unexpected error when sending email: {e}")
        return False, f"An unexpected error occurred while sending the email {e}"


def index(request):
    return render(request, '../templates/git_app/input_form.html')

def send_message(request):
    chat_history = request.GET.get('chat_history')
    message = request.GET.get('message')
    relevent_context = ''

    
    # Prepare the prompt for the LLM
    prompt = f"""
    You are an AI assistant helping with code analysis and development questions.
    
    Chat history:
    {chat_history}
    
    User's latest message: {message}

    relevent context: {relevent_context}
    
    Based on the chat history and the user's latest message, generate a helpful and relevant response. 
    If the user is asking about code or development, try to provide specific and accurate information.
    If you're not sure about something, it's okay to say so.
    
    Your response:
    """
    # Generate response using the LLM
    response = llm.generate_content(prompt)
    # Extract the generated text
    generated_text = response.text
    return JsonResponse({'response': generated_text})
           
def get_weekly_report(request):
    username = request.POST.get('username')
    repo_name = request.POST.get('repo_name')
    contributor = request.POST.get('contributor')
    token = request.POST.get('token')
    emails = request.POST.get('emails')
    sprint = request.POST.get('sprint')

    params = {
        'username': username,
        'repo_name': repo_name,
        'contributor': contributor,
        'token': token,
        'emails': emails
    }

    try:
        repo_url = f"https://{token}@github.com/{username}/{repo_name}.git" if token else f"https://github.com/{username}/{repo_name}.git"
        logger.info(f"Analyzing repo: {repo_url}")
            
        dest_folder = f"repositories/{username}/{repo_name}"

        clone_repo_and_get_commits(repo_url, dest_folder)
        frameworks = detect_framework(dest_folder)
        traverse_and_copy(dest_folder, 'weekly.txt')
        params['framework'] = ''.join(frameworks)
        logger.info(f"Framework: {''.join(frameworks)}")
        prompt, response = analyze_repo(params, 'weekly.txt')

        user, created_user = User.objects.get_or_create(username=username)
        repo, created_repo = GitHubRepo.objects.get_or_create(
            name=repo_name,
            user=user
        )
        if created_repo:
            repo.save()
        report, created_report = Report.objects.get_or_create(
            name=repo_name,
            emails=emails,
            repository_url=repo_url,
            repository_token=token,
            active=True,
            frequency='Weekly',
            user=user,
            prompt=prompt,
            output=response
        )
        if created_report:
            report.save()
        if not username or not repo_name:
            raise ValueError("Username and repository name are required")
        logger.info(f"New report created for project '{repo_name}': {response}")

        send_brevo_mail(subject=f"{repo_name}", 
                        html_content=response, 
                        emails=emails)
        
        job, created_job = CronJob.objects.get_or_create(
            repo_name=repo_name,
            defaults={
                'username': username,
                'contributor': contributor,
                'token': token,
                'emails': emails
            }
        )
        if created_job:
            job.save()
            logger.info("Repository Added to Cronjob Successfully")
        else:
            job.username = username
            job.contributor = contributor
            job.token = token
            job.emails = emails
            job.save()
            logger.info("CronJob already Exists for this Repository, Details Updated")
        
    except Exception as e:
        logger.info(f"Error: {e}")
        return JsonResponse({"status": "Failed", "error":e}, status=500)
    
    return JsonResponse({"status": "success"})
