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
from git_app.models import GitHubRepo, Project, Report
import schedule
import threading

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

def clone_repo_and_get_commits(repo_url, dest_folder):
    content = ""
    # dest_folder = "./repo/" + repo_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(dest_folder):
        try:
            git.Repo.clone_from(repo_url, dest_folder)
            print(f"Repository cloned to {dest_folder}")
        except git.exc.GitCommandError as e:
            print(f"Error cloning repository: {e}")
            content = f"Error cloning repository: {e}"
    else:
        print(f"Repository already exists at {dest_folder}. Pulling latest changes...")
        repo = git.Repo(dest_folder)
        repo.remotes.origin.pull()

    # Initialize GitPython Repo object
    print(f"Initializing Repo : {dest_folder}")
    repo = git.Repo(dest_folder)

    # Get the commits from the last week
    last_week = datetime.datetime.now() - datetime.timedelta(weeks=1)
    commits = list(repo.iter_commits(since=last_week.isoformat()))

    # Print the commit details
    if not commits:
        print("No commits found in the last week")
        content = "<h2>No commits found in the last week</h2>"
    else:
        content = commit_diff(commits)
    return content

def commit_diff(commits):
    content = ""
    for commit in commits:
        print("Commmit")
        print(f"Commit: {commit.hexsha}")
        print(f"Author: {commit.author.name}")
        print(f"Date: {commit.committed_datetime}")
        print(f"Message: {commit.message}")
        print("\n" + "-"*60 + "\n")
        # print("Changes:")
        
        # Iterate over all files in the commit
        for item in commit.tree.traverse():
            if isinstance(item, git.objects.blob.Blob):
                file_path = item.path
                blob = commit.tree / file_path
                file_contents = blob.data_stream.read()
                content += f"\n\n--- {file_path} ---\n\n"
                content += f"```{file_contents}```"


        # Parent commits
        parent_shas = [parent.hexsha for parent in commit.parents]
        print(f"Parent Commits: {', '.join(parent_shas)}")
        content += f"Parent Commits: {', '.join(parent_shas)} <br>"
        # Commit stats
        stats = commit.stats.total
        # content += str(stats)
        print(f"Stats: {stats}")
        # commits_changes = f"""Commit: {commit.hexsha}\n Author: {commit.author.name}\nDate: {commit.committed_datetime}\nMessage: {commit.message}\n
                # Parent Commits: {', '.join(parent_shas)}\nStats: {stats}"""

        # Diff with parent
        if commit.parents:
            diffs = commit.diff(commit.parents[0])
            for diff in diffs:
                content += f"<br> Changed Files: <br> --- {diff.a_path} ---"
                print("Difference:")
                print(f"File: {diff.a_path}")
                print(f"New file: {diff.new_file}")
                print(f"Deleted file: {diff.deleted_file}")
                print(f"Renamed file: {diff.renamed_file}")
                # print(f"Changes:\n{diff.diff}")

                if diff.diff:
                    print(diff.diff.decode('utf-8'))#, language='diff')

            print("\n" + "-"*60 + "\n")
        # print(f"Content: \n{content}")
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
    unwanted_files = ['LICENSE', 'README.md', '.dockerignore',  'manifest.js', 'exclude']
    print("Copying the files")
    print(f"Skipping Extensions {unwanted_extensions} and Files {unwanted_files}.")
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
        print(f"Could not read {file_path} to check if it's binary: {e}")
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
        
        prompt = base_prompt.replace("context_here", content)
        print(f"Prompt: {llm.count_tokens(prompt)}")
        response = llm.generate_content(prompt)
        report = response.text
        print(f"Output: {llm.count_tokens(response.text)}")
        print("Response generated successfully")
        return prompt, response.text
    except Exception as e:
        print(f"Error: {e}")

# Send Email using Brevo
def send_brevo_mail(subject, html_content, emails):
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    # subject = "My Subject"
    # html_content = "<html><body><h1>This is my first transactional email </h1></body></html>"
    to = None
    if isinstance(emails, str):
        to = [{"email":email.strip(), "name":email.split("@")[0]} for email in emails.split(',')]
    print(f"Number of emails: {len(emails)}")
    
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
    print(f"To: {to}")
    try:
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(to=to, cc=cc, bcc=bcc, headers=headers, html_content=html_content, sender=sender, subject=subject)
        api_response = api_instance.send_transac_email(send_smtp_email)
        print(f"Email sent successfully: {api_response}")
        return True, "Email sent successfully"
    except Exception as e:
        print(f"Unexpected error when sending email: {e}")
        return False, f"An unexpected error occurred while sending the email {e}"

# Function to run the schedule in a separate thread
def run_scheduler():
    while True:
        today = datetime.date.today()
        last_week = today - datetime.timedelta(weeks=1)
        print("Running")
        schedule.run_pending()
        time.sleep(60*60*24*7)

def index(request):
    return render(request, '../templates/git_app/input_form.html')
           
def get_weekly_report(request):
    username = request.POST.get('username')
    repo_name = request.POST.get('repo_name')
    contributor = request.POST.get('contributor')
    token = request.POST.get('token')
    emails = request.POST.get('emails')

    params = {
        'username': username,
        'repo_name': repo_name,
        'contributor': contributor,
        'token': token,
        'emails': emails
    }

    try:
        repo_url = f"https://{token}@github.com/{username}/{repo_name}.git" if token else f"https://github.com/{username}/{repo_name}.git"
        print(f"Analyzing repo: {repo_url}")
            
        dest_folder = f"repositories/{username}/{repo_name}"

        clone_repo_and_get_commits(repo_url, dest_folder)
        frameworks = detect_framework(dest_folder)
        traverse_and_copy(dest_folder, 'weekly.txt')
        params['framework'] = ''.join(frameworks)
        print(f"Framework: {''.join(frameworks)}")
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
        print(f"New report created for project '{repo_name}': {response}")
        emails = ['mdkhaisars118@gmail.com']
        send_brevo_mail(subject=f"{repo_name} - Prompt Engineering", 
                        html_content=response, 
                        emails=emails)
        
        today = datetime.date.today()
        last_week = today - datetime.timedelta(weeks=1)
        
        
        if emails is not None:
            wsgi_request = HttpRequest()
            wsgi_request.POST = {
                'username': username,
                'repo_name': repo_name,
                'contributor': contributor,
                'token': token,
                'emails': emails
            }
            schedule.every().monday.at("01:00").do(get_weekly_report, 
                        wsgi_request)
            # schedule.every().minute.do(get_weekly_report, 
            #             wsgi_request)

            # Start a new thread for the scheduler
            scheduler_thread = threading.Thread(target=run_scheduler)
            scheduler_thread.daemon = True  # Daemonize thread so it exits when main program exits
            scheduler_thread.start()
            print("CronJob scheduled succesfully")
        else:
            print("Emails not provided")
        
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({"status": "Failed"}, status=500)
    
    return JsonResponse({"status": "success"})
    
