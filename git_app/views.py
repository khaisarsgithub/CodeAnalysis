import time
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
import git
import datetime
import os
from git_app.prompts import base_prompt, sprint_prompt
import google.generativeai as genai
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from dotenv import load_dotenv
from django.contrib.auth.models import User
from git_app.models import Project, Report, CronJob, Error
import schedule
import threading
import subprocess
import logging
# from openai import OpenAI
# from anthropic import Anthropic
import json
# from llamaapi import LlamaAPI
import tiktoken

from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import requests
import pytz
from typing import Dict
import base64
from urllib.parse import quote


load_dotenv()

# Configure logging to output to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Brevo Email Configuration
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))


# Configuring API keys for LLMs
# llama = LlamaAPI(os.getenv("LLAMA_API_KEY"))
# gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# claude_sonet = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "temperature": 0.7,  
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)

def test_repo_access(repo_owner, repo_name, token):
    # Base URL for the repo
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    
    # Headers with authentication
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github.v3+json',
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    logger.info(f"Testing access to: {url}")
    
    try:
        # Test repository access
        response = requests.get(url, headers=headers)
        logger.info(f"Status Code: {response.status_code}")
        status_code = response.status_code
        
        if response.status_code == 404:
            logger.info("Repository not found! Please check:")
            logger.info("1. Repository owner name is correct")
            logger.info("2. Repository name is correct")
            logger.info("3. Repository exists and is not private")
            logger.info(f"4. Full repo path: {repo_owner}/{repo_name}")
            status_code = response.status_code
            raise ValueError("Invalid Repository")
        elif response.status_code == 401:
            logger.info("Authentication failed! Please check your token.")
            status_code = response.status_code
            raise ValueError("Authentication failed! Please check your token.")
        elif response.status_code == 200:
            logger.info("Repository found and accessible!")
            repo_info = response.json()
            logger.info(f"\nRepository Details:")
            logger.info(f"Full Name: {repo_info['full_name']}")
            logger.info(f"Visibility: {repo_info['visibility']}")
            logger.info(f"Default Branch: {repo_info['default_branch']}")
            status_code = response.status_code
        else:
            logger.info(f"Unexpected status: {response.status_code}")
            logger.info(f"Response: {response.text}")
            status_code = response.status_code
            raise ValueError(response.text)
        return status_code, "Success"
    except Exception as e:
        logger.info(f"Error: {str(e)}")
        return status_code, str(e)

# Function to run the schedule in a separate thread
def run_scheduler():
    while True:
        now = datetime.datetime.now()

        # Calculate the next Monday 12 AM
        next_monday = now + datetime.timedelta((0 - now.weekday()) % 7)  # 0 represents Monday
        next_monday_midnight = next_monday.replace(hour=0, minute=0, second=0, microsecond=0)

        # Calculate the time difference in seconds
        time_until_monday_midnight = (next_monday_midnight - now).total_seconds()

        # Ensure time_until_monday_midnight is non-negative
        if time_until_monday_midnight < 0:
            logger.info("Current time is past next Monday 12 AM, recalculating for the following week.")
            time_until_monday_midnight += 604800  # Add 7 days in seconds

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

def get_code_changes(repo_owner, repo_name, token):
    # Calculate date range for last week
    end_date = datetime.datetime.now(pytz.utc)
    start_date = end_date - datetime.timedelta(days=7)
    
    logger.info(f"Token: {token}")
    
    # Format dates for GitHub API
    since = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    until = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github.v3+json',
        "X-GitHub-Api-Version": "2022-11-28"
    }
    content = ''
    
    try:
        # Get commits from last week
        commits_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
        logger.info(f"Fetching commits from URL: {commits_url}?since={since}&until={until}")
        response = requests.get(
            commits_url,
            headers=headers,
            params={'since': since, 'until': until}
        )
        response.raise_for_status()
        commits = response.json() 
        for commit in commits:
            # Get detailed commit info with patch data
            commit_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit['sha']}"
            commit_response = requests.get(commit_url, headers=headers)
            commit_response.raise_for_status()
            commit_detail = commit_response.json()
            
            # Print commit header
            logger.info(f"Commit: {commit['sha'][:7]}")
            logger.info(f"Author: {commit['commit']['author']['name']}")
            logger.info(f"Date: {commit['commit']['author']['date']}")
            logger.info(f"Message: {commit['commit']['message']}")
            
            content += f"Commit: {commit['sha'][:7]}"
            content += f"Author: {commit['commit']['author']['name']}"
            content += f"Date: {commit['commit']['author']['date']}"
            content += f"Message: {commit['commit']['message']}"
            
            logger.info("\nChanges:")
            
            # Process each file in the commit
            for file in commit_detail['files']:
                logger.info(f"\nFile: {file['filename']}")
                logger.info(f"Status: {file['status']}")
                logger.info(f"Changes: +{file['additions']} -{file['deletions']}")

                content += f"\nFile: {file['filename']}"
                content += f"Status: {file['status']}"
                content += f"Changes: +{file['additions']} -{file['deletions']}"
                
                if 'patch' in file:
                    # Format and print the patch data
                    logger.info("\nDiff:")
                    patch_lines = file['patch'].split('\n')
                    for line in patch_lines:
                        if line.startswith('+'):
                            logger.info(f"\033[92m{line}\033[0m")  # Green for additions
                        elif line.startswith('-'):
                            logger.info(f"\033[91m{line}\033[0m")  # Red for deletions
                        else:
                            logger.info(line)
                        content += line
                else:
                    logger.info("No diff available (file might be binary or too large)")
                
            logger.info("\n" + "="*80 + "\n")

        logger.info(f"Content before: {len(content)}")
        res = {
            "status" : True,
            "content" : content
        }
        return res
    
    except Exception as e:
        logger.info(f"Error: {e}")
        res = {
            "status" : False,
            "content" : str(e)
        }
        return res
    

def get_file_content(repo_owner: str, repo_name: str, file_path: str, commit_sha: str, headers: Dict) -> str:
    """Helper function to get file content at a specific commit"""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{quote(file_path)}'
    response = requests.get(url, headers=headers, params={'ref': commit_sha})
    
    if response.status_code == 200:
        content = response.json()
        if content.get('encoding') == 'base64':
            return base64.b64decode(content['content']).decode('utf-8')
    return None

def clone_repo_and_get_commits(repo_url, dest_folder, params):
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
            token_url = f"https://{params['contributor']}:{params['token']}@github.com/{params['username']}/{params['repo_name']}"
            logger.info(f"Token URL: {token_url}")
            repo.remotes.origin.set_url(token_url)
            # repo.git.checkout('test')  # Switch to the test branch
            repo.remotes.origin.fetch()
            repo.remotes.origin.pull()
            logger.info("Pull complete on test branch")
        except Exception as e:
            logger.info(f"Error: Pulling {e}")
    

    if repo is None:
        repo = git.Repo(dest_folder)

    # Get the commits from the last week
    last_week = datetime.datetime.now() - datetime.timedelta(weeks=1)
    commits = list(repo.iter_commits(since=last_week.isoformat()))
    logger.info(f"Last Week: {last_week}")
    logger.info(f"Commits: {len(commits)}")

    # Print the commit details
    if not commits:
        logger.info("No commits found in the last week")
        content = None
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
    unwanted_folders = ['assets']
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

# Chunking Prompt into LLM context length
def manage_prompt_size(content, model):
    gemini_max_tokens = 1000000
    gpt_max_tokens = 32000
    llama3_max_tokens = 128000
    claude_sonet_max_tokens = 200000
    prompts = []
    total_tokens = None
    content_chunks = None
    chunks = None
    if model is None:
        model = "gemini"
    
    logger.info(f"Content: {content}")
    try:
        if model.lower() == "gemini":
            max_tokens = gemini_max_tokens
            total_tokens = gemini.count_tokens(content).total_tokens
            logger.info(f"Gemini Token count: {total_tokens}")

        elif model.lower() == "gpt-3.5-turbo":
            max_tokens = gpt_max_tokens
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            total_tokens = len(encoding.encode(content))
            logger.info(f"GPT Token count: {total_tokens}")

        elif model.lower() == "llama":
            max_tokens = llama3_max_tokens
        elif model.lower() == "claude":
            max_tokens = claude_sonet_max_tokens
        else:
            raise ValueError(f"Unsupported model: {model}")

        
        logger.info(f"Total Tokens: {total_tokens}")
        if total_tokens > max_tokens:
            chunk_size = max_tokens
            chunks = [content[i:i+chunk_size] if i+chunk_size <= len(content) else content[i:] for i in range(0, len(content), chunk_size)]
            prompts = [sprint_prompt.replace("context_here", chunk) for chunk in chunks]
            logger.info(f"Divided into {len(prompts)} prompts")
        else:
            prompts.append(sprint_prompt.replace("context_here", content))
            if model == "gpt-3.5-turbo":
                total_tokens = len(encoding.encode(content))
            else:
                logger.info(f"Prompt: {gemini.count_tokens(prompts[-1])}")
        content_chunks = {
            "prompt_chunks" : prompts,
            "chunks" : chunks
        }
        return content_chunks  
    except Exception as e:
        logger.info(f"Error - Manage Prompt Size: {e}")
        return content_chunks
        

def analyze_repo(params, content):
    try:
        username = params['username']
        repo_name = params['repo_name']
        contributor = params['contributor']
        token = params['token']
        model = params['model']
        if model is None:
            model = "gemini"
        # content = ''
        # with open(output_file, 'r', encoding='utf-8') as file:
        #     content = file.read()
        
        content_chunks = manage_prompt_size(content, model)

        if content_chunks is None: raise ValueError("Invalid chunks error")

        # Change this line
        logger.info(f"Divided into {len(content_chunks['prompt_chunks'])} Prompts for {model} Model")
        reports = []

        if model == 'gpt-3.5-turbo':
            logger.info("Analyzing using GPT 3.5")
            try:
                # Change this line
                for prompt in content_chunks['prompt_chunks']:
                    chunk_index = 0
                    messages = [
                        {"role": "system", "content": base_prompt},
                        {"role": "user", "content": content_chunks['chunks'][chunk_index] if content_chunks['chunks'] else prompt}
                    ]
                    chunk_index += 1
                    logger.info("Using GPT 3.5 Turbo Model")
                    try:
                        response = gpt.chat.completions.create(
                            model=model,
                            messages=messages
                        )
                        report = response.choices[0].message.content
                        reports.append(report)
                        logger.info(f"Response Text: {report}")
                        # logger.info(f"Usage : {response.usage}")
                    except Exception as e:
                        logger.error(f"Error generating content with GPT-3.5: {e}")
                    logger.info("Sleeping for 10 seconds")
                    time.sleep(10)  # Add a small delay between requests
                
                if len(reports) > 1:
                    combined_report = "\n\n".join(reports)
                    response = gpt.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that merges multiple reports of the same format into a single one."},
                            {"role": "user", "content": combined_report }
                        ]
                    )
                    reports.append(response.choices[0].message.content)
            except ValueError as e:
                logger.error(f"Error: {e}")
            except Exception as e:
                logger.error(f"Error in GPT-3.5-turbo analysis: {e}")

        elif model == str('gemini'):
            logger.info("Analyzing using Gemini 1.5 pro")
            for prompt in content_chunks['prompt_chunks']:
                logger.info("Generating Response")
                response = gemini.generate_content(prompt) # If model geminni
                report = response.text
                reports.append(report)
                logger.info(f"Output: {gemini.count_tokens(response.text)}")
                logger.info("Wait for 30 seconds")
                time.sleep(30)  # Wait for 30 seconds before generating the next report
            if len(reports) > 1:
                combined_report = "\n\n".join(reports)
                response = gemini.generate_content(f"Combine these Contents and Generate a single one in HTML format ```content : {combined_report}```")
                reports.append(response.text)
                logger.info(f"Combined Output: {gemini.count_tokens(response.text)}")
            logger.info("Responses generated successfully")
        
        # elif model == "claude-3.5-sonet":
        #     pass

        # elif model == "llama3":
        #     pass

        else:
            logger.info("Select a Valid LLM to Proceed")   
        return content_chunks['prompt_chunks'] , reports[-1] if reports else "No report generated"
    except Exception as e:
        logger.error(f"Error in analyze_repo: {e}")
        return None, f"Error occurred during analysis: {e}"

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


@api_view(['GET'])
def index(request):
    cronjobs = CronJob.objects.all()
    return render(request, '../templates/git_app/index.html', {"cronjobs": cronjobs})


        
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
    response = gemini.generate_content(prompt)
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
    model = request.POST.get('model')
    logger.info(f"Model: {model}")

    if not username or not repo_name:
        raise ValueError("Username and repository name are required")

    params = {
        'username': username,
        'repo_name': repo_name,
        'contributor': contributor,
        'token': token,
        'emails': emails,
        'model' : model
    }

    try:
        repo_url = f"https://{token}@github.com/{username}/{repo_name}.git" if token else f"https://github.com/{username}/{repo_name}.git"
        logger.info(f"Analyzing repo: {repo_url}")
            
        dest_folder = f"repositories/{username}/{repo_name}"

        status, message = test_repo_access(username, repo_name, token)
        if status != 200:
            return JsonResponse({"message": message}, status=status)
        # content = clone_repo_and_get_commits(repo_url, dest_folder, params)
        content = get_code_changes(username, repo_name, token)
        
        # Check if content is valid
        if content['status'] is False:
            logger.info(content['content'])
            if '401' in content['content']:
                return JsonResponse({"message": "Invalid Token or Permissions not granted"}, status=401)
            return JsonResponse({"message": content['content']}, status=400)

        # Ensure content is not empty
        content = content['content']
        if not content:
            logger.error("No content retrieved from get_code_changes.")
            return JsonResponse({"message": "No content retrieved from the repository, No Commits Found"}, status=400)

        logger.info(f"Content: {len(content)}")
        prompts, response = analyze_repo(params, content)
        if prompts is None:
            logger.info("Prompts are None")
            return JsonResponse({"message":"Internal Server Error"}, status=500)

        project, created_project = Project.objects.get_or_create(
            name = f"{username}/{repo_name}",
            defaults={
                'username': username,
                'repo_name': repo_name,
                'url': repo_url,
                'contributor': contributor,
                'emails': emails,
                'repository_token': token,
                'prompts': prompts, 
                'frequency': 'Weekly'
            }
        )
        if created_project:
            project.save()
        else:
            project.username = username
            project.repo_name = repo_name
            project.url = repo_url
            project.contributor = contributor
            project.emails = emails
            project.repository_token = token
            project.prompts = prompts
            project.save()


        report = Report.objects.create(
            name=f"{username}/{repo_name}",
            output=response,
            project=project
        ).save()

        logger.info(f"New report created for project '{repo_name}': {response}")

        send_brevo_mail(subject=f"{repo_name}", 
                        html_content=response, 
                        emails=emails)
        
        job, created_job = CronJob.objects.get_or_create(
            name=f"{username}/{repo_name}",
            defaults={
                'username': username,
                'repo_name': repo_name,
                'contributor': contributor,
                'token': token,
                'emails': emails,
                'project': project
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
            job.project = project
            job.save()
            logger.info("CronJob already Exists for this Repository, Details Updated")
        
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 401:
            logger.info("401 Unauthorized error encountered. Please check your credentials.")
            return JsonResponse({"status": "Failed", "error": "401 Unauthorized"}, status=401)
    except Exception as e:
        logger.info(f"Error: {e}")
        return JsonResponse({"status": "Failed", "error": str(e)}, status=500)
    
    return JsonResponse({"status": "success"})











