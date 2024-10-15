from django.db import models

# Create your models here.
# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class GitHubRepo(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='repos')
    name = models.CharField(max_length=255)
    url = models.URLField()
    description = models.TextField(null=True, blank=True)
    language = models.CharField(max_length=100, null=True, blank=True)
    stars = models.IntegerField(default=0)
    forks = models.IntegerField(default=0)
    watchers = models.IntegerField(default=0)
    open_issues = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name

class Project(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='projects')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Report(models.Model):
    FREQUENCY_CHOICES = [
        ('Weekly', 'Weekly'),
        ('Bi-Weekly', 'Bi-Weekly')
    ]
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    name = models.CharField(max_length=255)
    emails = models.TextField(help_text="Comma-separated list of emails")
    repository_url = models.URLField()
    repository_token = models.CharField(max_length=255, null=True, blank=True)
    prompt = models.TextField()
    active = models.BooleanField(default=True)
    frequency = models.CharField(max_length=10, choices=FREQUENCY_CHOICES)
    output = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
    

class CronJob(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=255)
    repo_name = models.CharField(max_length=255)
    contributor = models.CharField(max_length=255)
    token = models.CharField(max_length=255)
    emails = models.CharField(max_length=255)
    
    