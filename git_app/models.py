from django.db import models


class Project(models.Model):
    FREQUENCY_CHOICES = [
        ('Weekly', 'Weekly'),
        ('Bi-Weekly', 'Bi-Weekly')
    ]
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    repo_name = models.CharField(max_length=255)
    url = models.URLField()
    contributor = models.CharField(max_length=255)
    emails = models.TextField(help_text="Comma-separated list of emails")
    repository_url = models.URLField()
    repository_token = models.CharField(max_length=255, null=True, blank=True)
    prompts = models.JSONField(default=list, blank=True, null=True)
    active = models.BooleanField(default=True)
    frequency = models.CharField(max_length=10, choices=FREQUENCY_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Report(models.Model):
    id = models.AutoField(primary_key=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='reports')
    name = models.CharField(max_length=255)
    output = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
    

class CronJob(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    repo_name = models.CharField(max_length=255)
    contributor = models.CharField(max_length=255)
    token = models.CharField(max_length=255)
    emails = models.CharField(max_length=255)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return self.username
    
class Error(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    message = models.CharField(max_length=255)
    context = models.TextField()