from django.contrib import admin
from .models import Report, GitHubRepo, Project

# Register your models here.
admin.site.register(Report)
admin.site.register(Project)
admin.site.register(GitHubRepo)