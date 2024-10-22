from django.contrib import admin
from .models import Report, Project, CronJob

# Register your models here.
admin.site.register(Report)
admin.site.register(Project)
admin.site.register(CronJob)