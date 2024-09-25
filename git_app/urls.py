from django.urls import path, include
from git_app import views

urlpatterns = [
    path('weekly_report/', views.get_weekly_report, name='weekly_report'),
    path('', views.index, name='index'),
]