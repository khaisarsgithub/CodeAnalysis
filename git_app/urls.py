from django.urls import path
from git_app import views

urlpatterns = [
    path('weekly_report/', views.get_weekly_report, name='weekly_report'),
    path('', views.index, name='index'),
    path('send_message', views.send_message, name='send_message'),
]