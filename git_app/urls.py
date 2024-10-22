from django.urls import path
from git_app import views

urlpatterns = [
    path('weekly_report/', views.get_weekly_report, name='weekly_report'),
    # path('<str:username>', views.index, name='index'),
    path('', views.index, name='home'),
    # path('user_login', views.user_login, name='user_login'),
    # path('user_logout/', views.user_logout, name='user_logout'),
    path('send_message', views.send_message, name='send_message'),
]
