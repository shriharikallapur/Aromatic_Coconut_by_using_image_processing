from django.urls import path
from django.contrib import admin
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('camera',views.camera,name='camera'),
    path('accurate',views.accurate,name='accurate'),
    path('about',views.about,name='about'),
]