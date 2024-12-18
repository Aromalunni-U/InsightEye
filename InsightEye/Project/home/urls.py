from django.urls import path
from .views import *

urlpatterns = [
    path("",index,name="index"),
    path("code",code,name="code"),
    path("test",test,name="test"),
    path("camera",camera,name="camera"),


]
