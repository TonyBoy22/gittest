# stores all the url local to this app
from django.urls import path
from .views import RoomView

urlpatterns = [
    path('home', RoomView.as_view()),
]