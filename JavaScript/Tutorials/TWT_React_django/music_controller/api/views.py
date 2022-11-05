from django.shortcuts import render
from rest_framework import generics

from .serializers import RoomSerializer
from .models import Room

# Create your views here. what is seen when clients get to an endpoint
# Endpoints are the leaves of the website. The last node at the end of an URL.

# Will return 
class RoomView(generics.CreateAPIView):
    queryset = Room.objects.all()
    serializer_class = RoomSerializer