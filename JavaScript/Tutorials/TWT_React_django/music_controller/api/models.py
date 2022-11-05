from django.db import models
import string
import random

def GenerateUniqueCode():
    length = 6
    
    while True:
        code = ''.join(random.choices(string.ascii_uppercase))
        if Room.objects.filter(code=code).count() == 0:
            break
    return code

# Create your models here. Les models sont une couche d'abstraction pour construire un bloc d'information similaire à rentrer dans un base de données. Créer une classe à partir
# de l'interface models

# Put most of the logic in the model
class Room(models.Model):
    code = models.CharField(max_length=8, default="", unique=True) # a bunch of characters
    host = models.CharField(max_length=50, unique=True)
    guestCanPause = models.BooleanField(null=False, default=False)
    votesToSkip = models.IntegerField(null=False, default=1)
    createdAt = models.DateTimeField(auto_now_add=True)
    