# le serializer tourne une request depuis la database et sort les infos dans un .json
from rest_framework import serializers
# Façon de dire: depuis un fichier du meme dossier, importe...
from .models import Room

class RoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        # Le id field est généré automatiquement associé à Room
        # on pourrait aussi avoir fields = ('__all__')?
        fields = ('id', 'code', 'host', 'guestCanPause', 'votesToSkip', 'createdAt')