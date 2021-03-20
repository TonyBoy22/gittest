import socket

PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'

# Créé l'objet socket. Instanciation de la classe.
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode(FORMAT) # On encode dans le format utf-8 le message à envoyer
    msg_length = len(message) # Longueur du message à envoyer
    # On encode le message puis la valeur de la longueur de ce message, qui doit être envoyée avant
    send_length = str(msg_length).encode(FORMAT)
    # On pad en ajoutant des espaces en format bytecode pour remplir le reste de l'espace qu'on a
    send_length += b' '*(HEADER - len(send_length))
    client.send(send_length)
    client.send(message)

send('Allo')