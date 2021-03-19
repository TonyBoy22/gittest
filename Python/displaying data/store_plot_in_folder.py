import matplotlib.pyplot as plt
import numpy as np
import os

# En ordre, les instructions sont:
# Créer le nom du folder que l'on veut avoir. Commande adaptée pour fonctionner
# peu importe le nom du 
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

# Créer le folder du même nom
if not os.path.exists(CONTROLLER_OUTPUT_FOLDER):
    os.makedirs(CONTROLLER_OUTPUT_FOLDER)

# Faire des plots dans des figures avec l'approche objet API
for i in range(5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(8,100))

    graph_name = f'graph_{i}'
    os.path.join(CONTROLLER_OUTPUT_FOLDER, graph_name) # Joindre le chemin jusqu'au dossier et le nom du fichier à sauvegarder
    fig.savefig(os.path.join(CONTROLLER_OUTPUT_FOLDER, graph_name)) # Nom complet du path incluant le nom du fichier que l'on veut sauvegarder
    # Juste mettre le nom du fichier va faire sauvegarder dans le dossier de travail
