import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

np.random.seed(0)
'''
Créer des plots et les sauvegarder dans un dossier out
'''
# En ordre, les instructions sont:
# Créer le nom du folder que l'on veut avoir. Commande adaptée pour fonctionner
# peu importe le nom du 
# CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
#                            '/controller_output/'

# # Créer le folder du même nom
# if not os.path.exists(CONTROLLER_OUTPUT_FOLDER):
#     os.makedirs(CONTROLLER_OUTPUT_FOLDER)

# # Faire des plots dans des figures avec l'approche objet API
# for i in range(5):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(range(8,100))

#     graph_name = f'graph_{i}'
#     os.path.join(CONTROLLER_OUTPUT_FOLDER, graph_name) # Joindre le chemin jusqu'au dossier et le nom du fichier à sauvegarder
#     fig.savefig(os.path.join(CONTROLLER_OUTPUT_FOLDER, graph_name)) # Nom complet du path incluant le nom du fichier que l'on veut sauvegarder
#     # Juste mettre le nom du fichier va faire sauvegarder dans le dossier de travail


'''
Créer un histogramme 3D pour 2 grappes de points
'''
fig = plt.figure()
ax = fig.add_subplot()

x = np.random.normal(1.0, scale=1.0, size=10000)
y = np.random.normal(1.0, scale=1.0, size=10000)
ax.hist2d(x, y, bins=[100,100], cmap='YlGn')

x2 = np.random.normal(5.0, scale=1.0, size=10000)
y2 = np.random.normal(5.0, scale=1.0, size=10000)
ax.hist2d(x2, y2, bins = [100, 100], cmap='plasma')
hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.8 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()