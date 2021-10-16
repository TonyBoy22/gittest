'''
Knn demo on labeled data
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# CSTES
SIZE = 1000
K_PPV = 1

# import labeled data in arrays
C1 = np.loadtxt('C1.txt')
C2 = np.loadtxt('C2.txt')
C3 = np.loadtxt('C3.txt')

data = [C1, C2, C3]

labels = []
for i in range(len(data)):
    class_label = np.ones((1000,1))*(i+1)
    data[i] = np.concatenate((data[i], class_label), axis=1)
data = np.array(data)
# print('data: ', data)
'''
dans un premier temps, génère des points aléatoires dans l'espace
'''
# creation points aléatoires entre -10 et 10 pour ax des x et -10et 20 pour y
donnee_test = np.transpose(np.array([20*np.random.random(SIZE)-10, \
    30*np.random.random(SIZE)-10]))

# 1-PPV avec les labels
x, y, z = data.shape
data = data.reshape(x*y, z)
# flatten X_train
X_train = data[:,:2]
X_test = donnee_test
y_labels = data[:,2]

'''
ensuite, dans un premier temps, tu peux faire un 1-PPV avec comme représentants de classes l'ensemble des points déjà classés
'''
# Creation classificateur
# n_neighbors est le nombre k
# metric est le type de distance entre les points. La liste est disponible ici:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
kNN = KNeighborsClassifier(n_neighbors = K_PPV, metric='minkowski')
kNN.fit(X_train, y_labels)
predictions = kNN.predict(X_test)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(data[:,0], data[:,1], c=data[:,2], cmap='viridis')
ax2.scatter(donnee_test[:,0], donnee_test[:,1], c=predictions, cmap='viridis')


'''
ensuite fais un 1-mean sur chacune des classes
'''
# Ensuite, on applique le Kmeans sur les classes de base
kmeans_classifier = KMeans(n_clusters=3)

kmeans_classifier.fit(X_train)

cluster_centers = kmeans_classifier.cluster_centers_
cluster_labels = kmeans_classifier.labels_
cluster_labels += 1

averages = np.array([np.average(cluster_labels[0:1000]), np.average(cluster_labels[1000:2000]), \
    np.average(cluster_labels[2000:3000])])

print('averages: ', averages)

sorted_averages = np.sort(averages)
print('sorted_averages: ', sorted_averages)

while any(averages != sorted_averages):
    kmeans_classifier.fit(X_train)

    cluster_centers = kmeans_classifier.cluster_centers_
    cluster_labels = kmeans_classifier.labels_
    cluster_labels += 1

    averages = np.array([np.average(cluster_labels[0:1000]), np.average(cluster_labels[1000:2000]), \
    np.average(cluster_labels[2000:3000])])

    print('averages: ', averages)

    sorted_averages = np.sort(averages)
    print('sorted_averages: ', sorted_averages)

figK, (axK1, axK2) = plt.subplots(2,1)
axK1.scatter(X_train[:,0], X_train[:,1], c=y_labels, cmap='viridis')
axK2.scatter(X_train[:,0], X_train[:,1], c=cluster_labels, cmap='viridis')


'''
suivi d'un 1-PPV avec tes points aléatoires du début et tes nouveaux représentants de classes
'''
# KNN sur KMeans estimation
# Preparer nouvelles données
full_train = np.concatenate((X_train, donnee_test), axis=0)
full_labels = np.concatenate((cluster_labels, predictions))


kNN2 = KNeighborsClassifier(n_neighbors=K_PPV, metric='minkowski')
kNN2.fit(X_train, cluster_labels)

predictions_2 = kNN2.predict(donnee_test)

figNN, (axNN1, axNN2) = plt.subplots(2,1)
axNN1.scatter(X_train[:,0], X_train[:,1], c=cluster_labels, cmap='viridis')
axNN2.scatter(donnee_test[:,0], donnee_test[:,1], c=predictions_2, cmap='viridis')

plt.tight_layout()
plt.show()