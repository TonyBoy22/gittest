'''
Knn demo on labeled data
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st
from matplotlib import cm

# CSTES
SIZE = 1000
K_PPV = 1

def creer_hist2D(data):
    # list of colors
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    x = []
    y = []
    for classes in data:
        x.append(classes[:,0])
        y.append(classes[:,1])
    x = np.array(x)
    x = x.flatten()
    y = np.array(y)
    y = y.flatten()
    
    deltax = (np.max(x) - np.min(x))/10
    deltay = (np.max(y) - np.min(y))/10
    
    xmin = np.min(x) - deltax
    xmax = np.max(x) + deltax
    ymin = np.min(y) - deltay
    ymax = np.max(y) + deltay
    
    # Meshgrid
    # xx, yy = np.meshgrid(x, y[])
    
    # positions = np.vstack([xx.ravel(), yy.ravel()])
    # values = np.vstack([x, y])
    # kernel = st.gaussian_kde(values)
    # f = np.reshape(kernel(positions).T, xx.shape)
    
    # surf = ax.plot_surface(xx, yy, f, cmap='coolwarm', edgecolor='none')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    
    bin_size = 40
    
    
    
    # ax.hist2d(x, y, bins=[bin_size, bin_size], cmap='inferno')
    hist, xedges, yedges = np.histogram2d(x, y, bins=[bin_size, bin_size])
    
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = np.ones_like(0)
    dz = hist.ravel()
    
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
    # ax.view_init()
    
    
    return None

def afficher_erreur_classification(original_data, cls_data):
    # génère le vecteur d'erreur de classification
    vect_err = np.absolute(original_data - cls_data)
    vect_err = vect_err.astype(bool)
    # print('vect_err: ', vect_err)
    indexes = np.where(vect_err == True)
    # print('indexes: ', indexes)
    
    return indexes

# import labeled data in arrays
C1 = np.loadtxt('C1.txt')
C2 = np.loadtxt('C2.txt')
C3 = np.loadtxt('C3.txt')

data = [C1, C2, C3]

creer_hist2D(data)
# labels = []
# for i in range(len(data)):
#     class_label = np.ones((1000,1))*(i+1)
#     data[i] = np.concatenate((data[i], class_label), axis=1)
# data = np.array(data)
# # print('data: ', data)
# '''
# dans un premier temps, génère des points aléatoires dans l'espace
# '''
# # creation points aléatoires entre -10 et 10 pour ax des x et -10et 20 pour y
# donnee_test = np.transpose(np.array([20*np.random.random(SIZE)-10, \
#     30*np.random.random(SIZE)-10]))

# # 1-PPV avec les labels
# x, y, z = data.shape
# data = data.reshape(x*y, z)
# # flatten X_train
# X_train = data[:,:2]
# X_test = donnee_test
# y_labels = data[:,2]

# '''
# ensuite, dans un premier temps, tu peux faire un 1-PPV avec comme représentants de classes l'ensemble des points déjà classés
# '''
# # Creation classificateur
# # n_neighbors est le nombre k
# # metric est le type de distance entre les points. La liste est disponible ici:
# # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
# kNN = KNeighborsClassifier(n_neighbors = K_PPV, metric='minkowski')
# kNN.fit(X_train, y_labels)
# predictions = kNN.predict(X_test)

# fig, (ax1, ax2) = plt.subplots(2,1)
# ax1.scatter(data[:,0], data[:,1], c=data[:,2], cmap='viridis')
# ax2.scatter(donnee_test[:,0], donnee_test[:,1], c=predictions, cmap='viridis')


# '''
# ensuite fais un 1-mean sur chacune des classes
# '''
# # Ensuite, on applique le Kmeans sur les classes de base
# kmeans_classifier = KMeans(n_clusters=3)

# kmeans_classifier.fit(X_train)

# cluster_centers = kmeans_classifier.cluster_centers_
# cluster_labels = kmeans_classifier.labels_
# cluster_labels += 1

# averages = np.array([np.average(cluster_labels[0:1000]), np.average(cluster_labels[1000:2000]), \
#     np.average(cluster_labels[2000:3000])])

# print('averages: ', averages)

# sorted_averages = np.sort(averages)
# print('sorted_averages: ', sorted_averages)

# # Condition pour que l'étiquettage des classes du KMeans soit le même que nos données originales
# while any(averages != sorted_averages):
#     kmeans_classifier.fit(X_train)

#     cluster_centers = kmeans_classifier.cluster_centers_
#     cluster_labels = kmeans_classifier.labels_
#     cluster_labels += 1

#     averages = np.array([np.average(cluster_labels[0:1000]), np.average(cluster_labels[1000:2000]), \
#     np.average(cluster_labels[2000:3000])])

#     print('averages: ', averages)

#     sorted_averages = np.sort(averages)
#     print('sorted_averages: ', sorted_averages)

# figK, (axK1, axK2) = plt.subplots(2,1)
# axK1.scatter(X_train[:,0], X_train[:,1], s= 5.0,c=y_labels, cmap='viridis')
# axK2.scatter(X_train[:,0], X_train[:,1], s= 5.0, c=cluster_labels, cmap='viridis')

# indexes = afficher_erreur_classification(y_labels, cluster_labels)

# axK2.scatter(X_train[indexes, 0], X_train[indexes, 1], s=5.0, c='red')

# '''
# suivi d'un 1-PPV avec tes points aléatoires du début et tes nouveaux représentants de classes
# '''
# # KNN sur KMeans estimation
# # Preparer nouvelles données
# full_train = np.concatenate((X_train, donnee_test), axis=0)
# full_labels = np.concatenate((cluster_labels, predictions))


# kNN2 = KNeighborsClassifier(n_neighbors=K_PPV, metric='minkowski')
# kNN2.fit(X_train, cluster_labels)

# predictions_2 = kNN2.predict(donnee_test)

# figNN, (axNN1, axNN2) = plt.subplots(2,1)
# axNN1.scatter(X_train[:,0], X_train[:,1], c=cluster_labels, cmap='viridis')
# axNN2.scatter(donnee_test[:,0], donnee_test[:,1], c=predictions_2, cmap='viridis')

plt.tight_layout()
plt.show()