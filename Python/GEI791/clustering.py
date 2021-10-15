'''
Script for the kNN and image visualization

'''

import matplotlib.pyplot as plt
import numpy as np
import os
# import sys
import glob
from sklearn.cluster import KMeans
# from sklearn.datasets.samples_generator import make_blobs
from PIL import Image
import cv2

# pour reproductibilité
# random.seed(0)

# D'abord, faire une liste de toutes les images par leur titre
path = glob.glob(r"C:\Users\antoi\Desktop\Github\gittest\Python\GEI791\baseDeDonneesImages\*.jpg")
image_folder = r"C:\Users\antoi\Desktop\Github\gittest\Python\GEI791\baseDeDonneesImages"
image_list = os.listdir(image_folder)

# Filtrer pour juste garder les images du dossier
image_list = [i for i in image_list if '.jpg' in i]

# Créer un array qui contient toutes les images
# Dimensions [980, 256, 256, 3]
# Valeurs    [# image, hauteur, largeur, RGB]
images = np.array([np.array(Image.open(image)) for image in path])


# def plot_RGB(images, indexes=None):
#     '''
#     Function that creates the RGB plot for
#     selected images
#     :param images: array containing all images
#     :param indexes: index for selecting images to
#     display their RGB profile
#
#     :return: None
#     '''
#
#     fig = plt.figure()
#     ax = fig.add_subplot()
#
#     if indexes is not None:
#         images = images[indexes]
#     else:
#         pass
#
#
#     R_average = np.sum(images[:,:,:,0], axis=(1,2))/(256*256)
#     G_average = np.sum(images[:,:,:,1], axis=(1,2))/(256*256)
#     B_average = np.sum(images[:,:,:,2], axis=(1,2))/(256*256)
#     # print(R_average)
#     # print(G_average)
#     # print(B_average)
#     ax.plot(range(np.size(images, 0)), R_average, c='red')
#     ax.plot(range(np.size(images, 0)), G_average, c='green')
#     ax.plot(range(np.size(images, 0)), B_average, c='blue')
#     ax.set_title('image RGB profil')
#     plt.show()
#     return None


def histogrammes(images, index):
    '''
    Takes images array and an index to pick the
    appropriate image
    images format: (number of image, Width, Height, RGB channel)
    index: int or list of int
    '''

    fig = plt.figure()
    ax = fig.add_subplot()

    # Number of bins per color
    n_bins = 256

    # image selection
    image = images[index, :, :, :]

    # A list per color channel
    pixel_values = np.zeros((3,256))

    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            pixels = image[i,j,:]
            pixel_values[0, pixels[0]] += 1
            pixel_values[1, pixels[1]] += 1
            pixel_values[2, pixels[2]] += 1
    print('pixel values: ', pixel_values[0,:])
    print('sum pixels: ', np.sum(pixel_values[0,:]))
    # ax.hist(pixel_values[0,:], bins = range(0,n_bins))
    # ax.hist(pixel_values[0, :], bins=n_bins+1)
    ax.scatter(range(n_bins), pixel_values[0,:], c='red')
    ax.scatter(range(n_bins), pixel_values[1,:], c='green')
    ax.scatter(range(n_bins), pixel_values[2,:], c='blue')
    ax.set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
    # ajouter le titre de la photo observée dans le titre de l'histograme
    image_name = image_list[index]
    ax.set_title(f'histogramme de {image_name}')
    plt.show()

    return None


def random_image_selector(image_list, number):
    '''
    Génère une liste d'indexes pour choisir des images au hasard dans la liste
    image_list: liste de strings de 980 items
    number: int
    '''
    indexes_list = np.random.randint(low=0, high=np.size(image_list, 0), size=number)

    # Protection contre les doublons
    unique_indexes = np.unique(indexes_list)
    while np.size(unique_indexes) != number:
        size_diff = number - np.size(unique_indexes)
        extra_indexes = np.random.randint(low=0, high=np.size(image_list, 0), size=size_diff)
        new_array = np.append(unique_indexes, extra_indexes)
        unique_indexes = np.unique(new_array)

    # s'assure que les indexes sont des valeurs uniques
    indexes_list = unique_indexes
    indexes_list = np.sort(indexes_list)
    return indexes_list


def images_display(image_list, indexes=None):
    '''
    fonction pour afficher les images correspondant aux indices

    :param image_list: liste d'image à montrer
    :param indexes: indices de la liste d'image
    :return: None
    '''
    if type(indexes) == int:
        indexes = [indexes]

    if indexes is not None:
        for index in indexes:
            im = Image.open(image_folder + '\\' + image_list[index])
            im.show()
    else:
        return 'veuillez spécifier un indice'


def kmeans_clustering(images, k=5, encoding='RGB', indexes=None):
    '''
    Prend une image et applique le kmeans pour segmenter les images
    :param images:      images array
    :param k:           number of cluster per image
    :param encoding:    format of images
    :param indexes:     indexes to select images
    :return:            None
    '''
    # Reformate en array à 2D
    image = images[0]
    x, y, z = image.shape
    assert z == 3
    image_2d = image.reshape(x*y, z)
    print('image shape: ', image_2d.shape)

    kmeans_cluster = KMeans(n_clusters=k)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_centers = cluster_centers.astype(int)
    cluster_labels = kmeans_cluster.labels_
    print('cluster_centers: ', cluster_centers, cluster_centers)
    print('cluster_labels: ', cluster_labels)
    # Now display Kmeans clusters

    plt.figure()
    plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))
    plt.show()
    return None
# ============= Script principal ======================== #

# Appeler ici les fonctions


images_display(image_list, 6)

# kmeans_clustering(images, 6)

histogrammes(images, 6)