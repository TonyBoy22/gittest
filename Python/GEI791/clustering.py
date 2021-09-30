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

# for reproducibility
# random.seed(0)

# First, let's open a few images
path = glob.glob(r"C:\Users\antoi\Desktop\Github\gittest\Python\GEI791\baseDeDonneesImages\*.jpg")
image_folder = r"C:\Users\antoi\Desktop\Github\gittest\Python\GEI791\baseDeDonneesImages"

# Make a list to have a track
image_list = os.listdir(image_folder)

# List comprehension to filter folder for images only
image_list = [i for i in image_list if '.jpg' in i]

# Séparation à priori des classes?
# coast_list = [i for i in image_list if 'coast' in i]
# forest_list = [i for i in image_list if 'forest' in i]
# street_list = [i for i in image_list if 'street' in i]

print(image_list[0])
print(len(image_list))

images = np.array([np.array(Image.open(image)) for image in path])
print(images.shape)
print(type(images))


def plot_RGB(images, indexes=None):
    '''
    Function that creates the RGB plot for
    selected images
    :param images: array containing all images
    :param indexes: index for selecting images to
    display their RGB profile

    :return: None
    '''

    fig = plt.figure()
    ax = fig.add_subplot()

    if indexes is not None:
        images = images[indexes]
    else:
        pass


    R_average = np.sum(images[:,:,:,0], axis=(1,2))/(256*256)
    G_average = np.sum(images[:,:,:,1], axis=(1,2))/(256*256)
    B_average = np.sum(images[:,:,:,2], axis=(1,2))/(256*256)
    # print(R_average)
    # print(G_average)
    # print(B_average)
    ax.plot(range(np.size(images, 0)), R_average, c='red')
    ax.plot(range(np.size(images, 0)), G_average, c='green')
    ax.plot(range(np.size(images, 0)), B_average, c='blue')
    ax.set_title('image RGB profil')
    plt.show()
    return None


def random_image_selector(image_list, number):
    '''
    Generates a list of number between 0 and size of folder
    then displays the image of the corresponding indexes
    from the image_list
    '''
    indexes_list = np.random.randint(low=0, high=np.size(image_list, 0), size=number)

    # Protection against duplicates
    unique_indexes = np.unique(indexes_list)
    while np.size(unique_indexes) != number:
        size_diff = number - np.size(unique_indexes)
        extra_indexes = np.random.randint(low=0, high=np.size(image_list, 0), size=size_diff)
        new_array = np.append(unique_indexes, extra_indexes)
        unique_indexes = np.unique(new_array)

    # assign unique values to be our indexes list
    indexes_list = unique_indexes
    indexes_list = np.sort(indexes_list)
    return indexes_list


def images_display(image_list, indexes=None):
    '''
    function to display the images from appropriates indexes

    :param image_list: list of image to display
    :param indexes: generated indexes
    :return: None
    '''
    if indexes is not None:
        for index in indexes:
            im = Image.open(image_folder + '\\' + image_list[index])
            im.show()
    else:
        return 'Please specify indexes'


def visualize_encodings(images):
    '''
    Displays selected images in 3 encodings
    RGB -> YUV -> Lab

    :param
    :return:
    '''
    # Test on one image
    image_rgb = images[0]

    # we know it's RGB at this stage
    cv2.imshow('RGB', image_rgb)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


    # Inversion of axes because for cv2 RGB -> BGR
    # image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    # Conversion to YUV
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YCrCb)
    cv2.imshow('YUV', image_yuv)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    # conversion to Lab
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2LAB)
    cv2.imshow('Lab', image_lab)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return None


def RGB2YUV(images):
    # test sur une seule image

    return None


def apply_kmeans_clustering(images, k=5, encoding='RGB', indexes=None):
    '''
    Takes the image and apply kmeans clustering for specified
    :param images:      images array
    :param k:           number of cluster per image
    :param encoding:    format of images
    :param indexes:     indexes to select images
    :return:            None
    '''
    # Reshaping in 2D array
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


# index_list = random_image_selector(images, 5)
# print(index_list)

# images_display(image_list, index_list)

# plot_RGB(images, index_list)

# visualize_encodings(images)

apply_kmeans_clustering(images, 6)