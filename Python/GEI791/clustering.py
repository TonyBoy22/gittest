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

# First, let's open a few images
path = glob.glob(r"C:\Users\antoi\Desktop\Github\gittest\Python\GEI791\baseDeDonneesImages\*.jpg")
# image_list = os.listdir(path)
# print(image_list)
images = np.array([np.array(Image.open(image)) for image in path])
print(images.shape)
print(type(images))

def plot_RGB():
    
    return None
# for image in path:
#     print('image: ', image)
#     im = Image.open(image)
#     im.show()
# def open