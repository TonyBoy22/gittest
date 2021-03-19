# waypoint_list.npy --> list of arrays

import numpy as np
import itertools

w_list = np.load('waypoint_list.npy')

def list_format():
    length = len(w_list)
    element_type = type(w_list[0])
    list_specs = {length: length,
                  el_type: element_type}
    return list_specs

def break_true():
    # Est-ce que return data pendant la boucle va arrêter la fonction?
    a = 0
    while True:
        a += 1
        if a >= 10:
            return a

# Turn it into a list of just a few of its columns
# 1. List --> Array --> List
# or Array --> List if already Array
# shortened_list = w_list[:,0:3]
# print(shortened_list)

a = break_true()
print(a)