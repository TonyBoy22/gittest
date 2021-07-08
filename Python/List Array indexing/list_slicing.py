'''
List filtering and slicing with lambda, maps, list comprehension and array conversion

Tutorial sources:
https://medium.com/swlh/lambda-vs-list-comprehension-6f0c0c3ea717

https://towardsdatascience.com/speeding-up-python-code-fast-filtering-and-slow-loops-8e11a09a9c2f
'''

# waypoint_list.npy --> list of arrays

import numpy as np
import itertools
import timeit
from operator import itemgetter
# import numba (for boolean indexing)

# w_list = np.load('waypoint_list.npy')

# Example from jedwards stackoverflow
class Flexlist(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return [self[k] for k in keys]

aList = Flexlist(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
myIndices = [0, 3, 4]
vals = aList[myIndices]

print(vals)  # ['a', 'd', 'e']    


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

# a = break_true()
# print(a)

# ========================================================= List Comprehensions ================================================
list_example = [['Simons', 42, 1.0],['Bilodeau', 15, 2.0],['Babayaga', 80, 7.0],['boutentrain', 7, 7.5]]
indices = [0,2]


print(itemgetter(*indices)(list_example))
'''
If we don't unpack the container list indices, the function itemgetter will try
to read a list object instead of the values it contains and cannot work
'''
list_example.append(['party goer', 0.75])

new_list = []

# ======================================================== Lambda map combo =====================================================

list_example2 = [[1, 1.5],[2, 2.5],[3, 3.5],[4, 4.5],[5, 5.5]]

lst = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

indices = [sub_element for element in list_example2 for sub_element in element if sub_element % 2 == 0]
print('indices first try: ', indices)
indices = [sub_ele for element in list_example2 for sub_ele in element if type(sub_ele) == int]
print('indices second try: ', indices)
