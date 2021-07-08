'''
Tutorial on verifying if some coordinates are comprised in a certain range
A list with a few elements is screened by a for loop in the any() function

Always required to slice by converting the list into an array. Otherwise, cannot take a column or a row
'''

import math
import numpy as np

data = [['1', 1, 11], ['2', 2, 12], ['3', 3, 15]]
dummy_x = [[1, 2, 3, 4, 'testing1'], [11, 12, 13, 14, 'testing2'], [21, 22, 23, 24, 'testing3']]

dummy1, dummy2, dummy3 = list(zip(*([(x[:3], x[3], x[4]) for x in dummy_x])))
'''w
Expected result:
dummy1 == ([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
dummy2 == 
dummy3 == ('testing1', 'testing2', 'testing3')
'''
x_data = list(zip(*([(x[1:2]) for x in data])))
# x_array = np.array(zip([x[1:2] for x in data]))
# Not working because Zip element has a size of one and values are not numeric objects for Python
# print('first x_array element: ', x_array[0])

# Index search
list_to_search = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 4]]
# We start form 2n index and for first occurence of value 6
index_value = list_to_search.index([1, 2, 3], 1)
print('index_value: ', index_value)

