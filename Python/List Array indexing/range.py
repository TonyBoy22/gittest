'''
explore possibilities of arguments with the range() function
'''
import numpy as np

number_of_circles = 3
threshold = 30
lst = list(range(0,50))
indices = range(0,len(lst), int((len(lst) - 1)/(number_of_circles)))
# print('indices: ', indices)
# for i in indices:
#     print(i)
circles = np.array([lst[i] for i in indices])
print('circles: ', circles)
norm = np.linalg.norm([], axis=1)