'''
Testing functions to reorganize elements in a list
'''

from operator import itemgetter, attrgetter

list_2D = [['Simons', 8.5, 50],['Equipeur', 6.5, 30],['Yellow', 3.2, 45]]

'''# First will sort items by 2nd elements, and for items with same value of 2nd element
# will sort by the 3rd value
list_sorted = sorted(list_2D, key=itemgetter(1,2))
'''




list_2D.sort(key=lambda x: x[1])
print('list_2D: ', list_2D)


