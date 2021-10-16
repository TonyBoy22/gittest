'''
uses of list.sort(key=lambda x:) to sort elements in a list in more complex cases
'''

# Case of a 2D list
example_list = [[],[]]

# Case of a tuple containing a 2D list to be sorted
example_tuple = [(1, [[0.9, 42.0],[0.5, -9.0]]), (2, [[0.5, 81.0],[55.5, -71.5]])]
print('example_tuple: ', example_tuple)
example_tuple.sort(key=lambda x: x[1][0][1])
print('example_tuple: ', example_tuple)