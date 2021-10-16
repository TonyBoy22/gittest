'''
Question to figure out: Can you unpack a list with the asterisk
in the same way you can unpack a tuple?
'''

example_list = [1, 2, 3]

element_to_add = 5.0

merging_list = [*example_list, element_to_add]
print('merging_list: ', merging_list)