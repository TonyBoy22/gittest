'''
Question to figure out: Can you unpack a list with the asterisk
in the same way you can unpack a tuple?
'''


# example_list = [1, 2, 3]

# element_to_add = 5.0

# merging_list = [*example_list, element_to_add]
# print('merging_list: ', merging_list)

# ====================================== 
# Converting list of lists in one list
# ======================================


example_list = [[1.005, 6.784], [9], [3.87, 6]]
# merged_list = [f for i in example_list]
merged_list = []
merged_list = [merged_list + i for i in example_list]

print('merged_list: ', merged_list)