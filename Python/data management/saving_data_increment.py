'''
Impossible to append data to a numpy file
with csv?
'''

import numpy as np
import csv
import pandas as pd

FILENAME = 'test_np_file.npy'
CSV_FILE = 'test.csv'

# def create_np_file(FILENAME):
#     np.savetxt(FILENAME, np.array(['test file']), fmt='%s')
#     return None

# def save_data_to_np(np_file, data):
#     '''
#     takes the np file and add data
#     '''
#     np.savetxt(np_file, np.array(['infos on the episode']), fmt='%s', newline='\n')
#     np.save(np_file, data, allow_pickle=True)
#     return None

# create_np_file(FILENAME)
# # with 
# # for i in range(5):
# #     save_data_to_np(FILENAME, np.array([1, 2, i]))

# data = np.load(FILENAME, allow_pickle=True)
# print('data: ', data)

# =================================== CSV ================================================

# creating a csv file
header = ['number of episode', 'actual speed', 'predicted speed']
data1 = [[0, [5, 5, 6], [3, 5, 6]],
         [1, [5, 6, 8], [3, 6, 8]]]
with open(CSV_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # add data
    writer.writerow(header)
    for data in data1:
        writer.writerow(data)

# with open(CSV_FILE, 'r', newline='') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         for element in row:
#             print('element: ', element)
#             print('element type: ', type(element))
#             print('element length: ', len(element))

df = pd.read_csv('test.csv')
print(df)
# picking element 
test_list = df.iloc[0,1]
print(test_list)
# removing non numerical character
test_list = [i for i in test_list if i.isnumeric()]
# 
test_list = list(map(np.float64, test_list))
print(test_list)