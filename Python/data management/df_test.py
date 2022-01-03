import pandas as pd
import numpy as np
import csv

CSV_NAME = 'test.csv'

# ============================ CSV module ============================================
# with open(CSV_NAME, 'a', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(10):
#         actual_speed = [[i, i*2, i*3, i*4],[i*1.5, i*2.5, i*3.5]]
#         preds = [i, i/2, i/3, i/4]
#         liste = [i, actual_speed, preds]
#         writer.writerow(liste)
#     print(writer)

# ============================= DataFrame appending in for loop ======================
# df = pd.DataFrame()
# HEADER = False
# for i in range(10):
#     if i == 0:
#         HEADER = True
#     else:
#         HEADER = False
#     # the_dict = {'number of episode':i, 'actual speed':30-i, 'predicted_speed':20+i/5}
#     # df1 = pd.DataFrame(list(the_dict.items()), columns=[])
#     list_like = [[i, 30-i, 20+i/2]]
#     df1 = pd.DataFrame(data = list_like, columns=['number of episode', 'actual speed', 'predicted speed'])
#     df1.to_csv(CSV_NAME, index=False, mode='a', header=HEADER)
    
# ============================ DataFrames with data unequal length ====================

x = [0, 1, 2, 3, 4, 5]
y = [4, 8, 9, 1, 3, 4, 5, 6, 9, 7]

df = pd.DataFrame(data=[x, y]).T
print(df)
df = df.fillna(-1)
# Convert NaN to lists
df.to_csv(CSV_NAME, index=False, header=['x', 'y'])