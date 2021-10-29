import pandas as pd
import numpy as np
import csv

CSV_NAME = 'test.csv'

with open(CSV_NAME, 'a', newline='') as file:
    writer = csv.writer(file)
    for i in range(10):
        actual_speed = [[i, i*2, i*3, i*4],[i*1.5, i*2.5, i*3.5]]
        preds = [i, i/2, i/3, i/4]
        liste = [i, actual_speed, preds]
        writer.writerow(liste)
    print(writer)
    


