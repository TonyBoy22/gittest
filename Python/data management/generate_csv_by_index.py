'''
creates a bunch of data in a for loop and save it in a different csv file each time
'''

import numpy as np
import pandas as pd
import os


class Generator:
    def __init__(self) -> None:
        self._csv_string = 'donnee_csv'
        
    def generate_data(self, index):
        '''
        will be index of for loop in 
        '''
        data = pd.DataFrame(data=[np.array([index]*10)])
        return data

    def create_csv(self, index, data):
        '''
        takes the csv string and add a num 2 str to file
        '''
        # Look for file with the same index first
        if os.path.exists(self._csv_string + str(index)):
            file = self._csv_string + str(index+1)
            open(file, 'a').close()
            data.to_csv(file, mode='a')
            
            
        else:
            file = self._csv_string + str(index)
            open(self._csv_string + str(index), 'a').close()
            data.to_csv(file, mode='a')
        
        return

def main():
    gen = Generator()
    index_list = [0,2,2,3,4]
    for i in index_list:
        data = gen.generate_data(i)
        
        gen.create_csv(i, data)
        # Append data
        # csv_file.write(data)
            

if __name__ == '__main__':
    main()