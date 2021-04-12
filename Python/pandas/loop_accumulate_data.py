'''
File where the loop will generate data from the data_generation function, then store it in the same csv file

'''

import pandas as pd
from data_generation import data_creation

def main(NUM_LOOP = 5):
    df = pd.DataFrame()
    for i in range(NUM_LOOP):
        st, itg, lst = data_creation(i)
        data = {
            'string' : st,
            'integer': itg,
            'list'   : lst
        }
        print(data)
        df = df.append(data, ignore_index=True)
        print('df after append: ', df)
        # Important de spécifier mode='a' si on veut que ça 'append' les infos, autrement \
        # la méthode est en 'write', ce qui fait que le contenu du .csv sera exactement celui \
        # du DataFrame au moment de l'écriture
    df.to_csv('data.csv', mode='a')

    return df
    

df = main(5)

df = pd.read_csv('data.csv')
print(df)
