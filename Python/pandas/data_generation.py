'''
Script to generate data based on a few parameters. This data will be used to test the pandas write to csv functions
Should return and int, a string and a list

'''

import pandas
import numpy

def data_creation(NUMBER = 0):
    the_string = 'a string and the number ' + str(NUMBER)
    the_int = NUMBER + 3
    # Ne pas oublier que les listes contiennent juste les références, donc on doit les déréférencer pour avoir les valeurs
    the_list = [*range(NUMBER, NUMBER + 6)]
    return the_string, the_int, the_list

string, integer, liste = data_creation(2)
print(string, integer, liste)