'''
using os module to count specific files

in this case, the file1 and file2
'''

import os


# path is '/' because same folder
path = "C:\\Users\\mara1928\\Desktop\\Github\\gittest\\Python\\sys"
number_of_files = len([name for name in os.listdir('.') if ('csv' in name and 'file' in name)])

print('number of \'file*\': ', number_of_files)