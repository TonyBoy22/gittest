'''
Class to manage files to translate and search.
Might redo it with regex

'''
import os

class FileToConvert(object):
    def __init__(self, file_name):
        self.file_name = str(file_name)# complete path to file
        self.file_type = None
        self.file_leaf = None
        self.containing_folder = None
        
    def get_file_type(self):
        dot_position = self.file_name.rfind('.')
        self.file_type = self.file_name[dot_position+1:]
        
    def get_file_containing_folder(self):
        slash_position = self.file_name.rfind('\\')
        if slash_position == -1:
            slash_position = self.file_name.rfind('/')
            if slash_position == -1:
                self.containing_folder = '.'
            else
                self.containing_folder = self.file_name[:slash_position]
        else
            self.containing_folder = self.file_name[:slash_position]
            
    def get_file_leaf(self):
        self.file_leaf = '.' # A changer en utilisant les regex pour trimmer avant les slash et après le point
