'''
Class to handle data about the wav file treated
wave proprerties until now
-raw data
-byterate
'''
import io
import numpy as np

class CWaveFile(object):
    def __init__(self, file_path) -> None:
        self.audio_data = None
        self.byte_rate = None
        self.__wave_file = file_path
        
    def get_audio_data(self):
        file = io.FileIO(self.__wave_file, mode='rb')
        raw_buffer = io.RawIOBase.read(file)
        self.audio_data = np.frombuffer(raw_buffer[46:], dtype=np.int16)
        self.byte_rate = np.frombuffer(raw_buffer[28:32], dtype=np.int16)[0]
        
    def set_new_wave_file(self, new_wave_file_path):
        self.__wave = new_wave_file_path
        self.audio_data = None
        self.byte_rate = None
        
    # Si on veut que chaque objet de la classe puisse aller chercher les données dans le 
    def __getitem__(self, index):
        return self.audio_data[index]
        