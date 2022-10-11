'''
Class to handle data about the wav file treated
'''
import io
import numpy as np

class CWaveFile(object):
    def __init__(self, file_path) -> None:
        self.audio_data
        self.byte_rate
        
    
    def get_audio_data(self, wave_file):
        file = io.FileIO(wave_file, mode='rb')
        raw_buffer = io.RawIOBase.read(file)
        self.byte_rate = np.frombuffer(raw_buffer[28:32], dtype=np.int16)[0]
        self.audio_data = np.frombuffer(raw_buffer[46:], dtype=np.int16)