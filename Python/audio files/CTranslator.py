'''
Class to handle the translating process
Reponsibility: taking a string input and returning the sound to text conversion
also converts the audio file format if it is not a wave file

'''
import subprocess


class Translator(object):
    def __init__(self):
        file_to_convert = None
        file_name = None
        pass
    
    def get_file_format(file_to_convert):
        # first find the slash or backslash
        assert(type(file_to_convert) == str)
        dot_position = file_to_convert.rfind('.')
        file_name = file_to_convert[:dot_position]
        file_format = file_to_convert[dot_position+1:]
        return file_to_convert, file_format
    
    def translate_file_format(audio_file_to_convert_to_wave):
        
        command = f"ffmpeg -i {} .wav"
        return None
        
    def translate_speech_to_text():
        
        