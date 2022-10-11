'''
First demo wll be as is

'''

from deepspeech import Model
import numpy as np
import os
import subprocess
import wave
from pydub import AudioSegment
from pathlib import Path


model_file_path = 'deepspeech-0.9.3-models.pbmm'
lm_file_path = 'deepspeech-0.9.3-models.scorer'
beam_width = 500
lm_alpha = 0.93
lm_beta = 1.18

model = Model(model_file_path)
model.enableExternalScorer(lm_file_path)

model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)

# https://kkroening.github.io/ffmpeg-python/
# def convert_to_wav(audio_file):
#     input = ffmpeg.input('')

def convert_flac_to_wav(flac_file, wav_file):
    command = f"ffmpeg -i {flac_file} {wav_file}"
    subprocess.call(command, shell=True)
    
def convert_wav_to_mp3(wav_file, mp3_file):
    command = f"ffmpeg -i {wav_file} {mp3_file}"
    subprocess.call(command, shell=True)

def read_wav_file(filename):
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
        print(frames)
    return buffer, rate

# def read_mp3_file(mp3_file):
#     buffer = AudioSegment.from_file(mp3_file, format="mp3")
    
    
#     return buffer

def transcribe(audio_file):
    buffer_wav, rate_wav = read_wav_file(audio_file)
    data16 = np.frombuffer(buffer_wav, dtype=np.int16)
    return model.stt(data16)

def main():
    # audio_file = './198/19-198-0000.flac'
    audio_file = "./198/wav_file.wav"
    # print(os.path.exists(audio_file))

    print(transcribe(audio_file))
    
    # wav_file.audio.node.short_repr
    
if __name__ == "__main__":
    main()
    # convert_flac_to_wav('./198/19-198-0001.flac', './198/wav_file.wav')
    # convert_wav_to_mp3('./198/wav_file.wav', './198/mp3_file.mp3')
    print(os.path.exists("./198/mp3_file.mp3"))
    # mp3_path = Path("./198/mp3_file.mp3")
    # print(read_mp3_file(mp3_path))
