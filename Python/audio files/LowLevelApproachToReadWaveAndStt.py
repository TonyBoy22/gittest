'''
To be called main file eventually
Trying BytesIO and Wave module to read the buffer of a wav file
based on https://aneescraftsmanship.com/wav-file-format/

Also try to benchmark the ffmpeg mp3 to wav conversion for a specific file length and resolution

'''
import numpy as np
import io
import os
import subprocess
import concurrent.futures # concurrent > multiprocessing since 3.2
import time
import deepspeech
import GPUtil

# https://github.com/anderskm/gputil#usage
gpu_lists = GPUtil.getGPUs()

if gpu_lists:
    # case where there is a detected compatible gpu
    print(gpu_lists)
    
print(gpu_lists)

model_file_path = 'deepspeech-0.9.3-models.pbmm'
lm_file_path = 'deepspeech-0.9.3-models.scorer'
beam_width = 500
lm_alpha = 0.93
lm_beta = 1.18

model = deepspeech.Model(model_file_path)
model.enableExternalScorer(lm_file_path)

model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)

CHUNK_LENGTH    = 4 # in seconds
OVERLAP         = 1 # in seconds

def get_wav_bytes(wave_file):
    file = io.FileIO(wave_file, mode='rb')
    raw_buffer = io.RawIOBase.read(file)
    byteRate = np.frombuffer(raw_buffer[28:32], dtype=np.int16)[0]
    audio_data = np.frombuffer(raw_buffer[46:], dtype=np.int16)
    return byteRate, audio_data

def get_start_and_end_indexes_per_segment(byteStream, process_index, byteRate):
    chunk_length_in_bytes = (CHUNK_LENGTH - OVERLAP)*byteRate
    start_index = process_index * chunk_length_in_bytes
    end_index = min(start_index + chunk_length_in_bytes, len(byteStream))
    return start_index, end_index

def translate_a_segment(start_index, end_index, byteStream):
    translated_string = model.stt(byteStream[start_index:end_index])
    return translated_string

def translate_from_tuple(data_in_tuple):
    audio_data, indexes, byterate = data_in_tuple
    start_index, end_index = indexes
    translated_string = translate_a_segment(start_index, end_index, audio_data)
    return translated_string

def main_cpu(wav_file):
    assert(os.path.exists(wav_file))
    # assert() sur le type de fichier
    
    # Instanciate waveFile
    byterate, audio_data = get_wav_bytes(wave_file=wav_file)
    
    # Index list
    bytes_per_segment = byterate*(CHUNK_LENGTH - OVERLAP)/2   # divisé par deux car deux bytes par int16
    number_of_segment = int(len(audio_data)/bytes_per_segment) + 1

    index_list = np.zeros((number_of_segment, 2), dtype=np.int16)
    for i in range(number_of_segment):
        start_index, end_index = get_start_and_end_indexes_per_segment(audio_data, i, byterate)
        index_list[i][0] = start_index
        index_list[i][1] = end_index
    
    result_list = ['']*number_of_segment
    
    # Algo général
    # for i in range(number_of_segment):
    #     # 1er objectif
    #     # obtenir les index en fonction de i
    #     start_index, end_index = get_start_and_end_indexes_per_segment(audio_data, i, byterate)
        
    #     # donner les index et le byte stream à la traduction d'un segment
    #     translated_string = translate_a_segment(start_index, end_index, audio_data)
        
    #     # Annexer les résultats à la liste des résultat 
    #     result_list[i] = translated_string
        
    # print(result_list)

    # En mode multiprocessing
    start = time.perf_counter()
    ############ With newer multiprocessing class ######################
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pas besoin de déclarer une result list initialement, 
        results = [executor.submit(translate_from_tuple, (audio_data, indexes, byterate)) for indexes in index_list]

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')
    print(results)

def main_gpu(wav_file):
    assert(os.path.exists(wav_file))
    # Instanciate waveFile
    byterate, audio_data = get_wav_bytes(wave_file=wav_file)
    
    # Index list
    bytes_per_segment = byterate*(CHUNK_LENGTH - OVERLAP)/2   # divisé par deux car deux bytes par int16
    number_of_segment = int(len(audio_data)/bytes_per_segment) + 1

    index_list = np.zeros((number_of_segment, 2), dtype=np.int16)
    result_list = ['']*number_of_segment
    start = time.perf_counter()
    for i in range(number_of_segment):
        start_index, end_index = get_start_and_end_indexes_per_segment(audio_data, i, byterate)
        index_list[i][0] = start_index
        index_list[i][1] = end_index
            #     # donner les index et le byte stream à la traduction d'un segment
        translated_string = translate_a_segment(start_index, end_index, audio_data)
        
    #     # Annexer les résultats à la liste des résultat 
        result_list[i] = translated_string
    
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} seconds')
    print(result_list)
    

if __name__ == "__main__":
    audio_file = "./198/wav_file.wav"
    # main_cpu(audio_file)
    main_gpu(audio_file)


    
    

