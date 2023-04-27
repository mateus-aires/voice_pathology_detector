import wave


from pydub.utils import make_chunks
from pydub.silence import split_on_silence
from pydub import AudioSegment
import os

import warnings
warnings.filterwarnings("ignore")

from scipy.io.wavfile import read

print("Hello world!")

def is_one_second_or_more(audio_file):
    
    with wave.open(audio_file, "r") as mywav:
        duration_seconds = mywav.getnframes() / mywav.getframerate()
        return duration_seconds >= 1
    

def get_first_segment(audio_file, audio_ext, chunk_length=1000):
    
    myaudio = AudioSegment.from_file(audio_file , audio_ext) 
    chunks = make_chunks(myaudio, chunk_length)
    
    if len(chunks) == 1:
        if is_one_second_or_more(audio_file):
            return audio_file
        else:
            raise IOError
    else:
        chunks[0].export("chunk0.wav", format=audio_ext)
        return "chunk0.wav"
    
  
def remove_ext(file_name):
    return file_name.replace(".wav", "")

def remove_silence(filename, filepath, audio_format="wav"):

    aud = AudioSegment.from_file(filepath, format = audio_format)
    
    audio_chunks = split_on_silence(
        aud
        ,min_silence_len = 100
        ,silence_thresh = -45
        ,keep_silence = 50
    )
    
    combined = AudioSegment.empty()

    for chunk in audio_chunks:
        combined += chunk

    combined.export(filename, format="wav", bitrate='800k')
    
    
def remove_first_half_second(file_name, file_path, audio_format='wav'):
    file_name = remove_ext(file_name)
    myaudio = AudioSegment.from_file(file_path , audio_format) 
    chunk_length_ms = 500
    
    chunks = make_chunks(myaudio, chunk_length_ms)
    
    chunks_qtd = len(chunks)
    
    exported_names = []
    
    if(len(chunks) > 1):
        
        for i in range(1, chunks_qtd):
            j = i + 1
            
            if j == chunks_qtd - 1:
                break
            
            
            one_second_combined = AudioSegment.empty()
            
            chunk1 = chunks[i]
            chunk2 = chunks[j]
            
            one_second_combined = chunk1 + chunk2
            
            chunk_name = "{filename}-{index1}{index2}:{size}.wav".format(filename=file_name, index1=j, index2=j+1, size=chunks_qtd)
            print ("exporting", chunk_name)
            one_second_combined.export(chunk_name, format="wav")
            exported_names.append(chunk_name)
        return exported_names
    raise Exception("Audio too short")
        
          
def preprocess_and_create_chunks(filename, filepath, audio_format="wav"):
    remove_silence(filename, filepath)
    return remove_first_half_second(filename, filepath)

def delete_files(files_list):
    for file in files_list:
        os.remove("demofile.txt")
