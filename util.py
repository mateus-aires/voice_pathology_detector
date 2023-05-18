import wave
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
from pydub import AudioSegment
import os
import constants as c
from joblib import load
from xgboost import XGBClassifier
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

from scipy.io.wavfile import read

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
    myaudio = AudioSegment.from_file(file_path, audio_format) 
    chunk_length_ms = 500
    
    chunks = make_chunks(myaudio, chunk_length_ms)
    chunks_qtd = len(chunks)
    last_index = chunks_qtd - 1
    
    exported_names = []
    
    if(len(chunks) > 1):
        
        for i in range(1, min(chunks_qtd, 8), 2):
            j = i + 1
            
            if i == last_index or j == last_index:
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
    raise Exception(c.AUDIO_TOO_SHORT_ERROR_MESSAGE)
        
          
def preprocess_and_create_chunks(filename, filepath):
    remove_silence(filename, filepath)
    return remove_first_half_second(filename, filepath)

def convert(app, audio_list):
    audio_names = []

    for idx, audio in enumerate(audio_list):
        audio_path = os.path.join(app.root_path, f"audio{idx+1}.wav")
        audio_file = AudioSegment.from_file(audio)
        audio_file.export(audio_path, format='wav')
        audio_names.append(audio_path)
    return audio_names[0], audio_names[1], audio_names[2]


def init(scaler_name='scaler.bin', model_name = 'model.json'):
    scaler=load(scaler_name)
    model = XGBClassifier()
    model.load_model(model_name)

    return model, scaler

def now():
    now = datetime.utcnow()
    now_here = now - timedelta(hours=c.DEFAULT_TIMEDELTA)
    print(f"{now_here.day}/{now_here.month}/{now_here.year}", now_here.strftime("%H:%M:%S"))

