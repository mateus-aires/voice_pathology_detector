import parselmouth
from parselmouth.praat import call
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

import os
import util
import constants as c

def extrair_features_mfcc(path_arquivo, segundos=1):

    audio, sr = librosa.load(path_arquivo)

    # Calcula o espectrograma de curto tempo com tamanho de janela n_fft
    n_fft = int(0.025 * sr)  # tamanho de janela de 25ms
    hop_length = int(0.01 * sr)  # avança a janela em 10ms
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Calcula os coeficientes MFCCs
    n_mfcc = 13  # usa 13 coeficientes MFCCs
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(np.abs(stft)), sr=sr, n_mfcc=n_mfcc)

    # Calcula a média e o desvio padrão de cada coeficiente MFCC
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Concatena as médias e os desvios padrão para obter os atributos finais
    mfccs_features = np.concatenate((mfccs_mean, mfccs_std))

    return mfccs_features


def extract_acoustic_features(voiceID, segundos=1):

  sound = parselmouth.Sound(voiceID) # read the sound
  pitch = call(sound, "To Pitch", 0.0, 75, 600) #create a praat pitch object
  meanF0 = call(pitch, "Get mean", 0, segundos, 'Hertz') # get mean pitch
  stdevF0 = call(pitch, "Get standard deviation", 0, segundos, 'Hertz') # get standard deviation
  harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
  hnr = call(harmonicity, "Get mean", 0, segundos)
  pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
  localJitter = call(pointProcess, "Get jitter (local)", 0, segundos, 0.0001, 0.02, 1.3)
  localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, segundos, 0.0001, 0.02, 1.3)
  rapJitter = call(pointProcess, "Get jitter (rap)", 0, segundos, 0.0001, 0.02, 1.3)
  ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, segundos, 0.0001, 0.02, 1.3)
  ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, segundos, 0.0001, 0.02, 1.3)
  localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, segundos, 0.0001, 0.02, 1.3, 1.6)
  localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, segundos, 0.0001, 0.02, 1.3, 1.6)
  apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, segundos, 0.0001, 0.02, 1.3, 1.6)
  aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, segundos, 0.0001, 0.02, 1.3, 1.6)
  apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, segundos, 0.0001, 0.02, 1.3, 1.6)
  ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, segundos, 0.0001, 0.02, 1.3, 1.6)
    
  return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer,  localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer


def extract_spectral_contrast(audio_file, n_bands):
    y, sr = librosa.load(audio_file)
    S = np.abs(librosa.stft(y))
    spec_cont = librosa.feature.spectral_contrast(S=S, sr=sr,n_bands=n_bands)
    a, b = spec_cont.shape
    return spec_cont.reshape((1, a*b))


def extract_rms(audio_file):
    y, sr = librosa.load(audio_file)
    return librosa.feature.rms(y=y)


def extract_zero_crossing(audio_file, audio_ext='wav'):
#     audio_file = get_first_or_middle_segment(audio_file, audio_ext)

    y, sr = librosa.load(audio_file)
    zero_crossing_arr = librosa.feature.zero_crossing_rate(y)
    return zero_crossing_arr

def extract_features_predict(audio_file, duration, taxa_nova_amostragem=44100, audio_ext='wav'):
    
    try:
        
        chunk_name = util.get_first_segment(audio_file, audio_ext, duration * 1000)
        features_mfcc = extrair_features_mfcc(audio_file, segundos=duration)
        acoustic_features = extract_acoustic_features(audio_file, segundos=duration)
        spectral_contrast = extract_spectral_contrast(chunk_name, 4)[0]
        zero_crossing = extract_zero_crossing(chunk_name)[0]
        rms = extract_rms(chunk_name)[0]

        features = np.concatenate((features_mfcc, acoustic_features, spectral_contrast, zero_crossing, rms))

        return features
    except IOError:
        raise Exception(c.INTERNAL_ERROR_MESSAGE)

def extract_and_scale(audio_file, duration, scaler, taxa_nova_amostragem):
    arr_attr = extract_features_predict(audio_file, duration, taxa_nova_amostragem=taxa_nova_amostragem)
    return scaler.transform(arr_attr.reshape(1, -1))

def test_predict(audio_file, model, scaler, proba=False):
    scaled = extract_and_scale(audio_file, 1, scaler, 22500)
    
    if proba:
        return model.predict_proba(scaled)
    
    return model.predict(scaled)


def extract_mean_poba(file_name, model, scaler, test=False):
    
    if not util.is_one_second_or_more(file_name):
        raise Exception(c.AUDIO_TOO_SHORT_ERROR_MESSAGE)
    
    iterate_file = [file_name]
    
    if not test:
        iterate_file = util.preprocess_and_create_chunks(file_name, file_name)
    
    sum_total = 0
    
    for name in iterate_file:
        sum_total += test_predict(name, model, scaler, proba=True)[0][1]
    
    mean = (sum_total / len(iterate_file))
    return iterate_file, mean


def predict_all(audio1, audio2, audio3, model, scaler, threshold = 0.5, is_test=False, delete=False):
    
    audio_list = [audio1, audio2, audio3]
    chunks_names = []
    
    sum_predictions = 0
    sum_predictions_prob = 0

    predictions = []

    try:
        for audio in audio_list:
            files, x = extract_mean_poba(audio, model, scaler, test=is_test)
            chunks_names += files
            print(x)
            sum_predictions_prob += x
            predictions.append(x)
            if x > threshold:
                sum_predictions += 1

        mean = sum_predictions_prob / 3

        if delete:
            delete_files(chunks_names + audio_list)
        
        result, mean, pred1, pred2, pred3 = process_preds(sum_predictions, threshold, predictions, mean)

        return True, "", result, mean, pred1, pred2, pred3
    
    except Exception as e:
        return False, str(e), "false", 0, 0, 0, 0
    


def process_preds(sum_predictions, thr, predictions, mean):
    result = bool(mean > thr)
    if sum_predictions == 3:
        return True, mean, predictions[0], predictions[1], predictions[2]
    elif sum_predictions == 2:
        if result:
            return True, mean, predictions[0], predictions[1], predictions[2]
        else:
            sum = 0
            for i in range(predictions):
                if predictions[i] > thr:
                   sum += predictions[i]
            new_mean = sum / 2 
            return True, new_mean, predictions[0], predictions[1], predictions[2]
    else:
        return result, mean, predictions[0], predictions[1], predictions[2]

def delete_files(files_list):
    files_list.append("chunk0.wav")
    for file in files_list:
        if os.path.exists(file):
            os.remove(file)
