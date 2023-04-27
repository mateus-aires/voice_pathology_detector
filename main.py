from flask import Flask, jsonify, request
from joblib import load
from pydub import AudioSegment
import os

import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
import extract_features

def init(scaler_name='scaler.bin', model_name = 'model.json'):
    scaler=load(scaler_name)
    model = XGBClassifier()
    model.load_model(model_name)

    return model, scaler

def convert(audio_list, format_received="3gp"):
    print(audio_list)
    print(type(audio_list[0]))
    audio_names = []
    for idx, audio in enumerate(audio_list):
        audio_path = os.path.join(app.root_path, f"audio{idx+1}.wav")
        audio_file = AudioSegment.from_file(audio, format=format_received)
        audio_file.export(audio_path, format='wav')
        audio_names.append(audio_path)
    return audio_names[0], audio_names[1], audio_names[2]

app = Flask(__name__)

model, scaler = init()

@app.route('/predict', methods=['POST'])
def predict():
    # checa se o arquivo de audio foi enviado
    if 'audio1' not in request.files or 'audio2' not in request.files or 'audio3' not in request.files:
        return jsonify({'error': 'no audio files found'}), 400
    print(request)
    audio1, audio2, audio3 = convert([request.files['audio1'], request.files['audio2'], request.files['audio3']], format_received="wav")
    print(audio1)

    prediction = extract_features.predict_all(audio1, audio2, audio3, model, scaler, delete=True)

    return jsonify({'result': prediction}), 200

if __name__ == "__main__":
    app.run()
