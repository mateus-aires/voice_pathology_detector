from flask import Flask, jsonify, request, render_template
from joblib import load
from pydub import AudioSegment
import os
import constants as c

import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
import extract_features

def init(scaler_name='scaler.bin', model_name = 'model.json'):
    scaler=load(scaler_name)
    model = XGBClassifier()
    model.load_model(model_name)

    return model, scaler

def convert(audio_list):
    audio_names = []

    for idx, audio in enumerate(audio_list):
        audio_path = os.path.join(app.root_path, f"audio{idx+1}.wav")
        audio_file = AudioSegment.from_file(audio)
        audio_file.export(audio_path, format='wav')
        audio_names.append(audio_path)
    return audio_names[0], audio_names[1], audio_names[2]


app = Flask(__name__)

model, scaler = init()

@app.route('/')
def hello():
  return render_template('hello.html')

@app.route('/predict', methods=['POST'])
def predict():

    if c.AUDIO_1 not in request.files or c.AUDIO_2 not in request.files or c.AUDIO_3 not in request.files:
        return jsonify({'error': 'no audio files found'}), 400

    threshold = request.form.get(c.THRESHOLD) or c.DEFAULT_THR
    is_test = request.form.get(c.IS_TEST) == 'true'

    audio1, audio2, audio3 = convert([request.files[c.AUDIO_1], request.files[c.AUDIO_2], request.files[c.AUDIO_3]])
    result, pred1, pred2, pred3 = extract_features.predict_all(audio1, audio2, audio3, model, scaler, threshold=float(threshold), is_test=is_test, delete= not is_test)

    return jsonify({'result': result, c.PREDICTION_1: pred1, c.PREDICTION_2: pred2, c.PREDICTION_3: pred3}), 200

if __name__ == "__main__":
    app.run()
