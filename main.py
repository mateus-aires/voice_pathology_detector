from flask import Flask, jsonify, request, render_template

import os
import constants as c
import extract_features
import util

app = Flask(__name__)
model, scaler = util.init()

@app.route('/')
def hello():
  return render_template('hello.html')

@app.route('/predict', methods=['POST'])
def predict():

    if c.AUDIO_1 not in request.files or c.AUDIO_2 not in request.files or c.AUDIO_3 not in request.files:
        return jsonify({'error_message': 'no audio files found'}), 400

    threshold = request.form.get(c.THRESHOLD) or c.DEFAULT_THR
    is_test = request.form.get(c.IS_TEST) == 'true'

    audio1, audio2, audio3 = util.convert(app, [request.files[c.AUDIO_1], request.files[c.AUDIO_2], request.files[c.AUDIO_3]])
    success, error_message, result, mean, pred1, pred2, pred3 = extract_features.predict_all(audio1, audio2, audio3,
                                                                                              model, scaler, threshold=float(threshold), 
                                                                                              is_test=is_test, delete=(not is_test))

    return jsonify({'is_successful': success,
                    'error_message': error_message,
                    'result': result, 
                    'probability': mean, c.PREDICTION_1: pred1, c.PREDICTION_2: pred2, c.PREDICTION_3: pred3}), 200

if __name__ == "__main__":
    app.run()
