from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from io import BufferedReader

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'medical_trial_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)        # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        # f = request.files['file']
        # print(f)
        # # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # print(basepath)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        image_size=22
        try:
            image = cv2.imread(image,cv2.IMREAD_COLOR)
            image = cv2.resize(image(image_size,image_size))
        finally:
            print("done")
        val = np.array(image)
        vals = val.reshape(-1,image_size,image_size,3)    
        print(vals)        
        # Make prediction
        result = model.predict_classes(vals)
        print(result)
        return render_template('index.html', prediction_text='Your cataract prediction is {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)

