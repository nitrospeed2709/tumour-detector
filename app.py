from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '/app/CNN Model final.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    image = load_img(img_path, target_size=(64, 64, 1), grayscale=True)
    image = np.array(image)/255.0
    image=image.reshape(64, 64, 1)
    images = np.array([image])

    # img = image.load_img(img_path, target_size=(64, 64, 1), grayscale = True)
    #
    # # Preprocessing the image
    # x = image.img_to_array(img)/255
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    #
    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict_classes(images)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)[0]

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        label=['There might be a glioma!', 'There might be a meningioma!', 'Most probably no tumor!', 'There might be a pituitary tumor!']
        pred_class = label[preds]                       # ImageNet Decode
        return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=True)
