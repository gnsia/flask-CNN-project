
from flask import Flask, request, render_template, redirect, session
import base64
import cv2
import datetime
import numpy as np
from keras.models import load_model

import tensorflow as tf
from keras import backend as K

import io
labels_file = io.open('label.txt', 'r', encoding='utf-8').read().split()
label = [ str for str in labels_file]


init_Base64 = 21  # data:image/png;base64, 로 시작하는 
app = Flask(__name__)

# global model
# global sess
# global graph


# sess = tf.Session()
# graph = tf.get_default_graph()
# K.set_session(sess)


model = load_model('mnist_cnn.h5')
modelH = load_model('hand_written_korean_classification.hdf5')


@app.route('/')
def index():
    return render_template("mnist.html")


@app.route('/upload', methods=["POST"]) # post방식이 올때 업로드하려면
def upload():
    draw = request.form['url']
    draw = draw[init_Base64:]
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype='uint8')
    mode = request.form.get("mode", "digit")
    
    if mode == "digit" :
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(28,28), interpolation=cv2.INTER_AREA)
        image = image.reshape(1,28,28,1)
#         with graph.as_default():
#             K.set_session(sess)
        p = model.predict(image)
        p = np.argmax(p)
    
    if mode == "korean" :
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(32,32), interpolation=cv2.INTER_AREA)
        image = (255 - image) / 255
        image = image.reshape(1,32,32,3)
#         with graph.as_default():
#             K.set_session(sess)
        p = modelH.predict(image)
        p = np.argmax(p)
    
    return f"result : {p} <a href=javascript:history.back()>뒤로</a>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
