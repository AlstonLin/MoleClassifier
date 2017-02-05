from flask import Flask, jsonify, request, abort, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import tempfile
import sys
sys.path.insert(0, "./lib")
from ml import ML
from jagedness import calculateJagedness
from asymmetry import calculateAsymmetry
from coloring import calculateColoring
from crop import crop
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = './img/upload/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'bmp', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Init the ML
ml = ML.load("./dat/trained.dat")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def upload_file():
    print(request.files['file'])
    if 'file' not in request.files:
        print("File not found")
        abort(400)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        print("File has no name")
        abort(400)
    if file and allowed_file(file.filename):
        try:
            nparr = np.fromstring(file.read(), np.uint8)
            img = crop(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            j = calculateJagedness(gray)
            a = calculateAsymmetry(gray)
            c = calculateColoring(img)
            X = [[j, a, c]]
            prob = ml.predictProbability(X)
            print(ml.totalAsymmetry)
            response = {
                "likelihood": prob,
                "asymmetry": a / (ml.totalAsymmetry / ml.numTrained),
                "jagedness": j / (ml.totalJagedness / ml.numTrained),
                "coloring": c / (ml.totalColoring / ml.numTrained)
            }
            return jsonify(response)
        except Exception:
            abort(400)
    else:
        abort(400)

if __name__ == "__main__":
  app.run()
