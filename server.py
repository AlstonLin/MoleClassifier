from flask import Flask, jsonify, request, abort, url_for
from werkzeug.utils import secure_filename
import cv2
import tempfile
app = Flask(__name__)

UPLOAD_FOLDER = './img/upload'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def upload_file():
    print(request.files['file'])
    # if 'file' not in request.files['file']:
    #     print("File not found")
    #     abort(400)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        print("File has no name")
        abort(400)
    if file and allowed_file(file.filename):
        tempFile = tempfile.TemporaryFile()
        filename = secure_filename(file.filename)
        tmpImag = open(tempFile.name, 'wb')
        tmpImag.write(file.read())
        cv2.cvtColor(tmpImag, cv2.COLOR_BGR2RGB)
        return '500'


if __name__ == "__main__":
  app.run()
