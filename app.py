from flask import Flask, render_template, request
from predict import preprocess,encode,beam_search_predictions

import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}  # Define allowed file extensions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return 'No file part'
        file = request.files['imagefile']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            file.save(filename)
            photo = encode(filename)
            caption = beam_search_predictions(photo)
            print(caption)
            return render_template('index.html', prediction_text='Caption for the uploaded image: {}'.format(caption), image_url='uploads/input.jpg')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)