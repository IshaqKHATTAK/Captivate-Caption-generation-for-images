from flask import Flask, render_template
from predict import preprocess,encode,beam_search_predictions


# print("Encoding the image ...")
# img_name = "static/input.jpg"
# photo = encode(img_name)
# app = Flask(__name__)


# print("Running model to generate the caption...")
# caption = beam_search_predictions(photo)


@app.route("/")
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))

@app.route("/about owner")
def about():
    return "<p>about page</p>"

@app.route("/resource")
def resource():
    return "<p>resource page</p>"


if __name__ == '__main__':
    app.run(debug=True)






# from flask import Flask, render_template, request

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# #from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50

# app = Flask(__name__)
# model = ResNet50()

# @app.route('/', methods=['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile= request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)

#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     yhat = model.predict(image)
#     label = decode_predictions(yhat)
#     label = label[0][0]

#     classification = '%s (%.2f%%)' % (label[1], label[2]*100)


#     return render_template('index.html', prediction=classification)


# if __name__ == '__main__':
#     app.run(port=3000, debug=True)