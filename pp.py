import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model

loaded_keras_model = load_model('models/keras_image_captioning.h5')


print(loaded_keras_model)