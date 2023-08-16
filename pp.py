import tensorflow
import pickle
from tensorflow import keras
from tensorflow.keras.models import load_model

print("Loading the inception...")
loaded_keras_model = load_model('models/inception.h5', compile=False)




print(loaded_keras_model)