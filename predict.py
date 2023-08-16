
from PIL import Image
from time import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import keras.utils as image
from keras.utils import pad_sequences
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model


# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open ("models/wordtoindx.pkl", 'rb') as file:
    word_to_index = pickle.load(file)

index_to_word = {}
with open ("models/indxtoword.pkl", 'rb') as file:
    index_to_word = pickle.load(file)


#Load the trained model. (Pickle file)
print("Loading the model...")
model = load_model('models/keras_image_captioning.h5', compile=False)


#now load the inception v3 model with no last tow layers for feature extraction
print("Loading the inception...")
inception_customized = load_model('models/inception.h5', compile=False)


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode(image):
    image = preprocess(image)
    fea_vec = inception_customized.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

print("Encoding the image ...")
img_name = "static/input.jpg"
photo = encode(img_name)

# Generate Captions for a random image in test dataset
def beam_search_predictions(image, beam_index = 7):
    start = [word_to_index["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < 38: #38 max_length of caption
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=38, padding='post')
            print('percpas',par_caps.shape)
            print('image',image.reshape((1,2048)).shape)
            preds = model.predict([image.reshape((1,2048)),par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [index_to_word[i] for i in start_word]
    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption



print("Running model to generate the caption...")
caption = beam_search_predictions(photo)

# img_data = plt.imread(img_name)
# plt.imshow(img_data)
# plt.axis("off")

# plt.show()
# print(caption)