from flask import Flask, render_template, request
import os
import shutil
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import time
# import cv2

# prediction_model = tf.keras.models.load_model("./models/saved_model")
prediction_model = pickle.load(open("./models/pickle_model.pkl", "rb"))
prediction_model.load_weights("./models/h5_model_weights.h5")
prediction_model.compile()
prediction_model.summary()

max_length = 30

# def bg_white(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     dilated = cv2.dilate(edges, None, iterations=1)

#     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     for contour in contours:
#         cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

#     result = cv2.bitwise_and(image, image, mask=mask)
#     result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

#     mod_new_path = image_path[:-4]+"_white_bg.png"
#     cv2.imwrite(mod_new_path, cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))

#     return mod_new_path


def image_resizing(image):
    w, h = 128, 64
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if (pad_height % 2 != 0):
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if (pad_width % 2 != 0):
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(image, paddings=[[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0], ], )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    return image

def image_preprocessing(image_path, img_size=(128, 64)):
    # mod_new_path = bg_white(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = image_resizing(image)
    image = tf.cast(image, tf.float32) / 255.0

    return image

def decode_batch_predictions(pred):
    global max_length
    characters = {'*', ',', '#', 'i', '^', 'y', '"', '=', 'P', '!', '@', 'A', 'F', 'Z', '/', 'M', 'R', 'W', 'U', 'v', '?', 'z', 'K', 'c', '\\', '}', 'o', '4', '&', '0', 'J', 'n', 'f', 'Y', 'l', '3', 'C', 'a', 'd', '(', 'N', '+', 'x', ';', '[', 'E', 'm', 'Q', ')', 'D', ']', '2', 'p', "'", '|', '5', '7', '>', '1', 'S', '$', 'b', 'q', '_', '%', 'X', 'g', '~', 'O', 'w', 'u', ':', '8', '{', '-', '6', '.', '<', 'j', 'L', 'B', 'T', '9', 'h', 'r', 't', 'G', 'I', 'e', 'V', 'H', 'k', 's'}
    
    char_to_num = tf.keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)
    num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]

    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    
    return output_text

app = Flask(__name__, template_folder="./")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    time.sleep(5)

    global img_id, prediction_model
    if request.method == 'POST':
        old_path = "./../HTR_Image_Download.png"
        new_path = "./testimage/HTR_Image_Download.png"

        shutil.copy(old_path, new_path)
        if os.path.isfile(old_path):
            os.remove(old_path)

        image = image_preprocessing(new_path)
        image = np.expand_dims(image, axis=0)
        pred = decode_batch_predictions(prediction_model.predict(image))
    
    return render_template('predict.html', prediction = pred[0])


app.run(debug=True)