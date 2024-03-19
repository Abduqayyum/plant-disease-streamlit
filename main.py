import streamlit as st
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import os
from streamlit.components.v1 import components
from tensorflow.keras.utils import custom_object_scope
import keras.utils as keras_utils
import tensorflow_hub as hub



class_names = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_healthy',
       'Cherry_Powdery_mildew', 'Cherry_healthy', 'Corn_Common_rust',
       'Corn_Gray_leaf_spot', 'Corn_healthy', 'Grape_Black_rot',
       'Grape_Esca_(Black_Measles)',
       'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy',
       'Peach_Bacterial_spot', 'Peach_healthy',
       'Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy',
       'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
       'Strawberry_Leaf_scorch', 'Strawberry_healthy',
       'Tomato_Bacterial_spot', 'Tomato_Early_blight',
       'Tomato_Late_blight', 'Tomato_Leaf_Mold',
       'Tomato_Septoria_leaf_spot', 'Tomato_Tomato_mosaic_virus',
       'Tomato_healthy']



def load_and_preprocess_image(image, filesize=224):
    img = tf.image.resize(image, [filesize, filesize])
    img = img / 255
    return tf.convert_to_tensor(img)  # Convert NumPy array to TensorFlow tensor

def predict_image(image, model, class_names):
    img = load_and_preprocess_image(image)
    pred = model.predict(tf.expand_dims(img, axis=0))
    class_name = class_names[tf.argmax(pred[0])]
    max_indices = np.argmax(pred, axis=1)
    # Get the actual probability values for each sample
    max_probabilities = pred[np.arange(len(pred)), max_indices]
    return class_name, max_probabilities

st.title("Plants Disease Detection")

image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

with keras_utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = keras.models.load_model("efficiennet_model_aug.h5")

# model = keras.models.load_model("efficiennet_model_aug.h5")

if image is not None:
    image = Image.open(image)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    pred, prob = predict_image(image_array, model, class_names)
    st.image(image.resize((300, 300), resample=Image.Resampling.BILINEAR), caption="Uploaded Image", use_column_width=True)
    st.header(f"Prediction: {pred}")
    st.subheader(f"Probability: {int(prob[0]*100) % 100} %")
