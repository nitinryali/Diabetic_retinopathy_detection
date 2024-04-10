import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from time import sleep
from Preprocess_Gaussian_Blur import load_ben_color
from Size_Normalize import del_black_or_white
from ContratsNormalize_CLAHE import CLAHE
import os
import tempfile
from ensemble import ensemble

input_shape = (224, 224, 3)
model_input = tf.keras.Input(shape=input_shape)

# Define Streamlit app layout
st.title("DR-DETECTION: Diabetic Retinopathy Classifier")

st.sidebar.title("Ensemble Architecture")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "tif"])

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(img, caption="Original Image", use_column_width=True)

    if st.button("Preprocess"):
        st.text("Preprocessing...")
        crop_size = 1000
        image1 = del_black_or_white(img)
        min_width_height = min(image1.shape[0], image1.shape[1])
        image_size_before_hough = crop_size * 2
        if min_width_height < 100:
            crop_ratio = image_size_before_hough / min_width_height
            image1 = cv2.resize(image1, None, fx=crop_ratio, fy=crop_ratio)

        dim = (224, 224)
        fundus1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)

        temp_file1 = tempfile.mktemp('.jpg')
        temp_file2 = tempfile.mktemp('.jpg')

        cv2.imwrite(temp_file1, fundus1)
        Clahe = CLAHE(temp_file1)
        cv2.imwrite(temp_file2, Clahe)
        img2 = load_ben_color(temp_file2, sigmaX=10)

        temp_output_file = tempfile.mktemp('.jpg')
        cv2.imwrite(temp_output_file, img2)

        st.image(temp_output_file, caption="Preprocessed Image", use_column_width=True)

    if st.button("Predict"):
        st.text("Predicting...")

        # Load the image and preprocess it
        img = tf.keras.preprocessing.image.load_img(temp_output_file, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Load and compile the models
        Base_model1 = tf.keras.applications.DenseNet201(input_shape=input_shape, input_tensor=model_input,
                                                         include_top=False, weights=None)
        for layer in Base_model1.layers:
            layer.trainable = True
        Base_model1_last_layer = Base_model1.get_layer('relu')
        Base_model1_last_output = Base_model1_last_layer.output
        x1 = tf.keras.layers.GlobalAveragePooling2D()(Base_model1_last_output)
        x1 = tf.keras.layers.Dropout(0.25)(x1)
        x1 = tf.keras.layers.Dense(512, activation='relu')(x1)
        x1 = tf.keras.layers.Dropout(0.25)(x1)
        final_output1 = tf.keras.layers.Dense(4, activation='softmax', name='final_output')(x1)
        DensNet201_model = tf.keras.models.Model(model_input, final_output1)
        optimizer = tf.keras.optimizers.Adam()
        DensNet201_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        weights_path = "WEIGHT/Weight_DensNet201_Optimal_Ori.h5"
        DensNet201_model.load_weights(weights_path)
        print(DensNet201_model.summary())
        print("DensNet201 Start predict.........")

        Predict1 = DensNet201_model.predict(x)
        print("Normal Probability:", Predict1[0][0])
        print("Mild Probability:", Predict1[0][1])
        print("Moderate Probability:", Predict1[0][2])
        print("Severe Probability:", Predict1[0][3])

        Base_model2 = tf.keras.applications.InceptionV3(input_shape=input_shape, input_tensor=model_input,
                                                        include_top=False, weights=None)
        for layer in Base_model2.layers:
            layer.trainable = True
        Base_model2_last_layer = Base_model2.get_layer('mixed10')
        Base_model2_last_output = Base_model2_last_layer.output
        x2 = tf.keras.layers.GlobalAveragePooling2D()(Base_model2_last_output)
        x2 = tf.keras.layers.Dropout(0.25)(x2)
        x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
        x2 = tf.keras.layers.Dropout(0.25)(x2)
        final_output2 = tf.keras.layers.Dense(4, activation='softmax', name='final_output2')(x2)
        InceptionV3_model = tf.keras.models.Model(model_input, final_output2)
        optimizer = tf.keras.optimizers.Adam(1.0000e-06)
        InceptionV3_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        weights_path1 = "WEIGHT/Weight_InceptionV3_Optimal_Ori.h5"
        InceptionV3_model.load_weights(weights_path1)
        print("InceptionV3 Start predict.........")

        Predict2 = InceptionV3_model.predict(x)
        print("Normal Probability:", Predict2[0][0])
        print("Mild Probability:", Predict2[0][1])
        print("Moderate Probability:", Predict2[0][2])
        print("Severe Probability:", Predict2[0][3])

        Base_model3 = tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=model_input,
                                                        include_top=False, weights=None)
        for layer in Base_model3.layers:
            layer.trainable = True
        Base_model3_last_layer = Base_model3.get_layer('out_relu')
        Base_model3_last_output = Base_model3_last_layer.output
        x3 = tf.keras.layers.GlobalAveragePooling2D()(Base_model3_last_output)
        x3 = tf.keras.layers.Dropout(0.5)(x3)
        x3 = tf.keras.layers.Dense(512, activation='relu')(x3)
        x3 = tf.keras.layers.Dropout(0.5)(x3)
        final_output3 = tf.keras.layers.Dense(4, activation='softmax', name='final_output3')(x3)
        MobileNetV2_model = tf.keras.models.Model(model_input, final_output3)
        optimizer = tf.keras.optimizers.Adam()
        MobileNetV2_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        weights_path2 = "WEIGHT/Weight_MobileNetV2_Optimal_(Ori).h5"
        MobileNetV2_model.load_weights(weights_path2)
        print("MobileNetV2 Start predict.........")

        Predict3 = MobileNetV2_model.predict(x)
        print("Normal Probability:", Predict3[0][0])
        print("Mild Probability:", Predict3[0][1])
        print("Moderate Probability:", Predict3[0][2])
        print("Severe Probability:", Predict3[0][3])

        ensemble_model = ensemble([DensNet201_model, InceptionV3_model, MobileNetV2_model], model_input)
        ensemble_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        print("Ensemble Start predict.........")

        Predict = ensemble_model.predict(x)
        print("Normal Probability:", Predict[0][0])
        print("Mild Probability:", Predict[0][1])
        print("Moderate Probability:", Predict[0][2])
        print("Severe Probability:", Predict[0][3])

        diagnosis = np.argmax(Predict)
        if diagnosis == 0:
            st.write("Diagnosis: NORMAL")
        elif diagnosis == 1:
            st.write("Diagnosis: MILD")
        elif diagnosis == 2:
            st.write("Diagnosis: MODERATE")
        elif diagnosis == 3:
            st.write("Diagnosis: SEVERE")
