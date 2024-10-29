# GUI_img.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from screens.aux_functions import load_css, load_image

# Ruta del modelo entrenado
MODEL_PATH = 'src/model/nn_stroke_img.keras'

@st.cache_resource
def load_nn_model():
    """Función para cargar el modelo de Keras"""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

nn_model = load_nn_model()

def preprocess_image(image):
    """Función para preprocesar la imagen cargada por el usuario"""
    image = image.resize((224, 224))  # Cambiar el tamaño de la imagen a 224x224
    image = image.convert('RGB')      # Asegurarse de que la imagen tiene 3 canales
    image_array = np.array(image) / 255.0  # Normalizar la imagen
    image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión batch
    return image_array

def screen_image_prediction():
    load_css('style.css')
    # Cargar el archivo CSS para el estilo (opcional)
    st.markdown('<style>body {background-color: #f0f2f6;}</style>', unsafe_allow_html=True)

    # Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('predictus.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Título y subtítulo
    st.title("Predicción de Imágenes de Ictus")
    st.write("Sube una imagen para predecir si muestra signos de ictus.")

    # Carga de la imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", width=300)

        # Preprocesar la imagen
        image_array = preprocess_image(image)

        # Realizar la predicción
        prediction = nn_model.predict(image_array)
        prediction_probability = prediction[0][0]

        # Mostrar resultados
        st.subheader("Resultado de la Predicción")
        if prediction_probability >= 0.5:
            st.markdown('<div style="color: red;">Se han detectado signos de ictus en la TC.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: green;">No se han detectado signos de ictus en la TC.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    screen_image_prediction()