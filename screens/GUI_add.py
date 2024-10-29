import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.model.train_models_XGBoost import XGBoostStrokeModel
from screens.aux_functions import create_gauge_chart, load_css, load_image
from firebase_admin import firestore
from BBDD.database import FirebaseInitializer
import tensorflow as tf
import pickle
from threading import Thread 
from src.model.mlflow_xgboost import XGBoostStrokeModel, background_worker
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Initialize Firebase
firebase_init = FirebaseInitializer()
db = firebase_init.db

@st.cache_resource
def load_model():
    with open('src/Data/woe_dict.pkl', 'rb') as f:
        dict_woe = pickle.load(f)
    xgb_model = XGBoostStrokeModel(model_path = 'src/model/xgboost_model.joblib', scaler_path = 'src/model/xgb_scaler.joblib')
    return xgb_model, dict_woe

def load_nn_model():
    model = tf.keras.models.load_model('src/model/nn_stroke_model.keras')
    scaler = joblib.load('src/model/nn_scaler.joblib')
    return model, scaler

def transform_to_woe(df, woe_dict):
    df_woe = df.copy()
    for col, woe_df in woe_dict.items():
        woe_map = woe_df.set_index('Category')['WoE'].to_dict()
        df_woe[col] = df_woe[col].map(woe_map)
    return df_woe

def save_prediction_to_firebase(data):
    try:
        # Añadir timestamp
        data['timestamp'] = datetime.now()
        
        # Crear nueva referencia en la colección
        db.collection('new_data').add(data)
        print("Prediction saved to Firebase successfully")
        return True
    except Exception as e:
        print(f"Error saving to Firebase: {e}")
        return False

def convert_gender(gender_es):
    return "Male" if gender_es == "Masculino" else "Female"

def convert_work_type(work_type_es):
    mapping = {
        "Privado": "Private",
        "Autónomo": "Self-employed",
        "Gubernamental": "Govt_job",
        "Niño": "children",
        "Nunca ha trabajado": "Never_worked"
    }
    return mapping[work_type_es]

def convert_smoking_status(smoking_es):
    mapping = {
        "Nunca fumó": "never smoked",
        "Fumador": "smokes",
        "Exfumador": "formerly smoked"
    }
    return mapping[smoking_es]

def convert_yes_no(value_es):
    return "Yes" if value_es == "Sí" else "No"

def convert_residence_type(residence_es):
    return "Urban" if residence_es == "Urbana" else "Rural"

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_age_category(age):
    if age < 35:
        return "Young"
    elif age < 60:
        return "Middle"
    else:
        return "Elderly"

def get_glucose_category(glucose):
    if glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Pre-diabetes"
    else:
        return "Diabetes"
    

xgb_model, dict_woe = load_model()
nn_model, nn_scaler = load_nn_model()

bg_thread = Thread(target=background_worker, args=(xgb_model,))
bg_thread.start()

def screen_add():
    load_css('style.css')

    # Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('predictus.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Título y subtítulo 
    st.markdown('<h1 class="big-font">Predictor de Ictus</h1>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Ingrese los datos del paciente para predecir el riesgo de ictus</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Género", ["Masculino", "Femenino"])
        age = st.slider("Edad", 0, 100, 50)
        hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
        heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Sí"])
        ever_married = st.selectbox("Alguna vez casado", ["No", "Sí"])

    with col2:
        work_type = st.selectbox("Tipo de trabajo", ["Privado", "Autónomo", "Gubernamental", "Niño", "Nunca ha trabajado"])
        residence_type = st.selectbox("Tipo de residencia", ["Urbana", "Rural"])
        avg_glucose_level = st.slider("Nivel promedio de glucosa", 50.0, 300.0, 100.0)
        bmi = st.slider("IMC (Índice de Masa Corporal)", 10.0, 50.0, 25.0)
        smoking_status = st.selectbox("Estado de fumador", ["Nunca fumó", "Fumador", "Exfumador"])
        stroke = st.selectbox("Ictus", ["No", "Sí"])

    if st.button("Añadir nuevo caso a la base de datos."):
        patient_data = {
            "age": age,
            "gender": convert_gender(gender),
            "hypertension": 1 if hypertension == "Sí" else 0,
            "heart_disease": 1 if heart_disease == "Sí" else 0,
            "ever_married": convert_yes_no(ever_married),
            "work_type": convert_work_type(work_type),
            "Residence_type": convert_residence_type(residence_type),
            "smoking_status": convert_smoking_status(smoking_status),
            "bmi_category": get_bmi_category(bmi),
            "age_category": get_age_category(age),
            "glucose_level_category": get_glucose_category(avg_glucose_level),
            "stroke": 1 if stroke == "Sí" else 0  # Convertir a valor numérico
        }

        if save_prediction_to_firebase(patient_data):
            st.success("Los datos han sido añadidos exitosamente a la base de datos.")
        else:
            st.error("Hubo un error al guardar los datos. Por favor, intente nuevamente.")

    # Pie de página
    st.markdown('---')
    st.markdown('<p style="color: white;">© 2024 PREDICTUS - Tecnología Avanzada para la Prevención de Ictus. Todos los derechos reservados.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
        screen_add()