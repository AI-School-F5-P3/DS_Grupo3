import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.model.train_models_XGBoost import XGBoostStrokeModel
from screens.aux_functions import create_gauge_chart, load_css, load_image
import tensorflow as tf
import pickle
from firebase_admin import firestore
import datetime
from BBDD.database import FirebaseInitializer
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Firebase
firebase_init = FirebaseInitializer()
db = firebase_init.db

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

@st.cache_resource
def load_model():
    with open('src/Data/woe_dict.pkl', 'rb') as f:
        dict_woe = pickle.load(f)
    return XGBoostStrokeModel.load_model('src/model/xgboost_model.joblib', 'src/model/xgb_scaler.joblib'), dict_woe

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
        # Create a new document in the 'patient_predictions' collection
        predictions_ref = db.collection('patient_predictions').document()
        
        # Add timestamp
        data['timestamp'] = datetime.datetime.now()
        
        # Save to Firestore
        predictions_ref.set(data)
        print("Prediction saved to Firebase successfully")
        return True
    except Exception as e:
        print(f"Error saving to Firebase: {e}")
        return False

def save_metrics_to_firebase(metrics_data):
    try:
        metrics_ref = db.collection('model_metrics').document()
        metrics_data['timestamp'] = datetime.datetime.now()
        metrics_ref.set(metrics_data)
        print("Metrics saved to Firebase successfully")
        return True
    except Exception as e:
        print(f"Error saving metrics to Firebase: {e}")
        return False

xgb_model, dict_woe = load_model()
nn_model, nn_scaler = load_nn_model()

def screen_predict():
    load_css('style.css')

    # Logo and title setup (unchanged)
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('predictus.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

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

    if st.button("Predecir Riesgo de Ictus"):
        # Prepare model inputs (unchanged)
        inputs = pd.DataFrame({
            'gender': [1 if gender == "Masculino" else 0],
            'hypertension': [1 if hypertension == "Sí" else 0],
            'heart_disease': [1 if heart_disease == "Sí" else 0],
            'ever_married': [1 if ever_married == "Sí" else 0],
            'work_type': [0 if work_type == "Privado" else 1 if work_type == "Autónomo" else 2 if work_type == "Gubernamental" else 3 if work_type == "Niño" else 4],
            'Residence_type': [1 if residence_type == "Urbana" else 0],
            'smoking_status': [0 if smoking_status == "Nunca fumó" else 1 if smoking_status == "Exfumador" else 2],
            'bmi_category': ['Underweight' if bmi < 18.5 else 'Normal Weight' if bmi < 25 else 'Overweight' if bmi < 30 else 'Mega Obesity'],
            'age_category': ['Niño' if age < 13 else 'Joven' if age < 18 else 'Adulto' if age < 60 else 'Tercera Edad'],
            'glucose_level_category': ['Low' if avg_glucose_level < 100 else 'Medium' if avg_glucose_level < 140 else 'High']
        })

        # Transform and predict (unchanged)
        category_columns = ['work_type', 'smoking_status', 'bmi_category', 'age_category', 'glucose_level_category']
        inputs_woe = inputs.copy()
        inputs_woe[category_columns] = transform_to_woe(inputs_woe[category_columns], dict_woe)

        xgb_probabilities = xgb_model.predict_proba(inputs_woe)[0][1]
        inputs_nn = nn_scaler.transform(inputs_woe)
        nn_probabilities = nn_model.predict(inputs_nn)
        #nn_probability = nn_probabilities[0]
        final_probabilities = xgb_probabilities
        final_prediction = 1 if final_probabilities >= 0.5 else 0

        # Prepare data according to Firebase schema
        firebase_data = {
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
            "prediction": final_prediction,
            "prediction_probability": float(final_probabilities)
        }

        # Save prediction
        save_prediction_to_firebase(firebase_data)

        # Save metrics
        metrics_data = {
            "avg_prediction": float(final_probabilities),
            "entropy": float(-np.sum(final_probabilities * np.log(final_probabilities)) if final_probabilities > 0 else 0),
            "ks_statistic": float(np.max(final_probabilities))
        }
        save_metrics_to_firebase(metrics_data)

        # Display results (rest of the visualization code remains unchanged)
        st.subheader("Resultados de la Predicción")
        
        fig = create_gauge_chart(float(final_probabilities) * 100, "Probabilidad de Ictus")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recomendaciones")
        if final_prediction == 1:
            st.markdown('<div class="recommendation-high">Se recomienda consultar a un médico para una evaluación más detallada.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-low">Mantener un estilo de vida saludable para prevenir riesgos futuros.</div>', unsafe_allow_html=True)

        st.subheader("Factores de riesgo identificados")
        risk_factors = []
        if firebase_data['hypertension'] == 1:
            risk_factors.append("Hipertensión")
        if firebase_data['heart_disease'] == 1:
            risk_factors.append("Enfermedad cardíaca")
        if firebase_data['smoking_status'] == "smokes":
            risk_factors.append("Fumador activo")
        if firebase_data['bmi_category'] == "Obese":
            risk_factors.append("Obesidad")
        if firebase_data['glucose_level_category'] == "Diabetes":
            risk_factors.append("Nivel alto de glucosa")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f'<div class="recommendation-high">{factor}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-low">No se identificaron factores de riesgo principales.</div>', unsafe_allow_html=True)

        st.subheader("Más información sobre el ictus y su prevención")
        if st.button("Sobre el Ictus"):
            js = "window.open('https://www.fundacioictus.com/es/sobre-el-ictus/prevencion/factores-de-riesgo/')"
            html = f"<script>{js}</script>"
            st.markdown(html, unsafe_allow_html=True)

        st.markdown('---')
        st.markdown('<p style="color: white;">© 2024 PREDICTUS - Tecnología Avanzada para la Prevención de Ictus. Todos los derechos reservados.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    screen_predict()