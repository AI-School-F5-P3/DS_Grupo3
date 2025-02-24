import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.model.train_models_XGBoost import XGBoostStrokeModel
from screens.aux_functions import create_gauge_chart, load_css, load_image
from BBDD.database_utils import get_database_connection
from BBDD.models import ModelMetrics, PatientData
import tensorflow as tf
import pickle
from threading import Thread 
from src.model.mlflow_xgboost import XGBoostStrokeModel, background_worker

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

        session = get_database_connection()
        try:
            bd_inputs = PatientData(
                age=age,
                gender=1 if gender == "Masculino" else 0,
                hypertension=1 if hypertension == "Sí" else 0,
                heart_disease=1 if heart_disease == "Sí" else 0,
                ever_married=1 if ever_married == "Sí" else 0,
                work_type=0 if work_type == "Privado" else 1 if work_type == "Autónomo" else 2 if work_type == "Gubernamental" else 3 if work_type == "Niño" else 4,
                Residence_type=1 if residence_type == "Urbana" else 0,
                smoking_status=0 if smoking_status == "Nunca fumó" else 1 if smoking_status == "Exfumador" else 2,
                bmi_category=0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3,
                age_category=0 if age < 13 else 1 if age < 18 else 2 if age < 60 else 3,
                glucose_level_category=0 if avg_glucose_level < 100 else 1 if avg_glucose_level < 140 else 2,
                stroke = 1 if stroke == "Sí" else 0
            )

            session.add(bd_inputs)
            session.commit()
            print("Datos enviados a la base de datos")
        except Exception as e:
            # Si ocurre un error, revertir la transacción
            session.rollback()
            print(f"Error al insertar nuevos datos: {e}")
        finally:
        # Cerrar la sesión
            session.close()

        st.success("Los datos han sido añadidos exitosamente a la base de datos.")

        # Pie de página
    st.markdown('---')
    st.markdown('<p style="color: white;">© 2024 PREDICTUS - Tecnología Avanzada para la Prevención de Ictus. Todos los derechos reservados.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
        screen_add()