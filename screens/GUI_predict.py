import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.model.train_models_XGBoost import XGBoostStrokeModel
from screens.aux_functions import create_gauge_chart, load_css, load_image
from BBDD.database_utils import get_database_connection
from BBDD.models import ModelMetrics, PatientPredictions
import tensorflow as tf
import pickle

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

xgb_model, dict_woe = load_model()
nn_model, nn_scaler = load_nn_model()

def screen_predict():
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

    if st.button("Predecir Riesgo de Ictus"):
        # Preparar los inputs para el modelo
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

        category_columns = ['work_type', 'smoking_status', 'bmi_category', 'age_category', 'glucose_level_category']

        # Transformar variables categóricas a WOE
        inputs_woe = inputs.copy()
        inputs_woe[category_columns] = transform_to_woe(inputs_woe[category_columns], dict_woe)

        print("Datos enviados al modelo:")
        print(inputs)
        print(inputs_woe)

        # Realizar predicción
        xgb_probabilities = xgb_model.predict_proba(inputs_woe)[0][1]
        inputs_nn = nn_scaler.transform(inputs_woe)
        nn_probabilities = nn_model.predict(inputs_nn)
        nn_probability = nn_probabilities[0]
        final_probabilities = 0.6 * xgb_probabilities + 0.4 * nn_probability
        final_prediction = 1 if final_probabilities >= 0.5 else 0


         # Guardar la predicción en la base de datos
         # Convertir final_probabilities a un valor compatible con MySQL (float)
        if isinstance(final_probabilities, (list, np.ndarray)):
            final_probabilities = float(final_probabilities[0])  # Si es un array, toma el primer valor y lo convierte a float
        else:
            final_probabilities = float(final_probabilities)  

        session = get_database_connection()
        try:
            bd_inputs = PatientPredictions(
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
                prediction=final_prediction,
                prediction_probability=final_probabilities
            )

            session.add(bd_inputs)
            session.commit()
            print("Datos enviados a la base de datos")

            # Métricas del modelo
            avg_prediction = final_probabilities  
            entropy = -np.sum(final_probabilities * np.log(final_probabilities)) 
            ks_statistic = np.max(final_probabilities) 
            
            new_metric = ModelMetrics(
                avg_prediction=avg_prediction,
                entropy=entropy,
                ks_statistic=ks_statistic
            )
            session.add(new_metric)
            session.commit()
            print("Métricas del modelo guardadas en la base de datos")
        except Exception as e:
            # Si ocurre un error, revertir la transacción
            session.rollback()
            print(f"Error al insertar predicción o métricas: {e}")
        finally:
        # Cerrar la sesión
            session.close()

        # Mostrar resultados
        st.subheader("Resultados de la Predicción")
      
        final_probabilities = float(final_probabilities)
        fig = create_gauge_chart(final_probabilities * 100, "Probabilidad de Ictus")
        st.plotly_chart(fig, use_container_width=True)


        # Añadir recomendaciones basadas en el riesgo
        st.subheader("Recomendaciones")

        if final_prediction == 1:
            st.markdown('<div class="recommendation-high">Se recomienda consultar a un médico para una evaluación más detallada.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-low">Mantener un estilo de vida saludable para prevenir riesgos futuros.</div>', unsafe_allow_html=True)

        # Factores de riesgo identificados
        st.subheader("Factores de riesgo identificados")
        risk_factors = []
        if inputs['hypertension'].values[0] == 1:
            risk_factors.append("Hipertensión")
        if inputs['heart_disease'].values[0] == 1:
            risk_factors.append("Enfermedad cardíaca")
        if inputs['smoking_status'].values[0] == 1:
            risk_factors.append("Fumador activo")
        if inputs['bmi_category'].values[0] == 'Obesidad':
            risk_factors.append("Obesidad")
        if inputs['glucose_level_category'].values[0] == 'Alto':
            risk_factors.append("Nivel alto de glucosa")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f'<div class="recommendation-high">{factor}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-low">No se identificaron factores de riesgo principales.</div>', unsafe_allow_html=True)

        # Botón que lleva a una página web externa sobre prevención del ictus
        st.subheader("Más información sobre el ictus y su prevención")
        if st.button("Sobre el Ictus"):
            js = "window.open('https://www.fundacioictus.com/es/sobre-el-ictus/prevencion/factores-de-riesgo/')"  # Cambia el enlace a la URL deseada
            html = f"<script>{js}</script>"
            st.markdown(html, unsafe_allow_html=True)

        # Pie de página
        st.markdown('---')
        st.markdown('<p style="color: white;">© 2024 PREDICTUS - Tecnología Avanzada para la Prevención de Ictus. Todos los derechos reservados.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
        screen_predict()