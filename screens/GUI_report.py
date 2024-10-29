import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ks_2samp
from screens.aux_functions import load_css, load_image
from firebase_admin import firestore
from BBDD.database import FirebaseInitializer

def calculate_metrics(prediction, probabilities, reference_data):
    avg_prediction = np.mean(probabilities)
    entropy = -np.mean(probabilities * np.log2(probabilities + 1e-10) + 
                       (1 - probabilities) * np.log2(1 - probabilities + 1e-10))
    ks_statistic, _ = ks_2samp(probabilities, reference_data['prediction_probability'])
    return avg_prediction, entropy, ks_statistic

def update_performance_charts(metrics_df):
    fig_avg = go.Figure(data=go.Scatter(x=metrics_df['timestamp'], y=metrics_df['avg_prediction']))
    fig_avg.update_layout(title='Predicción promedio a lo largo del tiempo', xaxis_title='Timestamp', yaxis_title='Predicción promedio')
    
    fig_entropy = go.Figure(data=go.Scatter(x=metrics_df['timestamp'], y=metrics_df['entropy']))
    fig_entropy.update_layout(title='Entropía de predicción a lo largo del tiempo', xaxis_title='Timestamp', yaxis_title='Entropía')
    
    fig_ks = go.Figure(data=go.Scatter(x=metrics_df['timestamp'], y=metrics_df['ks_statistic']))
    fig_ks.update_layout(title='Estadístico KS a lo largo del tiempo', xaxis_title='Timestamp', yaxis_title='Estadístico KS')
    
    return fig_avg, fig_entropy, fig_ks

def plot_prediction_distribution(predictions):
    fig = go.Figure(data=[go.Histogram(x=predictions)])
    fig.update_layout(title='Distribución de Predicciones', xaxis_title='Probabilidad', yaxis_title='Frecuencia')
    return fig

def get_firebase_data(collection_name, limit=None, order_by='timestamp', descending=True):
    """Función auxiliar para obtener datos de Firebase y convertirlos a DataFrame"""
    firebase = FirebaseInitializer()
    collection_ref = firebase.db.collection(collection_name)
    
    # Crear la consulta base
    query = collection_ref.order_by(order_by, direction=firestore.Query.DESCENDING if descending else firestore.Query.ASCENDING)
    
    if limit:
        query = query.limit(limit)
    
    # Ejecutar la consulta y convertir a DataFrame
    docs = query.stream()
    data = [doc.to_dict() for doc in docs]
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)

def screen_informe():
    load_css('style.css')

    # Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('predictus.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Título y subtítulo 
    st.markdown('<h1 class="big-font">Predictor de Ictus</h1>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Métricas de rendimiento del modelo</p>', unsafe_allow_html=True)

    try:

        metrics_history = get_firebase_data('model_metrics', limit = 1000)

        if not metrics_history.empty:
            latest_metrics = metrics_history.iloc[0]
            st.metric("Predicción promedio", f"{latest_metrics['avg_prediction']:.4f}")
            st.metric("Entropía", f"{latest_metrics['entropy']:.4f}")
            st.metric("Estadístico KS", f"{latest_metrics['ks_statistic']:.4f}")
        else:
            st.warning("No hay datos de métricas disponibles.")

        st.markdown('<p class="medium-font">Gráficos de rendimiento del modelo</p>', unsafe_allow_html=True)
        if not metrics_history.empty:
            fig_avg, fig_entropy, fig_ks = update_performance_charts(metrics_history)
            st.plotly_chart(fig_avg)
            st.plotly_chart(fig_entropy)
            st.plotly_chart(fig_ks)
        else:
            st.warning("No hay suficientes datos para generar gráficos.")

        if len(metrics_history) >= 50:
            recent_metrics = metrics_history.tail(50)
            avg_prediction_change = recent_metrics['avg_prediction'].pct_change().mean()
            entropy_change = recent_metrics['entropy'].pct_change().mean()
            ks_change = recent_metrics['ks_statistic'].pct_change().mean()

            st.subheader("Detección de Cambios")
            if abs(avg_prediction_change) > 0.1 or abs(entropy_change) > 0.1 or abs(ks_change) > 0.1:
                st.warning("Se han detectado cambios significativos en las métricas del modelo. Se recomienda una revisión.")
            else:
                st.success("No se han detectado cambios significativos en las métricas del modelo.")
        else:
            st.info("Se necesitan al menos 50 entradas de métricas para la detección de cambios.")

        st.markdown('<p class="medium-font">Distribución de Predicciones Recientes</p>', unsafe_allow_html=True)

        recent_predictions = get_firebase_data('patient_predictions', limit=1000)
        
        if not recent_predictions.empty:
            fig_dist = plot_prediction_distribution(recent_predictions['prediction_probability'])
            st.plotly_chart(fig_dist)
        else:
            st.warning("No hay datos de predicciones recientes disponibles.")
            
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")

if __name__ == "__main__":
    screen_informe()