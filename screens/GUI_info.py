import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ks_2samp
from screens.aux_functions import load_css, load_image
from BBDD.database_utils import get_database_connection

def screen_info():
    load_css('style.css')

    # Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('predictus.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Título y subtítulo 
    st.markdown('<h1 class="big-font">Predictor de Ictus</h1>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Información del modelo</p>', unsafe_allow_html=True)

    # Título para los gráficos
    st.markdown('<h2 class="medium-font">Gráficos del Modelo</h2>', unsafe_allow_html=True)

    # Selectbox para seleccionar el gráfico a mostrar
    option = st.selectbox(
        'Selecciona el gráfico que deseas ver:',
        ('Curva ROC', 'Importancia de Características')
    )

    # Mostrar el gráfico seleccionado
    if option == 'Curva ROC':
        st.image(load_image('roc_curve.png'), caption='Curva ROC')
    elif option == 'Importancia de Características':
        st.image(load_image('feature_importance.png'), caption='Importancia de Características')