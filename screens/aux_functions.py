import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
from PIL import Image

def load_data():
    # Función para cargar los datos
    return pd.read_csv('src/Data/stroke_woe_smote.csv')

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00a2bb"},
            'bgcolor': "#000e26",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#3BE980'},
                {'range': [50, 75], 'color': '#E8ED47'},
                {'range': [75, 100], 'color': '#E34F24'}],
            'threshold': {
                'line': {'color': "#000e26", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(paper_bgcolor = "#000e26", font = {'color': "white", 'family': "Arial"})
    
    return fig
def get_project_root():
    """Obtiene la ruta raíz del proyecto."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def load_css(file_name):
    project_root = get_project_root()
    css_path = os.path.join(project_root, 'src', 'styles', file_name)
    
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_image(image_name):
    """Carga una imagen desde la carpeta src/images."""
    project_root = get_project_root()
    image_path = os.path.join(project_root, 'src', 'images', image_name)
    return Image.open(image_path)

class DataCleaner:
    def __init__(self, woe_dict_path='woe_dict.pkl'):
        with open(woe_dict_path, 'rb') as f:
            self.woe_dict = pickle.load(f)

    def categorize_bmi(self, bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 24.9:
            return 'Normal weight'
        elif 25 <= bmi < 29.9:
            return 'Overweight'
        elif 30 <= bmi < 39.9:
            return 'Obesity'
        else:
            return 'Mega Obesity'

    def categorize_age(self, age):
        if age < 18:
            return 'Niño'
        elif 18 <= age < 35:
            return 'Joven'
        elif 35 <= age < 60:
            return 'Adulto'
        else:
            return 'Tercera Edad'

    def categorize_glucose_level(self, glucose_level):
        if glucose_level < 70:
            return 'Low'
        elif 70 <= glucose_level < 140:
            return 'Medium'
        else:
            return 'High'

    def transform_to_woe(self, df):
        df_woe = df.copy()
        for col, woe_df in self.woe_dict.items():
            woe_map = woe_df.set_index('Category')['WoE'].to_dict()
            df_woe[col] = df_woe[col].map(woe_map)
        return df_woe

    def cleanup_data(self, data):
        data['bmi_category'] = data['bmi'].apply(self.categorize_bmi)
        data['age_category'] = data['age'].apply(self.categorize_age)
        data['glucose_level_category'] = data['avg_glucose_level'].apply(self.categorize_glucose_level)
        data = self.transform_to_woe(data)
        return data