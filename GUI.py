import streamlit as st
from threading import Thread
import time
from src.model.modelo_optimizado import OptimizedStrokeModel

# Configuración de la página
st.set_page_config(
    page_title="PREDICTUS - Predicción de Ictus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mostrar mensaje de carga mientras se inicializa
with st.spinner('Cargando aplicación... Por favor espere.'):
    # Inicialización del modelo
    @st.cache_resource
    def load_model():
        return OptimizedStrokeModel()

    if 'model' not in st.session_state:
        st.session_state.model = load_model()

    # Importar las pantallas después de cargar el modelo
    from screens.GUI_home import home_screen
    from screens.GUI_predict import screen_predict
    from screens.GUI_report import screen_informe
    from screens.GUI_info import screen_info
    from screens.GUI_add import screen_add

# Inicialización de estado
if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

def change_screen(new_screen):
    st.session_state.screen = new_screen

# Sidebar para navegación
with st.sidebar:
    st.header("Menú de Navegación")
    if st.button("Home"):
        change_screen("home")
    if st.button("Predicción de Stroke"):
        change_screen("predict")
    if st.button("Métricas de Rendimiento"):
        change_screen("informe")
    if st.button("Información del Modelo"):
        change_screen("info")
    if st.button("Añadir nuevo caso"):
        change_screen("nuevo")

# Renderizar la pantalla correspondiente
try:
    if st.session_state.screen == 'home':
        home_screen()
    elif st.session_state.screen == 'predict':
        screen_predict()
    elif st.session_state.screen == 'informe':
        screen_informe()
    elif st.session_state.screen == 'info':
        screen_info()
    elif st.session_state.screen == 'nuevo':
        screen_add()
except Exception as e:
    st.error(f"Error al cargar la pantalla: {str(e)}")
    st.error("Por favor, recarga la página si el problema persiste.")