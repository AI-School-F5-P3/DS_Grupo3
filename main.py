import os
import sys
import streamlit.web.bootstrap as bootstrap
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_streamlit():
    try:
        # Configurar variables de entorno necesarias
        os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '8080')
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'true'
        os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
        os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '50'
        
        # Encontrar GUI.py
        gui_path = Path(__file__).parent / "GUI.py"
        if not gui_path.exists():
            raise FileNotFoundError(f"GUI.py not found at {gui_path}")
        
        logger.info(f"Starting Streamlit with file: {gui_path}")
        
        # Configuraciones adicionales para el servidor
        flag_options = {
            'server.port': int(os.environ.get('PORT', 8080)),
            'server.address': '0.0.0.0',
            'server.headless': True,
            'server.enableCORS': True,
            'server.enableXsrfProtection': False,
            'browser.serverAddress': '0.0.0.0',
            'browser.gatherUsageStats': False,
            'server.maxUploadSize': 50,
            'server.maxMessageSize': 50,
        }
        
        # Usar el método bootstrap en lugar de cli
        bootstrap.run(str(gui_path), '', [], flag_options=flag_options)
        
    except Exception as e:
        logger.exception(f"Failed to start Streamlit: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_streamlit()