import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
import logging
from datetime import datetime
from dotenv import load_dotenv

class FirebaseInitializer:
    def __init__(self):
        self.logger = self._setup_logger()
        self._initialize_firebase()
        self.db = firestore.client()

    def _setup_logger(self):
        logger = logging.getLogger('FirebaseInit')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_firebase(self):
        try:
            if not firebase_admin._apps:
                # Obtener la ruta del directorio actual
                current_dir = Path(__file__).parent.absolute()
                
                # Buscar el archivo de credenciales en el directorio actual
                cred_path = current_dir / 'serviceAccountKey.json'
                
                if not cred_path.exists():
                    raise FileNotFoundError(
                        f"serviceAccountKey.json no encontrado en {current_dir}"
                    )
                
                self.logger.info(f"Usando credenciales desde: {cred_path}")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.logger.info("Firebase initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Firebase: {e}")
            raise

    # El resto de tu código de inicialización aquí...
    
    # Definición del esquema
    FIREBASE_SCHEMA = {
        "patient_predictions": {
            "fields": {
                "age": "integer",
                "gender": "string",
                "hypertension": "integer",
                "heart_disease": "integer",
                "ever_married": "string",
                "work_type": "string",
                "Residence_type": "string",
                "smoking_status": "string",
                "bmi_category": "string",
                "age_category": "string",
                "glucose_level_category": "string",
                "prediction": "integer",
                "prediction_probability": "float",
                "timestamp": "timestamp"
            },
            "validations": {
                "gender": ["Male", "Female"],
                "ever_married": ["Yes", "No"],
                "work_type": ["Private", "Self-employed", "Govt_job", "Never_worked", "children"],
                "Residence_type": ["Urban", "Rural"],
                "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
                "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
                "age_category": ["Young", "Middle", "Elderly"],
                "glucose_level_category": ["Normal", "Pre-diabetes", "Diabetes"]
            }
        },
        "model_metrics": {
            "fields": {
                "avg_prediction": "float",
                "entropy": "float",
                "ks_statistic": "float",
                "timestamp": "timestamp"
            }
        },
        "new_data": {
            "fields": {
                "age": "integer",
                "gender": "string",
                "hypertension": "integer",
                "heart_disease": "integer",
                "ever_married": "string",
                "work_type": "string",
                "Residence_type": "string",
                "smoking_status": "string",
                "bmi_category": "string",
                "age_category": "string",
                "glucose_level_category": "string",
                "prediction": "integer",
                "prediction_probability": "float",
                "stroke": "integer"
            },
            "validations": {
                "gender": ["Male", "Female"],
                "ever_married": ["Yes", "No"],
                "work_type": ["Private", "Self-employed", "Govt_job", "Never_worked", "children"],
                "Residence_type": ["Urban", "Rural"],
                "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
                "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
                "age_category": ["Young", "Middle", "Elderly"],
                "glucose_level_category": ["Normal", "Pre-diabetes", "Diabetes"]
            }
        }
    }

    def validate_data(self, collection_name: "string", data: dict) -> bool:
        """Valida los datos contra el esquema definido"""
        if collection_name not in self.FIREBASE_SCHEMA:
            return False
            
        schema = self.FIREBASE_SCHEMA[collection_name]
        fields = schema.get('fields', {})
        validations = schema.get('validations', {})
        
        for field, field_type in fields.items():
            if field not in data:
                continue  # Permitir campos opcionales
                
            value = data[field]
            if field in validations:
                if value not in validations[field]:
                    self.logger.error(f"Validation error: {field} value {value} not in {validations[field]}")
                    return False
            elif field_type != "timestamp" and not isinstance(value, field_type):
                self.logger.error(f"Type error: {field} should be {field_type}, got {type(value)}")
                return False
        
        return True

    def initialize_collections(self):
        """Inicializa las colecciones con sus índices y validaciones"""
        try:
            for collection_name, config in self.FIREBASE_SCHEMA.items():
                self.logger.info(f"Initializing collection: {collection_name}")
                
                # Crear colección si no existe
                collection_ref = self.db.collection(collection_name)
                
                # Crear documento de metadata para la colección
                metadata_ref = collection_ref.document('_metadata')
                metadata_ref.set({
                    'schema_version': '1.0',
                    'fields': config['fields'],
                    'validations': config.get('validations', {}),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                
            self.logger.info("Database initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing collections: {e}")
            raise

# Script de inicialización
if __name__ == "__main__":
    initializer = FirebaseInitializer()
    initializer.initialize_collections()