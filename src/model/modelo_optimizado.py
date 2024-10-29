import joblib
from google.cloud import storage
import os

class OptimizedStrokeModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_models()
        
    def _load_models(self):
        if os.getenv('USE_CLOUD_STORAGE') == 'true':
            self._load_from_cloud()
        else:
            self._load_local()
    
    def _load_from_cloud(self):
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(os.getenv('GCP_BUCKET_NAME'))
            
            # Descargar modelo
            model_blob = bucket.blob('models/xgboost_model.joblib')
            model_blob.download_to_filename('/tmp/model.joblib')
            self.model = joblib.load('/tmp/model.joblib')
            
            # Descargar scaler
            scaler_blob = bucket.blob('models/xgb_scaler.joblib')
            scaler_blob.download_to_filename('/tmp/scaler.joblib')
            self.scaler = joblib.load('/tmp/scaler.joblib')
            
            # Limpiar archivos temporales
            os.remove('/tmp/model.joblib')
            os.remove('/tmp/scaler.joblib')
        except Exception as e:
            print(f"Error loading from Cloud Storage: {e}")
            self._load_local()
    
    def _load_local(self):
        self.model = joblib.load('src/model/xgboost_model.joblib')
        self.scaler = joblib.load('src/model/xgb_scaler.joblib')
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)