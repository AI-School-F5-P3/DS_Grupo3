import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.sklearn
import time
from threading import Thread
from dotenv import load_dotenv
from datetime import datetime, timedelta
from BBDD.database import FirebaseInitializer

load_dotenv()

mlflow.set_tracking_uri('http://mlflow:5000') # Asignar la URL del servidor de MLflow
mlflow.set_experiment('stroke_prediction_xgboost') # Asignar el experimento de MLflow

class XGBoostStrokeModel:
    def __init__(self, csv_path=None, model_path=None, scaler_path=None):
        if csv_path:
            # Si le damos un archivo CSV, inicializamos el modelo completo
            self.model = None
            self.scaler = None
            self.feature_names = None
            self.last_check_file = 'last_check.txt'
            self.initial_training_file = 'initial_training_done.txt'
            self.check_interval = int(os.getenv('CHECK_INTERVAL', 3600))
            self.csv_path = csv_path
            
            # Inicializar Firebase
            firebase_init = FirebaseInitializer()
            self.db = firebase_init.db
            self.logger = firebase_init.logger
            self.schema = firebase_init.FIREBASE_SCHEMA['new_data']
            
            self.db_config = {
                'collection': 'new_data', # Nombre de la colección en Firestore
                'batch_size': int(os.getenv('FIRESTORE_BATCH_SIZE', '1000')),
                'cache_duration': int(os.getenv('FIRESTORE_CACHE_DURATION', '300'))
            }
            self._cached_data = None
            self._last_cache_time = None
        elif model_path and scaler_path:
            # Si le damos rutas a un modelo y un escalador, los cargamos
            self.model = joblib.load('src/model/xgboost_model.joblib')
            self.scaler = joblib.load('src/model/xgb_scaler.joblib')

    def load_data_from_csv(self):
        """Carga datos desde un archivo CSV y los almacena en un DataFrame."""
        try:
            df = pd.read_csv(self.csv_path)
            self.feature_names = df.drop('stroke', axis=1).columns
            self.logger.info(f"Datos cargados desde CSV. Número de filas: {len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"Error al cargar datos desde CSV: {e}")
            return pd.DataFrame()

    def _should_refresh_cache(self):
        """Determina si la caché de datos debe ser actualizada."""
        if self._last_cache_time is None:
            return True
        cache_age = datetime.now() - self._last_cache_time
        return cache_age.total_seconds() > self.db_config['cache_duration']

    def load_data_from_firestore(self):
        """Carga datos desde Firestore y los almacena en un DataFrame."""
        try:
            # Check if we can use cached data
            if not self._should_refresh_cache() and self._cached_data is not None:
                self.logger.info("Usando datos en caché")
                return self._cached_data

            self.logger.info("Cargando datos desde Firestore...")
            collection_ref = self.db.collection(self.db_config['collection'])
            
            # Get total document count first (excluding _metadata document)
            query = collection_ref.where('__name__', '!=', '_metadata')
            total_docs = len(list(query.limit(1).get()))
            
            if total_docs == 0:
                return pd.DataFrame()

            # Fetch documents in batches
            all_docs = []
            batch_size = self.db_config['batch_size']
            last_doc = None

            while True:
                if last_doc:
                    query = collection_ref.where('__name__', '!=', '_metadata')\
                                       .limit(batch_size)\
                                       .start_after(last_doc)
                else:
                    query = collection_ref.where('__name__', '!=', '_metadata')\
                                       .limit(batch_size)

                docs = list(query.get())
                if not docs:
                    break

                # Validar cada documento contra el esquema
                for doc in docs:
                    doc_dict = doc.to_dict()
                    if self._validate_document(doc_dict):
                        all_docs.append(doc_dict)
                
                last_doc = docs[-1]
                self.logger.info(f"Cargados {len(all_docs)} documentos de {total_docs}")

            df = pd.DataFrame(all_docs)
            
            if not df.empty:
                self.feature_names = df.drop('stroke', axis=1).columns
                
                # Cache the results
                self._cached_data = df
                self._last_cache_time = datetime.now()
                
                self.logger.info(f"Datos cargados desde Firestore. Número de filas: {len(df)}")
            return df

        except Exception as e:
            self.logger.error(f"Error al cargar datos desde Firestore: {e}")
            return pd.DataFrame()

    def _validate_document(self, doc_dict):
        """Valida un documento contra el esquema definido."""
        required_fields = set(self.schema['fields'].keys())
        doc_fields = set(doc_dict.keys())
        
        # Verificar campos requeridos
        if not required_fields.issubset(doc_fields):
            missing_fields = required_fields - doc_fields
            self.logger.warning(f"Campos faltantes en el documento: {missing_fields}")
            return False
        
        # Validar tipos y valores permitidos
        for field, field_type in self.schema['fields'].items():
            value = doc_dict.get(field)
            if field in self.schema.get('validations', {}):
                if value not in self.schema['validations'][field]:
                    self.logger.warning(f"Valor no válido para {field}: {value}")
                    return False
                    
        return True

    def preprocess_data(self, df):
        """Preprocesa los datos, escalando las características y separando las etiquetas."""
        if df.empty:
            self.logger.warning("El DataFrame está vacío")
            return None, None
        else:
            if 'stroke' in df.columns:
                X = df.drop('stroke', axis=1)
                y = df['stroke']
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    X_scaled = self.scaler.transform(X)
                return X_scaled, y
            else:
                self.logger.error("La columna 'stroke' no está presente en el DataFrame")
            return None, None

    def train_model(self, X, y):
        """Entrena el modelo XGBoost con los datos proporcionados."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Forma de X_train:", X_train.shape)
        print("Forma de y_train:", y_train.shape)
        print("Tipos de datos en X_train:", X_train.dtypes if isinstance(X_train, pd.DataFrame) else X_train.dtype)
        print("Valores únicos en y_train:", np.unique(y_train))
        print("Hay NaN en X_train:", np.isnan(X_train).any())
        print("Hay infinitos en X_train:", np.isinf(X_train).any())
        
        with mlflow.start_run():
            if self.model is None:
                self.model = xgb.XGBClassifier(
                    # Usamos los parámetros que se encontraron durante el primer entrenamiento para evitar usar optuna y comer más recursos y tiempo
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic'
                )
            
            try:
                self.model.fit(X_train, y_train)
            except Exception as e:
                print("Error al ajustar el modelo XGBoost:")
                print(str(e))
                raise
            
            mlflow.log_params(self.model.get_params())
            
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            
            mlflow.sklearn.log_model(self.model, "model")
            
            self.log_plots(X_test, y_test)
            
        return X_test, y_test

    def initial_training(self):
        """Realiza el entrenamiento inicial del modelo usando datos de un archivo CSV."""
        if not os.path.exists(self.initial_training_file):
            print("Realizando entrenamiento inicial desde CSV...")
            df = self.load_data_from_csv()
            X, y = self.preprocess_data(df)
            self.train_model(X, y)
            with open(self.initial_training_file, 'w') as f:
                f.write('Initial training completed')
            print("Entrenamiento inicial completado.")
        else:
            print("El entrenamiento inicial ya se ha realizado anteriormente.")
            self.load_model('src/model/xgboost_model.joblib', 'src/model/xgb_scaler.joblib')

    def check_and_retrain(self):
        """Verifica si se cumplen las condiciones para reentrenar el modelo y lo reentrena si es necesario."""
        if self.should_check():
            print("Verificando condiciones para reentrenamiento...")
            df = self.load_data_from_firestore()
            
            if len(df) > int(os.getenv('RETRAIN_THRESHOLD', 10000)): # Se requieren 10000 registros
                print("Condiciones cumplidas. Iniciando reentrenamiento...")
                X, y = self.preprocess_data(df)
                self.train_model(X, y)
            else:
                print("No se cumplen las condiciones para reentrenar.")
            
            self.update_last_check()

    def should_check(self):
        """Determina si se debe realizar una verificación para reentrenar el modelo."""
        if not os.path.exists(self.last_check_file):
            return True
        
        with open(self.last_check_file, 'r') as f:
            last_check = float(f.read().strip())
        
        return time.time() - last_check > self.check_interval

    def update_last_check(self):
        """Actualiza el archivo de última verificación con la marca de tiempo actual."""
        with open(self.last_check_file, 'w') as f:
            f.write(str(time.time()))

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def load_model(self, model_path, scaler_path):
        """Carga el modelo y el escalador desde archivos."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

def background_worker(model):
    """Función que se ejecuta en segundo plano para verificar y reentrenar el modelo periódicamente."""
    while True:
        model.check_and_retrain()
        time.sleep(model.check_interval)

# Inicialización del modelo y el worker en segundo plano
model = XGBoostStrokeModel(csv_path='src/Data/train_stroke_woe_smote.csv')
model.initial_training()
bg_thread = Thread(target=background_worker, args=(model,))
bg_thread.start()