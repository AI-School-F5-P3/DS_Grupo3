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
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import pymysql

load_dotenv()

mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_experiment('stroke_prediction_xgboost')

class XGBoostStrokeModel:
    def __init__(self, csv_path=None, model_path=None, scaler_path=None):
        if csv_path:
            self.model = None
            self.scaler = None
            self.feature_names = None
            self.last_check_file = 'last_check.txt'
            self.initial_training_file = 'initial_training_done.txt'
            self.check_interval = int(os.getenv('CHECK_INTERVAL', 3600))
            self.csv_path = csv_path
            self.db_config = {
                'host': os.getenv('DB_HOST', 'mysql'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'database': os.getenv('DB_NAME'),
                'table': os.getenv('DB_TABLE', 'new_data')
            }
            self.engine = self.create_db_engine()
        elif model_path and scaler_path:
            self.model = joblib.load('src/model/xgboost_model.joblib')
            self.scaler = joblib.load('src/model/xgb_scaler.joblib')

    def create_db_engine(self):
        db_uri = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
        return create_engine(db_uri)

    def load_data_from_csv(self):
        try:
            df = pd.read_csv(self.csv_path)
            self.feature_names = df.drop('stroke', axis=1).columns
            print(f"Datos cargados desde CSV. Número de filas: {len(df)}")
            return df
        except Exception as e:
            print(f"Error al cargar datos desde CSV: {e}")
            return pd.DataFrame()

    def load_data_from_mysql(self):
        try:
            metadata = MetaData()
            table = Table(self.db_config['table'], metadata, autoload_with=self.engine)
            Session = sessionmaker(bind=self.engine)
            session = Session()
            query = session.query(table)
            df = pd.read_sql(query.statement, self.engine)
            self.feature_names = df.drop('stroke', axis=1).columns
            print(f"Datos cargados desde MySQL. Número de filas: {len(df)}")
            return df
        except Exception as e:
            print(f"Error al cargar datos desde MySQL: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        if df.empty:
            print("El DataFrame está vacío")
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
                print("La columna 'stroke' no está presente en el DataFrame")
            # Manejar el error apropiadamente, por ejemplo:
            return None, None
        
        

    def train_model(self, X, y):
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
    
    def log_plots(self, X_test, y_test):
        # Generar y registrar la curva ROC
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png')
        plt.close()
        mlflow.log_artifact('roc_curve.png')

        # Generar y registrar la importancia de las características
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.savefig('feature_importance.png')
        plt.close()
        mlflow.log_artifact('feature_importance.png')

    def initial_training(self):
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
        if self.should_check():
            print("Verificando condiciones para reentrenamiento...")
            df = self.load_data_from_mysql()
            
            if len(df) > int(os.getenv('RETRAIN_THRESHOLD', 10000)):
                print("Condiciones cumplidas. Iniciando reentrenamiento...")
                X, y = self.preprocess_data(df)
                self.train_model(X, y)
            else:
                print("No se cumplen las condiciones para reentrenar.")
            
            self.update_last_check()

    def should_check(self):
        if not os.path.exists(self.last_check_file):
            return True
        
        with open(self.last_check_file, 'r') as f:
            last_check = float(f.read().strip())
        
        return time.time() - last_check > self.check_interval

    def update_last_check(self):
        with open(self.last_check_file, 'w') as f:
            f.write(str(time.time()))

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def load_model(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

def background_worker(model):
    while True:
        model.check_and_retrain()
        time.sleep(model.check_interval)

# Inicialización del modelo y el worker en segundo plano
model = XGBoostStrokeModel(csv_path='src/Data/train_stroke_woe_smote.csv')
model.initial_training()
bg_thread = Thread(target=background_worker, args=(model,))
bg_thread.start()