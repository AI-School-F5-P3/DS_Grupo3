import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import joblib
import mlflow
import mlflow.tensorflow

class NeuralNetworkStrokeModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = {
            'n_layers': 6,
            'n_units': 478,
            'activation': 'relu',
            'dropout_rate': 0.05,
            'batch_size': 49,
            'epochs': 229
        }

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        # Convertir columnas booleanas a enteros
        bool_columns = df.select_dtypes(include=['bool']).columns
        df[bool_columns] = df[bool_columns].astype(int)
        # Convertir valores no numéricos y eliminar filas con valores faltantes
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        return df

    def preprocess_data(self, df):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.flatten()

        # Escalado de características
        X_scaled = self.scaler.fit_transform(X)
        print("Características después del escalado:")
        print(pd.DataFrame(X_scaled).head())
        return X_scaled, y

    def build_model(self, input_shape):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=(input_shape,)))

        for _ in range(self.best_params['n_layers']):
            self.model.add(keras.layers.Dense(self.best_params['n_units'], activation=self.best_params['activation']))
            self.model.add(keras.layers.Dropout(self.best_params['dropout_rate']))

        # Capa de salida
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        # Compilación del modelo
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Construir y entrenar el modelo
            self.build_model(X_train.shape[1])
            self.model.fit(X_train, y_train, epochs=self.best_params['epochs'], batch_size=self.best_params['batch_size'], verbose=1)

            # Evaluar el modelo en el conjunto de prueba
            y_test_pred = (self.model.predict(X_test) > 0.5).astype("int32")
            test_accuracy = accuracy_score(y_test, y_test_pred)
            accuracies.append(test_accuracy)

            self.evaluate_model(X_test, y_test)

        average_accuracy = np.mean(accuracies)
        print(f"\nPrecisión promedio en los pliegues: {average_accuracy:.4f}")

    def evaluate_model(self, X_test, y_test):
        y_test_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        y_test_pred_proba = self.model.predict(X_test).flatten()

        print("\nInforme de clasificación para el conjunto de prueba:")
        print(classification_report(y_test, y_test_pred))

        print("\nMatriz de confusión para el conjunto de prueba:")
        print(confusion_matrix(y_test, y_test_pred))

        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        self.plot_roc_curve(y_test, y_test_pred_proba)

    def plot_roc_curve(self, y_test, y_test_pred_proba):
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva Característica Operativa del Receptor (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def save_model(self, model_path, scaler_path):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load_model(cls, model_path, scaler_path):
        instance = cls()
        instance.model = keras.models.load_model(model_path)
        instance.scaler = joblib.load(scaler_path)
        return instance

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return (self.model.predict(X_scaled) > 0.5).astype("int32")

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stroke_prediction_nn_experiment")

    with mlflow.start_run():
        model = NeuralNetworkStrokeModel()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'train_stroke_woe_smote.csv')
        df = model.load_data(data_path)
        X, y = model.preprocess_data(df)

        mlflow.log_params(model.best_params)
        model.train_model(X, y)

        model_path = os.path.join(current_dir, 'nn_stroke_model.keras')
        scaler_path = os.path.join(current_dir, 'nn_scaler.joblib')
        model.save_model(model_path, scaler_path)

        mlflow.tensorflow.log_model(tf_saved_model_dir=model_path, tf_meta_graph_tags=None, tf_signature_def_key=None, artifact_path="nn_stroke_model")
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(__file__)
