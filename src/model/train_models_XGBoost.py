import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

class XGBoostStrokeModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        self.feature_names = df.drop('stroke', axis=1).columns
        print("Columnas del dataset:")
        print(df.columns)
        return df

    def preprocess_data(self, df):
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        X_scaled = self.scaler.fit_transform(X)
        print("Características después del preprocesamiento:")
        print(pd.DataFrame(X_scaled, columns=X.columns).head())
        return X_scaled, y

    def train_model(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []

        # Calcular el peso de las clases
        class_0_weight = (y == 0).sum()
        class_1_weight = (y == 1).sum()
        scale_pos_weight = class_0_weight / class_1_weight

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Entrenamiento del modelo con `scale_pos_weight`
            self.model = xgb.XGBClassifier(
                n_estimators=425,
                learning_rate=0.04,
                max_depth=8,
                min_child_weight=1,
                subsample=0.55,
                colsample_bytree=0.88,
                random_state=42,
                scale_pos_weight=scale_pos_weight  # Añadir el peso de la clase positiva
            )
            self.model.fit(X_train, y_train)

            y_test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            accuracies.append(test_accuracy)

            self.evaluate_model(X_test, y_test)

        average_accuracy = sum(accuracies) / len(accuracies)
        print(f"\nAverage Accuracy across folds: {average_accuracy:.4f}")

    def evaluate_model(self, X_test, y_test, output_dir='output'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Save classification report
        report = classification_report(y_test, y_test_pred)
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        with open(os.path.join(output_dir, 'confusion_matrix.txt'), 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))

        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        with open(os.path.join(output_dir, 'roc_auc_score.txt'), 'w') as f:
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n")

        self.plot_roc_curve(y_test, y_test_pred_proba, output_dir)
        self.plot_feature_importance(output_dir)

    def plot_roc_curve(self, y_test, y_test_pred_proba, output_dir):
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)

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
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

    def plot_feature_importance(self, output_dir):
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature importance:")
        print(importance_df)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
    @classmethod
    def load_model(cls, model_path, scaler_path):
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        return instance

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    

if __name__ == "__main__":
    model = XGBoostStrokeModel()
    # Construir la ruta relativa al archivo de datos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'train_stroke_woe_smote.csv')
    df = model.load_data(data_path)
    X, y = model.preprocess_data(df)
    model.train_model(X, y)
    
    # Ajustar las rutas para guardar el modelo y el scaler
    model_path = os.path.join(current_dir, 'xgboost_model.joblib')
    scaler_path = os.path.join(current_dir, 'xgb_scaler.joblib')
    model.save_model(model_path, scaler_path)
    