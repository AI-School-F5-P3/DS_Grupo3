import pandas as pd
import joblib
import os

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el archivo CSV
csv_path = os.path.join(current_dir, '..', 'data', 'test_stroke_woe.csv')
df = pd.read_csv(csv_path)

# Separar características (X) y variable objetivo (y)
X_test = df.drop('stroke', axis=1)
y_test = df['stroke']

# Guardar como archivos joblib
joblib.dump(X_test, os.path.join(current_dir, '..', 'data', 'X_test.joblib'))
joblib.dump(y_test, os.path.join(current_dir, '..', 'data', 'y_test.joblib'))

print("Archivos X_test.joblib y y_test.joblib creados con éxito.")