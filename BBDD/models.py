from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ModelMetrics(Base):
        __tablename__ = 'model_metrics'
        id = Column(Integer, primary_key=True)
        avg_prediction = Column(Float)
        entropy = Column(Float)
        ks_statistic = Column(Float)
        timestamp = Column(DateTime, default=datetime.now)

    # Definir el modelo de predicciones de pacientes
class PatientPredictions(Base):
    __tablename__ = 'patient_predictions'
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    gender = Column(String(10))
    hypertension = Column(Integer)
    heart_disease = Column(Integer)
    ever_married = Column(String(5))
    work_type = Column(String(50))
    Residence_type = Column(String(10))
    smoking_status = Column(String(20))
    bmi_category = Column(String(20))
    age_category = Column(String(20))
    glucose_level_category = Column(String(20))
    prediction = Column(Integer)
    prediction_probability = Column(Float)

class PatientData(Base):
    __tablename__ = 'new_data'
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    gender = Column(String(10))
    hypertension = Column(Integer)
    heart_disease = Column(Integer)
    ever_married = Column(String(5))
    work_type = Column(String(50))
    Residence_type = Column(String(10))
    smoking_status = Column(String(20))
    bmi_category = Column(String(20))
    age_category = Column(String(20))
    glucose_level_category = Column(String(20))
    stroke = Column(Integer)