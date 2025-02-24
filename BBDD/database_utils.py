from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from sqlalchemy.orm import sessionmaker
from BBDD.models import Base

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def setup_database():
    load_dotenv()
    
    # Obtener los parámetros de conexión a MySQL desde las variables de entorno
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')
    
    # Crear la conexión a la base de datos
    engine = create_engine(f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    # Crear las tablas si no existen
    Base.metadata.create_all(engine)

    return engine

def get_database_connection():
    engine = setup_database()
    Session = sessionmaker(bind=engine)
    return Session()

