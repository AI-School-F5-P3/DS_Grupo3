import pytest
import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from BBDD.create_database import create_database_and_table
from BBDD.database_utils import save_prediction_to_db
from sqlalchemy.engine.row import Row

# Cargar variables de entorno
load_dotenv()

def get_database_connection():
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = 'HEALTH_DATA'  

    connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string, echo=True)
    Session = sessionmaker(bind=engine)
    return Session()

@pytest.fixture(scope="session")
def db_params():
    return {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': 'HEALTH_DATA'  # Asegúrate de que este es el nombre correcto de tu base de datos
    }

@pytest.fixture(scope="session", autouse=True)
def setup_database(db_params):
    # Crear la base de datos y la tabla si no existen
    create_database_and_table()
    
    # Crear una conexión a la base de datos
    connection = mysql.connector.connect(**db_params)
    yield connection
    connection.close()

@pytest.fixture(scope="function")
def db_cursor(setup_database):
    cursor = setup_database.cursor(dictionary=True)
    yield cursor
    cursor.close()

def test_database_exists(db_cursor, db_params):
    db_cursor.execute("SHOW DATABASES LIKE %s", (db_params['database'],))
    result = db_cursor.fetchone()
    assert result is not None, f"Database {db_params['database']} does not exist"

def test_table_exists(db_cursor):
    db_cursor.execute("SHOW TABLES LIKE 'patient_predictions'")
    result = db_cursor.fetchone()
    assert result is not None, "Table patient_predictions does not exist"

def test_table_structure(db_cursor):
    db_cursor.execute("DESCRIBE patient_predictions")
    columns = db_cursor.fetchall()
    expected_columns = [
        {'Field': 'id', 'Type': 'int', 'Null': 'NO', 'Key': 'PRI', 'Default': None, 'Extra': 'auto_increment'},
        {'Field': 'age', 'Type': 'int', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'gender', 'Type': 'varchar(10)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'hypertension', 'Type': 'int', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'heart_disease', 'Type': 'int', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'ever_married', 'Type': 'varchar(5)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'work_type', 'Type': 'varchar(50)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'Residence_type', 'Type': 'varchar(10)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'smoking_status', 'Type': 'varchar(20)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'bmi_category', 'Type': 'varchar(20)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'age_category', 'Type': 'varchar(20)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'glucose_level_category', 'Type': 'varchar(20)', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'prediction', 'Type': 'int', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''},
        {'Field': 'prediction_probability', 'Type': 'float', 'Null': 'YES', 'Key': '', 'Default': None, 'Extra': ''}
    ]
    assert columns == expected_columns, "Table structure does not match expected structure"

def test_database_connection():
    session = get_database_connection()
    assert session is not None, "Failed to get database connection"
    session.close()

def test_save_prediction():
    test_data = pd.DataFrame({
        'age': [50],
        'gender': ['Masculino'],
        'hypertension': [1],
        'heart_disease': [0],
        'ever_married': ['Sí'],
        'work_type': ['Privado'],
        'Residence_type': ['Urbana'],
        'smoking_status': ['Exfumador'],
        'bmi_category': ['Sobrepeso'],
        'age_category': ['Adulto'],
        'glucose_level_category': ['Normal'],
        'prediction': [0],
        'prediction_probability': [0.25]
    })
    
    save_prediction_to_db(test_data)
    
    session = get_database_connection()
    result = session.execute(text("SELECT * FROM patient_predictions WHERE age = 50")).fetchone()
    session.close()
    
    assert result is not None, "Failed to save prediction to database"
    
    print(f"Type of result: {type(result)}")
    print(f"Content of result: {result}")
    print(f"Dir of result: {dir(result)}")
    
    if isinstance(result, Row):
        print("Result is a SQLAlchemy Row object")
        try:
            # Intenta acceder a los elementos del Row como un mapping
            result_dict = dict(result._mapping)
        except AttributeError:
            print("Row object doesn't have _mapping attribute")
            try:
                # Intenta acceder a los elementos del Row como una secuencia
                column_names = result._fields
                result_dict = dict(zip(column_names, result))
            except AttributeError:
                print("Row object doesn't have _fields attribute")
                # Como último recurso, intenta iterar sobre el objeto
                result_dict = {i: value for i, value in enumerate(result)}
        
        print(f"Result dictionary: {result_dict}")
    elif isinstance(result, tuple):
        print("Result is a tuple")
        print(f"Length of tuple: {len(result)}")
        print(f"Elements of tuple: {[type(elem) for elem in result]}")
        column_names = ['id', 'age', 'gender', 'hypertension', 'heart_disease', 'ever_married', 
                        'work_type', 'Residence_type', 'smoking_status', 'bmi_category', 
                        'age_category', 'glucose_level_category', 'prediction', 'prediction_probability']
        result_dict = dict(zip(column_names, result))
    elif isinstance(result, dict):
        print("Result is a dictionary")
        result_dict = result
    else:
        raise TypeError(f"Unexpected result type: {type(result)}")
    
    print(f"Final result dictionary: {result_dict}")
    
    assert 'age' in result_dict, f"'age' not found in result. Keys: {result_dict.keys()}"
    assert 'gender' in result_dict, f"'gender' not found in result. Keys: {result_dict.keys()}"
    
    assert result_dict['age'] == 50, f"Saved data does not match test data. Expected age 50, got {result_dict['age']}"
    assert result_dict['gender'] == 'Masculino', f"Saved data does not match test data. Expected gender 'Masculino', got {result_dict['gender']}"

    print("All assertions passed successfully.")

if __name__ == "__main__":
    pytest.main([__file__])