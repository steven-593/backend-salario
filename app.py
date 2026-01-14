import os
import json
import numpy as np
import tensorflow as tf
import psycopg2 
from datetime import datetime 
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Carga variables de entorno (como DATABASE_URL)
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- VARIABLES GLOBALES ---
interpreter = None
input_details = None
output_details = None
mappings = None
scaler_info = None

# --- FUNCIONES DE BASE DE DATOS ---
def get_db_connection():
    """Establece conexión con PostgreSQL"""
    url = os.environ.get('DATABASE_URL')
    if not url:
        # Si no hay URL, retornamos None (para evitar crasheos en local sin configurar)
        return None
    try:
        conn = psycopg2.connect(url)
        return conn
    except Exception as e:
        print(f"❌ Error conectando a BD: {e}")
        return None

def init_db():
    """Crea las tablas y el usuario Admin si no existen"""
    try:
        conn = get_db_connection()
        if conn is None:
            print("⚠️ No hay base de datos configurada (DATABASE_URL no encontrada).")
            return
            
        cur = conn.cursor()
        
        # 1. Tabla Historial
        cur.execute('''
            CREATE TABLE IF NOT EXISTS historial_predicciones (
                id SERIAL PRIMARY KEY,
                age REAL,
                gender TEXT,
                education_level TEXT,
                job_title TEXT,
                years_experience REAL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_salary REAL
            );
        ''')

        # 2. Tabla Usuarios
        cur.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id SERIAL PRIMARY KEY,
                usuario TEXT UNIQUE NOT NULL,
                contrasena TEXT NOT NULL
            );
        ''')

        # 3. Insertar Admin por defecto (si no existe)
        cur.execute('''
            INSERT INTO usuarios (usuario, contrasena) 
            VALUES ('admin', 'admin')
            ON CONFLICT (usuario) DO NOTHING;
        ''')

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Base de datos: Tablas y Usuario Admin verificados.")
    except Exception as e:
        print(f"❌ Error iniciando BD: {e}")

# --- CARGA INICIAL DE MODELOS ---
def load_assets():
    global interpreter, input_details, output_details, mappings, scaler_info
    
    # Rutas relativas a la carpeta donde está app.py
    path_json = 'model/mappings.json'
    path_tflite = 'model/salary_model.tflite'

    try:
        # 1. Verificar existencia de archivos
        if not os.path.exists(path_json):
            raise FileNotFoundError(f"No se encontró {path_json}")
        if not os.path.exists(path_tflite):
            raise FileNotFoundError(f"No se encontró {path_tflite}")

        # 2. Cargar JSON (Mappings y Scaler)
        with open(path_json, 'r') as f:
            data = json.load(f)
            mappings = data['mappings']
            scaler_info = data['scaler']
        
        # 3. Cargar Modelo TFLite
        interpreter = tf.lite.Interpreter(model_path=path_tflite)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ TensorFlow Lite Assets cargados correctamente.")
        
        # 4. Inicializar BD
        init_db()
        
    except Exception as e:
        print(f"❌ Error fatal cargando assets: {e}")

# Ejecutamos la carga al iniciar la app
load_assets()

# --- RUTAS ---

@app.route('/', methods=['GET'])
def home():
    return "<h1>Backend AI + Login: Activo 🚀</h1>"

# RUTA LOGIN
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        usuario = data.get('usuario')
        contrasena = data.get('contrasena')

        if not usuario or not contrasena:
            return jsonify({'status': 'error', 'message': 'Faltan datos'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'status': 'error', 'message': 'Error de conexión BD'}), 500

        cur = conn.cursor()
        # Verificar credenciales
        cur.execute("SELECT * FROM usuarios WHERE usuario = %s AND contrasena = %s", (usuario, contrasena))
        user = cur.fetchone()
        
        cur.close()
        conn.close()

        if user:
            return jsonify({
                'status': 'success',
                'message': 'Login exitoso',
                'user': usuario
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Usuario o contraseña incorrectos'
            }), 401

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# RUTA HISTORIAL
@app.route('/historial', methods=['GET'])
def get_history():
    try:
        conn = get_db_connection()
        if not conn: return jsonify({'error': 'No hay conexión DB'}), 500 
        
        cur = conn.cursor()
        cur.execute("SELECT * FROM historial_predicciones ORDER BY prediction_date DESC")
        rows = cur.fetchall()
        
        historial = []
        for row in rows:
            historial.append({
                'id': row[0],
                'age': row[1],
                'gender': row[2],
                'education_level': row[3],
                'job_title': row[4],
                'years_experience': row[5],
                'prediction_date': str(row[6]),
                'predicted_salary': row[7]
            })
        cur.close()
        conn.close()
        return jsonify({'status': 'success', 'count': len(historial), 'data': historial})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# RUTA PREDICCIÓN
@app.route('/predict_api', methods=['POST'])
def predict_api():
    global interpreter, mappings, scaler_info
    try:
        if not interpreter: return jsonify({'error': 'Modelo no cargado'}), 500
        data = request.get_json()
        if not data: return jsonify({'error': 'Sin datos'}), 400

        # Mapeo de valores categóricos
        try:
            gender_val = mappings['Gender'][data['Gender']]
            edu_val = mappings['Education Level'][data['Education Level']]
            job_val = mappings['Job Title'][data['Job Title']]
        except KeyError as e:
            return jsonify({'error': f'Valor desconocido: {e}'}), 400

        # Crear vector de entrada (orden debe coincidir con entrenamiento)
        input_vector = [
            float(data['Age']),
            float(gender_val),
            float(edu_val),
            float(job_val),
            float(data['Years of Experience'])
        ]

        # Normalización manual (StandardScaler logic)
        mean = scaler_info['mean']
        scale = scaler_info['scale']
        normalized_input = [(input_vector[i] - mean[i]) / scale[i] for i in range(len(input_vector))]
        
        # Inferencia TFLite
        input_tensor = np.array([normalized_input], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = float(output_data[0][0])

        # Guardar en Base de Datos
        try:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO historial_predicciones 
                    (age, gender, education_level, job_title, years_experience, predicted_salary)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (data['Age'], data['Gender'], data['Education Level'], data['Job Title'], data['Years of Experience'], prediction))
                conn.commit()
                cur.close()
                conn.close()
                print("📝 Predicción guardada en BD.")
        except Exception as db_e:
            print(f"❌ Error guardando en BD: {db_e}")

        return jsonify({'status': 'success', 'salary': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)