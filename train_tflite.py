import pandas as pd
import numpy as np
import tensorflow as tf
import os  # Para manejar carpetas
import json # Para guardar los diccionarios
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def entrenar_modelo_tflite():
    print("--- INICIANDO ENTRENAMIENTO TFLITE ---")

    # --- PASO 0: CONFIGURACIÓN DE CARPETAS ---
    # CAMBIO AQUÍ: Ahora apuntamos a la carpeta 'model' en lugar de 'assets'
    output_folder = 'model' 
    
    # Creamos la carpeta 'model' si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📂 Carpeta '{output_folder}' creada/verificada.")

    # --- PASO 1: CARGA DE DATOS ---
    archivo_csv = 'salarydataset.csv'
    if not os.path.exists(archivo_csv):
        print(f"❌ Error: No se encontró el archivo '{archivo_csv}'")
        return

    df = pd.read_csv(archivo_csv)
    
    # --- PASO 2: LIMPIEZA DE DATOS ---
    df.dropna(subset=['Salary'], inplace=True)
    
    # Estandarizamos texto
    df['Education Level'] = df['Education Level'].replace({
        "Bachelor's": "Bachelor's Degree",
        "Master's": "Master's Degree",
        "phD": "PhD"
    })

    # --- PASO 3: PREPROCESAMIENTO (Codificación) ---
    encoders_map = {}
    categorical_cols = ['Gender', 'Education Level', 'Job Title']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders_map[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}

    # Definir X (Entradas) e y (Salida)
    X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']].values.astype(np.float32)
    y = df['Salary'].values.astype(np.float32)

    # --- PASO 4: NORMALIZACIÓN (SCALER) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    full_export_data = {
        "mappings": encoders_map,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }
    }

    # Guardamos en model/mappings.json
    json_path = os.path.join(output_folder, 'mappings.json')
    with open(json_path, 'w') as f:
        json.dump(full_export_data, f)
    print(f"✅ Mappings y Scaler exportados a '{json_path}'")

    # --- PASO 5: DIVISIÓN TRAIN/TEST ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- PASO 6: CREAR MODELO (Red Neuronal) ---
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)), 
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1) 
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("🧠 Entrenando red neuronal...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Evaluación rápida
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Error Promedio (MAE): ${mae:.2f}")

    # --- PASO 7: EXPORTAR A TFLITE ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Guardamos en model/salary_model.tflite
    tflite_path = os.path.join(output_folder, 'salary_model.tflite')
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Modelo móvil exportado a '{tflite_path}'")
    print("--- PROCESO TERMINADO ---")

if __name__ == "__main__":
    entrenar_modelo_tflite()