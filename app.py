from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from scipy.stats import skewnorm
from datetime import datetime
import time
import random

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={
    r"/*": {
        "origins": ["https://sitme.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

START_TIME = time.time()

# Cargar o crear el modelo
MODEL_PATH = 'endometriosis_model_v2.pkl'

def generate_realistic_data(n_samples=1000):
    """Genera datos sintéticos más realistas para endometriosis"""
    # Distribución de edad (sesgada hacia mujeres jóvenes)
    age = skewnorm.rvs(4, loc=25, scale=8, size=n_samples)
    age = np.clip(age, 15, 45).astype(int)
    
    # Edad de menarquia (normal alrededor de 12-13)
    menarche_age = np.random.normal(12.5, 1.5, n_samples)
    menarche_age = np.clip(menarche_age, 8, 16).astype(int)
    
    # Duración del ciclo (normal alrededor de 28 días)
    cycle_length = np.random.normal(28, 3, n_samples)
    cycle_length = np.clip(cycle_length, 21, 35).astype(int)
    
    # Duración del período (normal alrededor de 5 días)
    period_duration = np.random.normal(5, 1.5, n_samples)
    period_duration = np.clip(period_duration, 2, 10).astype(int)
    
    # Nivel de dolor (bimodal para casos con/sin endometriosis)
    pain_level = np.concatenate([
        np.random.normal(3, 1.5, int(n_samples*0.4)),
        np.random.normal(7, 1.5, int(n_samples*0.6))
    ])
    pain_level = np.clip(pain_level, 1, 10).astype(int)
    
    # Biomarcadores con distribuciones realistas
    ca125 = np.concatenate([
        np.random.uniform(5, 35, int(n_samples*0.6)),
        np.random.uniform(35, 150, int(n_samples*0.4))
    ])
    il6 = np.random.exponential(2, n_samples)
    crp = np.random.exponential(3, n_samples)
    
    # Síntomas binarios con probabilidades basadas en literatura médica
    pain_during_sex = np.random.binomial(1, 0.6, n_samples)
    family_history = np.random.binomial(1, 0.3, n_samples)
    bowel_symptoms = np.random.binomial(1, 0.5, n_samples)
    urinary_symptoms = np.random.binomial(1, 0.4, n_samples)
    fatigue = np.random.binomial(1, 0.7, n_samples)
    infertility = np.random.binomial(1, 0.4, n_samples)
    
    # Diagnóstico (endometriosis) basado en reglas clínicas
    endometriosis = (
        (pain_level > 6) & 
        ((pain_during_sex == 1) | (bowel_symptoms == 1)) & 
        ((ca125 > 35) | (family_history == 1))
    ).astype(int)
    
    # Añadir algo de ruido aleatorio
    noise = np.random.binomial(1, 0.1, n_samples)
    endometriosis = np.where(noise, 1-endometriosis, endometriosis)
    
    return {
        'age': age,
        'menarche_age': menarche_age,
        'cycle_length': cycle_length,
        'period_duration': period_duration,
        'pain_level': pain_level,
        'pain_during_sex': pain_during_sex,
        'family_history': family_history,
        'bowel_symptoms': bowel_symptoms,
        'urinary_symptoms': urinary_symptoms,
        'fatigue': fatigue,
        'infertility': infertility,
        'ca125': ca125,
        'il6': il6,
        'tnf_alpha': np.random.exponential(1.5, n_samples),
        'vegf': np.random.exponential(200, n_samples),
        'amh': np.random.normal(3.0, 1.5, n_samples),
        'crp': crp,
        'endometriosis': endometriosis
    }

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Generar datos más realistas
    data = generate_realistic_data(2000)
    df = pd.DataFrame(data)
    
    # Balancear las clases (50/50)
    df_endometriosis = df[df['endometriosis'] == 1]
    df_no_endometriosis = df[df['endometriosis'] == 0].sample(len(df_endometriosis))
    df_balanced = pd.concat([df_endometriosis, df_no_endometriosis])
    
    X = df_balanced.drop('endometriosis', axis=1)
    y = df_balanced['endometriosis']
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Crear y entrenar el modelo con calibración de probabilidades
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Calibrar el modelo para mejores probabilidades
    model = CalibratedClassifierCV(rf, method='isotonic', cv=5)
    model.fit(X_train, y_train)
    
    # Guardar el modelo
    joblib.dump(model, MODEL_PATH)

@app.route("/")
def serve_index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validar y preparar los datos de entrada
        input_data = pd.DataFrame([{
            'age': int(data.get('age', 25)),
            'menarche_age': int(data.get('menarche_age', 12)),
            'cycle_length': int(data.get('cycle_length', 28)),
            'period_duration': int(data.get('period_duration', 5)),
            'pain_level': int(data.get('pain_level', 5)),
            'pain_during_sex': int(data.get('pain_during_sex', 0)),
            'family_history': int(data.get('family_history', 0)),
            'bowel_symptoms': int(data.get('bowel_symptoms', 0)),
            'urinary_symptoms': int(data.get('urinary_symptoms', 0)),
            'fatigue': int(data.get('fatigue', 0)),
            'infertility': int(data.get('infertility', 0)),
            'ca125': float(data.get('ca125', 15.0)),
            'il6': float(data.get('il6', 2.5)),
            'tnf_alpha': float(data.get('tnf_alpha', 5.0)),
            'vegf': float(data.get('vegf', 200.0)),
            'amh': float(data.get('amh', 3.0)),
            'crp': float(data.get('crp', 2.0))
        }])
        
        # Manejar valores faltantes con imputación más inteligente
        biomarker_defaults = {
            'ca125': 15.0,
            'il6': 2.5,
            'tnf_alpha': 5.0,
            'vegf': 200.0,
            'amh': 3.0,
            'crp': 2.0
        }
        
        for col, default in biomarker_defaults.items():
            if col in input_data.columns:
                input_data[col].fillna(default, inplace=True)
            else:
                input_data[col] = default
        
        # Hacer la predicción
        prediction_prob = model.predict_proba(input_data)[0][1]
        
        # Ajustar umbrales basados en conocimiento clínico
        adjusted_prob = adjust_probability(
            prediction_prob,
            pain_level=input_data['pain_level'].iloc[0],
            ca125=input_data['ca125'].iloc[0],
            family_history=input_data['family_history'].iloc[0]
        )
        
        # Determinar nivel de riesgo con umbrales clínicos
        risk_level, recommendation = determine_risk_level(adjusted_prob, input_data.iloc[0])
        
        # Identificar factores de riesgo clave
        risk_factors = identify_risk_factors(input_data.iloc[0])
        
        return jsonify({
            'probability': float(adjusted_prob),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'patient_data': {
                'age': int(input_data['age'].iloc[0]),
                'pain_level': int(input_data['pain_level'].iloc[0]),
                'ca125': float(input_data['ca125'].iloc[0]),
                'crp': float(input_data['crp'].iloc[0])
            },
            'model_metadata': {
                'version': '2.0',
                'features_importance': get_feature_importance()
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'details': 'Error processing prediction'}), 400

def adjust_probability(base_prob, pain_level, ca125, family_history):
    """
    Ajusta la probabilidad basada en conocimiento clínico experto
    """
    # Aumentar probabilidad si hay dolor severo
    if pain_level >= 7:
        base_prob = min(1.0, base_prob * 1.3)
    
    # Aumentar si CA-125 está elevado
    if ca125 > 35:
        base_prob = min(1.0, base_prob * 1.2)
    
    # Aumentar si hay historia familiar
    if family_history == 1:
        base_prob = min(1.0, base_prob * 1.15)
    
    # Reducir si no hay síntomas clave
    if pain_level < 4 and ca125 < 20:
        base_prob = max(0.0, base_prob * 0.7)
    
    return base_prob

def determine_risk_level(probability, patient_data):
    """
    Determina el nivel de riesgo basado en probabilidad y características clínicas
    """
    # Umbrales dinámicos basados en síntomas
    base_threshold_high = 0.7
    base_threshold_moderate = 0.4
    
    # Ajustar umbrales si hay marcadores muy elevados
    if patient_data['ca125'] > 50 or patient_data['crp'] > 20:
        base_threshold_high = 0.6
        base_threshold_moderate = 0.3
    
    if probability > base_threshold_high:
        risk_level = 'high'
        recommendation = (
            "Alta probabilidad de endometriosis. "
            "Se recomienda:\n"
            "- Consulta prioritaria con especialista\n"
            "- Estudios de imagen (ecografía/resonancia)\n"
            "- Evaluación multidisciplinaria"
        )
    elif probability > base_threshold_moderate:
        risk_level = 'moderate'
        recommendation = (
            "Probabilidad moderada de endometriosis. "
            "Se recomienda:\n"
            "- Evaluación ginecológica\n"
            "- Prueba de tratamiento médico\n"
            "- Seguimiento en 3 meses"
        )
    else:
        risk_level = 'low'
        recommendation = (
            "Baja probabilidad de endometriosis. "
            "Se recomienda:\n"
            "- Monitoreo de síntomas\n"
            "- Analgesia según necesidad\n"
            "- Reevaluar si síntomas empeoran"
        )
    
    return risk_level, recommendation

def identify_risk_factors(patient_data):
    """Identifica factores de riesgo clave"""
    risk_factors = []
    
    if patient_data['pain_level'] >= 7:
        risk_factors.append("dolor_severo")
    if patient_data['pain_during_sex'] == 1:
        risk_factors.append("dispareunia")
    if patient_data['ca125'] > 35:
        risk_factors.append("ca125_elevado")
    if patient_data['family_history'] == 1:
        risk_factors.append("historia_familiar")
    if patient_data['bowel_symptoms'] == 1:
        risk_factors.append("sintomas_intestinales")
    if patient_data['urinary_symptoms'] == 1:
        risk_factors.append("sintomas_urinarios")
    if patient_data['crp'] > 10:
        risk_factors.append("inflamacion_elevada")
    if patient_data['menarche_age'] < 11:
        risk_factors.append("menarquia_temprana")
    if patient_data['cycle_length'] < 25:
        risk_factors.append("ciclos_cortos")
    
    return risk_factors

def get_feature_importance():
    """Devuelve la importancia de características del modelo"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(model.feature_names_in_, model.feature_importances_))
    elif hasattr(model, 'calibrated_classifiers_'):
        # Para modelos calibrados, obtener importancia del clasificador base
        return dict(zip(
            model.calibrated_classifiers_[0].estimator.feature_names_in_,
            model.calibrated_classifiers_[0].estimator.feature_importances_
        ))
    return {}

SERVICE_START_TIME = time.time()

def get_service_data():
    """Genera datos simulados del servicio"""
    uptime = time.time() - SERVICE_START_TIME
    hours, rem = divmod(uptime, 3600)
    minutes, seconds = divmod(rem, 60)
    
    return {
        "status": "active",
        "uptime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "cpu_usage": random.randint(5, 30),
        "memory_usage": random.randint(200, 500),
        "response_time": random.randint(10, 50),
        "requests": random.randint(1000, 5000),
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "components": {
            "API": random.choice(["online", "online", "online", "degraded"]),
            "Database": "online",
            "Cache": random.choice(["online", "online", "offline"]),
            "Auth": "online"
        },
        "logs": [
            f"{datetime.now().strftime('%H:%M:%S')} - Sistema iniciado",
            f"{datetime.now().strftime('%H:%M:%S')} - Conexión establecida con DB",
            f"{datetime.now().strftime('%H:%M:%S')} - Caché inicializada",
            f"{datetime.now().strftime('%H:%M:%S')} - Escuchando en puerto 5000"
        ]
    }

@app.route('/api/status')
def api_status():
    return get_service_data()

# Configuración para producción
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))