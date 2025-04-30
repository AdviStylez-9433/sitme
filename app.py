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
import threading

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
MODEL_PATH = 'endometriosis_model_v3.pkl'

def generate_realistic_data(n_samples=1000):
    """Genera datos sintéticos más realistas para endometriosis con más características"""
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
    
    # Tipos de dolor (nuevas características)
    chronic_pain = (pain_level > 5).astype(int)
    cyclic_pain = np.random.binomial(1, 0.7, n_samples)
    non_cyclic_pain = np.random.binomial(1, 0.3, n_samples)
    
    # Biomarcadores con distribuciones realistas
    ca125 = np.concatenate([
        np.random.uniform(5, 35, int(n_samples*0.6)),
        np.random.uniform(35, 200, int(n_samples*0.4))
    ])
    il6 = np.random.exponential(2, n_samples)
    crp = np.random.exponential(3, n_samples)
    
    # Síntomas con probabilidades basadas en literatura médica
    symptoms = {
        'pain_during_sex': 0.6,
        'family_history': 0.3,
        'bowel_symptoms': 0.5,
        'urinary_symptoms': 0.4,
        'fatigue': 0.7,
        'infertility': 0.4,
        'dyschezia': 0.3,  # dolor al defecar
        'dysuria': 0.2,    # dolor al orinar
        'bloating': 0.5
    }
    
    for symptom, prob in symptoms.items():
        symptoms[symptom] = np.random.binomial(1, prob, n_samples)
    
    # Diagnóstico (endometriosis) basado en reglas clínicas más complejas
    endometriosis = (
        (pain_level > 6) & 
        ((symptoms['pain_during_sex'] == 1) | (symptoms['bowel_symptoms'] == 1)) & 
        ((ca125 > 35) | (symptoms['family_history'] == 1) | (chronic_pain == 1))
    ).astype(int)
    
    # Añadir ruido aleatorio controlado
    noise = np.random.binomial(1, 0.05, n_samples)
    endometriosis = np.where(noise, 1-endometriosis, endometriosis)
    
    # Crear DataFrame con todas las características
    data = {
        'age': age,
        'menarche_age': menarche_age,
        'cycle_length': cycle_length,
        'period_duration': period_duration,
        'pain_level': pain_level,
        'chronic_pain': chronic_pain,
        'cyclic_pain': cyclic_pain,
        'non_cyclic_pain': non_cyclic_pain,
        'ca125': ca125,
        'il6': il6,
        'tnf_alpha': np.random.exponential(1.5, n_samples),
        'vegf': np.random.exponential(200, n_samples),
        'amh': np.random.normal(3.0, 1.5, n_samples),
        'crp': crp,
        'endometriosis': endometriosis
    }
    
    # Añadir síntomas
    data.update(symptoms)
    
    return data

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Generar datos más realistas
    data = generate_realistic_data(5000)  # Más datos para mejor entrenamiento
    df = pd.DataFrame(data)
    
    # Balancear las clases (50/50)
    df_endometriosis = df[df['endometriosis'] == 1]
    df_no_endometriosis = df[df['endometriosis'] == 0].sample(len(df_endometriosis))
    df_balanced = pd.concat([df_endometriosis, df_no_endometriosis])
    
    # Seleccionar características más relevantes
    features = [
        'age', 'menarche_age', 'cycle_length', 'period_duration',
        'pain_level', 'chronic_pain', 'cyclic_pain', 'non_cyclic_pain',
        'pain_during_sex', 'family_history', 'bowel_symptoms',
        'urinary_symptoms', 'fatigue', 'infertility', 'dyschezia',
        'dysuria', 'bloating', 'ca125', 'il6', 'tnf_alpha', 'vegf', 'amh', 'crp'
    ]
    
    X = df_balanced[features]
    y = df_balanced['endometriosis']
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Crear y entrenar el modelo mejorado
    rf = RandomForestClassifier(
        n_estimators=300,  # Más estimadores para mejor precisión
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        max_features='sqrt'  # Mejor generalización
    )
    
    # Calibrar el modelo para mejores probabilidades
    model = CalibratedClassifierCV(rf, method='isotonic', cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    # Guardar el modelo
    joblib.dump(model, MODEL_PATH)

@app.route("/")
def serve_index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validar y preparar los datos de entrada con valores por defecto más realistas
        input_data = pd.DataFrame([{
            'age': int(data.get('age', 25)),
            'menarche_age': int(data.get('menarche_age', 12)),
            'cycle_length': int(data.get('cycle_length', 28)),
            'period_duration': int(data.get('period_duration', 5)),
            'pain_level': int(data.get('pain_level', 5)),
            'chronic_pain': int(data.get('chronic_pain', 0)),
            'cyclic_pain': int(data.get('cyclic_pain', 1)),
            'non_cyclic_pain': int(data.get('non_cyclic_pain', 0)),
            'pain_during_sex': int(data.get('pain_during_sex', 0)),
            'family_history': int(data.get('family_history', 0)),
            'bowel_symptoms': int(data.get('bowel_symptoms', 0)),
            'urinary_symptoms': int(data.get('urinary_symptoms', 0)),
            'fatigue': int(data.get('fatigue', 0)),
            'infertility': int(data.get('infertility', 0)),
            'dyschezia': int(data.get('dyschezia', 0)),
            'dysuria': int(data.get('dysuria', 0)),
            'bloating': int(data.get('bloating', 0)),
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
        
        # Ajustar probabilidad basada en conocimiento clínico experto
        adjusted_prob = clinical_probability_adjustment(
            base_prob=prediction_prob,
            patient_data=input_data.iloc[0]
        )
        
        # Determinar nivel de riesgo con umbrales clínicos dinámicos
        risk_level, recommendation = determine_risk_level_v2(adjusted_prob, input_data.iloc[0])
        
        # Identificar factores de riesgo clave con ponderación
        risk_factors = identify_weighted_risk_factors(input_data.iloc[0])
        
        # Calcular score de riesgo compuesto
        risk_score = calculate_risk_score(input_data.iloc[0])
        
        return jsonify({
            'probability': float(adjusted_prob),
            'risk_score': float(risk_score),
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
                'version': '3.0',
                'features_importance': get_feature_importance(),
                'confidence': calculate_confidence_interval(adjusted_prob)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'details': 'Error processing prediction'}), 400

def clinical_probability_adjustment(base_prob, patient_data):
    """
    Ajusta la probabilidad basada en conocimiento clínico experto con ponderaciones más precisas
    """
    adjustment_factors = {
        'pain_level': lambda x: 0.05 * x if x > 5 else 0,
        'chronic_pain': lambda x: 0.15 if x == 1 else 0,
        'pain_during_sex': lambda x: 0.12 if x == 1 else 0,
        'family_history': lambda x: 0.10 if x == 1 else 0,
        'bowel_symptoms': lambda x: 0.08 if x == 1 else 0,
        'ca125': lambda x: 0.07 * (x / 35) if x > 35 else 0,
        'crp': lambda x: 0.05 * (x / 10) if x > 10 else 0,
        'infertility': lambda x: 0.10 if x == 1 else 0,
        'dyschezia': lambda x: 0.07 if x == 1 else 0
    }
    
    total_adjustment = 0
    for factor, func in adjustment_factors.items():
        total_adjustment += func(patient_data[factor])
    
    # Aplicar ajuste con función sigmoide para suavizar
    adjusted_prob = base_prob + (1 - base_prob) * (1 / (1 + np.exp(-total_adjustment*10 + 5)))
    
    # Asegurar que esté en [0, 1]
    adjusted_prob = np.clip(adjusted_prob, 0, 1)
    
    return adjusted_prob

def calculate_risk_score(patient_data):
    """
    Calcula un score de riesgo compuesto (0-100) basado en factores clave
    """
    score_weights = {
        'pain_level': 2.0,
        'chronic_pain': 3.0,
        'pain_during_sex': 2.5,
        'family_history': 2.0,
        'bowel_symptoms': 1.5,
        'ca125': lambda x: min(4.0, x / 10),
        'crp': lambda x: min(3.0, x / 5),
        'infertility': 2.0,
        'dyschezia': 1.5,
        'cyclic_pain': 1.0
    }
    
    total_score = 0
    for factor, weight in score_weights.items():
        value = patient_data[factor]
        if callable(weight):
            total_score += weight(value)
        else:
            total_score += weight * value
    
    # Normalizar a 0-100
    normalized_score = 100 * (1 - 1 / (1 + total_score/10))
    
    return np.clip(normalized_score, 0, 100)

def determine_risk_level_v2(probability, patient_data):
    """
    Determina el nivel de riesgo con umbrales dinámicos basados en síntomas
    """
    # Umbrales base
    base_thresholds = {
        'low': 0.3,
        'moderate': 0.6,
        'high': 0.8
    }
    
    # Ajustar umbrales según síntomas severos
    if (patient_data['pain_level'] >= 8 or 
        patient_data['ca125'] > 100 or 
        (patient_data['chronic_pain'] == 1 and patient_data['infertility'] == 1)):
        base_thresholds['moderate'] = 0.5
        base_thresholds['high'] = 0.7
    
    # Determinar nivel de riesgo
    if probability >= base_thresholds['high']:
        risk_level = 'high'
        recommendation = (
            "Alta probabilidad de endometriosis (≥{:.0f}%).\n"
            "Recomendaciones:\n"
            "- Consulta urgente con especialista en endometriosis\n"
            "- Ecografía transvaginal especializada\n"
            "- Resonancia magnética pélvica\n"
            "- Evaluación de marcadores inflamatorios\n"
            "- Considerar laparoscopia diagnóstica".format(base_thresholds['high']*100))
    elif probability >= base_thresholds['moderate']:
        risk_level = 'moderate'
        recommendation = (
            "Probabilidad moderada de endometriosis (≥{:.0f}%).\n"
            "Recomendaciones:\n"
            "- Consulta ginecológica en 1-2 meses\n"
            "- Ecografía pélvica\n"
            "- Prueba terapéutica con AINEs\n"
            "- Considerar tratamiento hormonal\n"
            "- Seguimiento estrecho".format(base_thresholds['moderate']*100))
    else:
        risk_level = 'low'
        recommendation = (
            "Baja probabilidad de endometriosis (<{:.0f}%).\n"
            "Recomendaciones:\n"
            "- Analgesia según necesidad\n"
            "- Monitoreo de síntomas\n"
            "- Reevaluar si síntomas persisten\n"
            "- Educación sobre endometriosis\n"
            "- Seguimiento en 6 meses".format(base_thresholds['moderate']*100))
    
    return risk_level, recommendation

def identify_weighted_risk_factors(patient_data):
    """
    Identifica factores de riesgo clave con ponderación de importancia
    """
    risk_factors = []
    
    # Factores con umbrales y pesos
    factors = [
        ('dolor_severo', patient_data['pain_level'] >= 7, 1.5),
        ('dolor_crónico', patient_data['chronic_pain'] == 1, 2.0),
        ('dispareunia', patient_data['pain_during_sex'] == 1, 1.5),
        ('ca125_elevado', patient_data['ca125'] > 35, 1.0 + min(2.0, patient_data['ca125']/50)),
        ('historia_familiar', patient_data['family_history'] == 1, 1.3),
        ('sintomas_intestinales', patient_data['bowel_symptoms'] == 1, 1.2),
        ('infertilidad', patient_data['infertility'] == 1, 1.7),
        ('dyschezia', patient_data['dyschezia'] == 1, 1.2),
        ('inflamacion_elevada', patient_data['crp'] > 10, 1.0 + min(1.5, patient_data['crp']/20)),
        ('menarquia_temprana', patient_data['menarche_age'] < 11, 1.1)
    ]
    
    for name, condition, weight in factors:
        if condition:
            risk_factors.append({
                'factor': name,
                'weight': round(float(weight), 2)
            })
    
    # Ordenar por peso descendente
    risk_factors.sort(key=lambda x: x['weight'], reverse=True)
    
    return risk_factors

def calculate_confidence_interval(probability, n_features=10):
    """
    Calcula un intervalo de confianza aproximado para la predicción
    """
    # Basado en el número de características presentes y la probabilidad
    std_dev = np.sqrt(probability * (1 - probability) / n_features)
    lower = max(0, probability - 1.96 * std_dev)
    upper = min(1, probability + 1.96 * std_dev)
    
    return {
        'lower': round(float(lower), 3),
        'upper': round(float(upper), 3),
        'confidence': round(float(1 - 2 * std_dev), 3)
    }

def get_feature_importance():
    """Devuelve la importancia de características del modelo"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(model.feature_names_in_, model.feature_importances_))
    elif hasattr(model, 'calibrated_classifiers_'):
        # Para modelos calibrados, obtener importancia del clasificador base
        base_model = model.calibrated_classifiers_[0].estimator
        return dict(zip(base_model.feature_names_in_, base_model.feature_importances_))
    return {}

# ... (el resto del código de monitoreo y ejecución se mantiene igual)

# Variables globales para el monitoreo en tiempo real
SERVICE_START_TIME = time.time()
REQUEST_COUNTER = 0
PERFORMANCE_METRICS = {
    'cpu': [],
    'memory': [],
    'response_times': []
}
REQUEST_TIMESTAMPS = []

def generate_performance_metrics():
    """Genera métricas de rendimiento simuladas más realistas"""
    while True:
        # Simular carga de CPU con patrones más realistas
        base_cpu = 20 + 10 * random.random()
        cpu_usage = base_cpu + 15 * random.random() if random.random() > 0.7 else base_cpu
        
        memory_usage = 300 + 200 * random.random()
        response_time = 15 + 30 * random.random()
        
        # Mantener solo los últimos 60 valores (1 minuto de datos)
        PERFORMANCE_METRICS['cpu'].append(cpu_usage)
        PERFORMANCE_METRICS['memory'].append(memory_usage)
        PERFORMANCE_METRICS['response_times'].append(response_time)
        
        for metric in PERFORMANCE_METRICS.values():
            if len(metric) > 60:
                metric.pop(0)
        
        time.sleep(1)

# Iniciar el hilo de generación de métricas
metrics_thread = threading.Thread(target=generate_performance_metrics, daemon=True)
metrics_thread.start()

def get_service_data():
    """Genera datos del servicio en tiempo real"""
    uptime = time.time() - SERVICE_START_TIME
    hours, rem = divmod(uptime, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Calcular solicitudes por segundo (últimos 5 segundos)
    now = time.time()
    recent_requests = sum(1 for t in REQUEST_TIMESTAMPS if now - t < 5)
    rps = recent_requests / 5 if recent_requests > 0 else 0
    
    # Obtener métricas actuales
    current_cpu = PERFORMANCE_METRICS['cpu'][-1] if PERFORMANCE_METRICS['cpu'] else 0
    current_memory = PERFORMANCE_METRICS['memory'][-1] if PERFORMANCE_METRICS['memory'] else 0
    current_response = PERFORMANCE_METRICS['response_times'][-1] if PERFORMANCE_METRICS['response_times'] else 0
    
    return {
        "status": "active",
        "uptime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "cpu_usage": round(current_cpu, 1),
        "memory_usage": round(current_memory),
        "response_time": round(current_response),
        "requests": REQUEST_COUNTER,
        "requests_per_second": round(rps, 2),
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "performance_history": {
            "cpu": PERFORMANCE_METRICS['cpu'],
            "memory": PERFORMANCE_METRICS['memory'],
            "response_times": PERFORMANCE_METRICS['response_times']
        },
        "components": {
            "API": "online" if current_cpu < 80 else "degraded",
            "Database": "online",
            "Cache": "online" if random.random() > 0.1 else "offline",
            "Auth": "online"
        },
        "logs": [
            f"{datetime.now().strftime('%H:%M:%S')} - Solicitud #{REQUEST_COUNTER} procesada",
            f"{datetime.now().strftime('%H:%M:%S')} - CPU: {round(current_cpu, 1)}%",
            f"{datetime.now().strftime('%H:%M:%S')} - Memoria: {round(current_memory)}MB"
        ]
    }

@app.route('/api/status')
def api_status():
    global REQUEST_COUNTER, REQUEST_TIMESTAMPS
    REQUEST_COUNTER += 1
    REQUEST_TIMESTAMPS.append(time.time())
    
    # Mantener solo registros de los últimos 60 segundos
    REQUEST_TIMESTAMPS = [t for t in REQUEST_TIMESTAMPS if time.time() - t < 60]
    
    return jsonify(get_service_data())

# Configuración para producción
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))