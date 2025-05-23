from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
import time
import threading
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame, PageTemplate, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import json
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
import stripe

# Configuración inicial
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuración de JWT
app.config['SECRET_KEY'] = 'k!ojiTN8oMJV'  # Cambia esto en producción!
app.config['JWT_EXPIRATION_DELTA'] = timedelta(hours=24)  # Tokens expiran en 24 horas

#Conexión a la base de datos PostgreSQL
DATABASE_URL = "postgresql://postgres.vsivmttzpipxffpywdfg:lbejTpKfjUu6Xrbl@aws-0-us-east-2.pooler.supabase.com:6543/postgres?sslmode=require"

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializar SQLAlchemy
db = SQLAlchemy(app)

# Función para obtener conexión a PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Función para crear la tabla si no existe (ejecutar al inicio)
def init_db():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Crear tabla patient_simulations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_simulations (
                -- Identificación
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                clinic_id VARCHAR(20),
                
                -- Datos personales
                full_name VARCHAR(255),
                id_number VARCHAR(12),  -- RUT
                birth_date DATE,
                age INTEGER,
                blood_type VARCHAR(3),
                insurance VARCHAR(20),
                
                -- Antecedentes médicos
                gynecological_surgery BOOLEAN,
                pelvic_inflammatory BOOLEAN,
                ovarian_cysts BOOLEAN,
                family_endometriosis BOOLEAN,
                family_autoimmune BOOLEAN,
                family_cancer BOOLEAN,
                comorbidity_autoimmune BOOLEAN,
                comorbidity_thyroid BOOLEAN,
                comorbidity_ibs BOOLEAN,
                medications TEXT,
                
                -- Historia menstrual
                menarche_age INTEGER,
                cycle_length INTEGER,
                period_duration INTEGER,
                last_period DATE,
                pain_level INTEGER,
                pain_premenstrual BOOLEAN,
                pain_menstrual BOOLEAN,
                pain_ovulation BOOLEAN,
                pain_chronic BOOLEAN,
                
                -- Síntomas actuales
                pain_during_sex BOOLEAN,
                bowel_symptoms BOOLEAN,
                urinary_symptoms BOOLEAN,
                fatigue BOOLEAN,
                infertility BOOLEAN,
                other_symptoms TEXT,
                
                -- Biomarcadores
                ca125 NUMERIC,
                il6 NUMERIC,
                tnf_alpha NUMERIC,
                vegf NUMERIC,
                amh NUMERIC,
                crp NUMERIC,
                imaging VARCHAR(50),
                imaging_details TEXT,
                
                -- Examen físico
                height NUMERIC,
                weight NUMERIC,
                bmi NUMERIC,
                pelvic_exam VARCHAR(255),
                vaginal_exam VARCHAR(255),
                clinical_notes TEXT,
                
                -- Resultados de la predicción
                probability NUMERIC(5,2),
                risk_level VARCHAR(20),
                model_version VARCHAR(20),
                
                -- Recomendaciones (podrían normalizarse en otra tabla)
                recommendations TEXT[]
            );
        """)
        
        # Crear tabla medicos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medicos (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                nombre VARCHAR(100),
                colegiado VARCHAR(50) UNIQUE,
                creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ultimo_login TIMESTAMP,
                activo BOOLEAN DEFAULT TRUE
            );
        """)
        
        # Opcional: Crear índice para búsquedas por email
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_medicos_email ON medicos(email);
        """)
        
        conn.commit()
        app.logger.info("Tablas creadas/existentes: patient_simulations y medicos")
        
    except Exception as e:
        app.logger.error(f"Error initializing database: {str(e)}")
        raise e  # Re-lanzar la excepción para que no pase desapercibida
    finally:
        if conn:
            conn.close()

# Ejecutar init_db al iniciar la aplicación
init_db()

# Clase Medico (actualizada)
class Medico(db.Model):
    __tablename__ = 'medicos'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    nombre = db.Column(db.String(100))
    colegiado = db.Column(db.String(50), unique=True)
    creado_en = db.Column(db.DateTime, default=datetime.utcnow)
    ultimo_login = db.Column(db.DateTime)
    activo = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_auth_token(self):
        payload = {
            'user_id': self.id,
            'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA']
        }
        return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# Decorador para rutas protegidaspsql -h db.vsivmttzpipxffpywdfg.supabase.co -p 5432 -U postgres -d postgres
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Obtener token del header
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
            
        if not token:
            return jsonify({'message': 'Token es requerido'}), 401
            
        try:
            # Decodificar token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = Medico.query.get(data['user_id'])
            
            if not current_user or not current_user.activo:
                return jsonify({'message': 'Usuario no válido o inactivo'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 401
        except (jwt.InvalidTokenError, Exception) as e:
            return jsonify({'message': 'Token inválido', 'error': str(e)}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated_function

# Ruta de Login
@app.route('/api/login', methods=['POST'])
def login():
    try:
        auth = request.get_json()
        
        # Validación más robusta
        if not auth or not isinstance(auth, dict):
            return jsonify({'success': False, 'error': 'Datos inválidos'}), 400
            
        email = auth.get('email')
        password = auth.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'error': 'Email y contraseña son requeridos'}), 400
            
        medico = Medico.query.filter_by(email=email).first()
        
        if not medico:
            return jsonify({'success': False, 'error': 'Credenciales inválidas'}), 401
            
        if not medico.check_password(password):
            return jsonify({'success': False, 'error': 'Credenciales inválidas'}), 401
            
        if not medico.activo:
            return jsonify({'success': False, 'error': 'Cuenta desactivada'}), 403
            
        # Actualizar último login
        medico.ultimo_login = datetime.utcnow()
        db.session.commit()
        
        # Generar token
        auth_token = medico.generate_auth_token()
        
        response = {
            'success': True,
            'message': 'Login exitoso',
            'token': auth_token,
            'user': {
                'id': medico.id,
                'email': medico.email,
                'nombre': medico.nombre,
                'colegiado': medico.colegiado,
                'ultimo_login': medico.ultimo_login.isoformat()
            }
        }
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'Error en el servidor'}), 500
    
# Añade esta nueva ruta al final de app.py, antes del if __name__ == '__main__':
@app.route('/save_simulation', methods=['POST'])
def save_simulation():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        # Extraer datos del formulario
        form_data = data.get('form_data', {})
        prediction = data.get('prediction', {})
        
        # Validar datos requeridos
        if not form_data or not prediction:
            return jsonify({'error': 'Datos incompletos'}), 400
        
        # Datos personales
        personal = form_data.get('personal', {})
        history = form_data.get('history', {})
        menstrual = form_data.get('menstrual', {})
        symptoms = form_data.get('symptoms', {})
        biomarkers = form_data.get('biomarkers', {})
        examination = form_data.get('examination', {})
        
        # Preparar consulta SQL con todos los campos
        query = """
            INSERT INTO patient_simulations (
                clinic_id,
                full_name, id_number, birth_date, age, blood_type, insurance,
                gynecological_surgery, pelvic_inflammatory, ovarian_cysts,
                family_endometriosis, family_autoimmune, family_cancer,
                comorbidity_autoimmune, comorbidity_thyroid, comorbidity_ibs,
                medications,
                menarche_age, cycle_length, period_duration, last_period,
                pain_level, pain_premenstrual, pain_menstrual, pain_ovulation, pain_chronic,
                pain_during_sex, bowel_symptoms, urinary_symptoms, fatigue, infertility,
                other_symptoms,
                ca125, il6, tnf_alpha, vegf, amh, crp,
                imaging, imaging_details,
                height, weight, bmi, pelvic_exam, vaginal_exam, clinical_notes,
                probability, risk_level, model_version, recommendations
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Preparar parámetros en el orden exacto de la consulta
        params = [
            # ID clínico
            data.get('clinic_id', ''),
            
            # Datos personales (6 campos)
            personal.get('full_name'),
            personal.get('id_number'),
            personal.get('birth_date'),
            int(personal.get('age', 0)) if personal.get('age') else None,
            personal.get('blood_type'),
            personal.get('insurance'),
            
            # Antecedentes médicos (10 campos)
            history.get('gynecological_surgery', False),
            history.get('pelvic_inflammatory', False),
            history.get('ovarian_cysts', False),
            history.get('family_endometriosis', False),
            history.get('family_autoimmune', False),
            history.get('family_cancer', False),
            history.get('comorbidity_autoimmune', False),
            history.get('comorbidity_thyroid', False),
            history.get('comorbidity_ibs', False),
            history.get('medications'),
            
            # Historia menstrual (9 campos)
            int(menstrual.get('menarche_age')) if menstrual.get('menarche_age') else None,
            int(menstrual.get('cycle_length')) if menstrual.get('cycle_length') else None,
            int(menstrual.get('period_duration')) if menstrual.get('period_duration') else None,
            menstrual.get('last_period'),
            int(menstrual.get('pain_level')) if menstrual.get('pain_level') else None,
            menstrual.get('pain_premenstrual', False),
            menstrual.get('pain_menstrual', False),
            menstrual.get('pain_ovulation', False),
            menstrual.get('pain_chronic', False),
            
            # Síntomas (6 campos)
            symptoms.get('pain_during_sex', False),
            symptoms.get('bowel_symptoms', False),
            symptoms.get('urinary_symptoms', False),
            symptoms.get('fatigue', False),
            symptoms.get('infertility', False),
            symptoms.get('other_symptoms'),
            
            # Biomarcadores (8 campos)
            float(biomarkers.get('ca125')) if biomarkers.get('ca125') else None,
            float(biomarkers.get('il6')) if biomarkers.get('il6') else None,
            float(biomarkers.get('tnf_alpha')) if biomarkers.get('tnf_alpha') else None,
            float(biomarkers.get('vegf')) if biomarkers.get('vegf') else None,
            float(biomarkers.get('amh')) if biomarkers.get('amh') else None,
            float(biomarkers.get('crp')) if biomarkers.get('crp') else None,
            biomarkers.get('imaging'),
            biomarkers.get('imaging_details'),
            
            # Examen físico (6 campos)
            float(examination.get('height')) if examination.get('height') else None,
            float(examination.get('weight')) if examination.get('weight') else None,
            float(examination.get('bmi')) if examination.get('bmi') else None,
            examination.get('pelvic_exam'),
            examination.get('vaginal_exam'),
            examination.get('clinical_notes'),
            
            # Resultados de la predicción (3 campos)
            float(prediction.get('probability', 0)),
            prediction.get('risk_level', 'unknown'),
            prediction.get('model_version', 'v4.1-xgboost'),
            
            # Recomendaciones (1 campo)
            prediction.get('recommendations', [])
        ]
        
        # Verificar que el número de parámetros coincida con los marcadores
        expected_params = query.count('%s')
        if len(params) != expected_params:
            raise ValueError(f"Número incorrecto de parámetros. Esperados: {expected_params}, Obtenidos: {len(params)}")
        
        cursor.execute(query, params)
        simulation_id = cursor.fetchone()['id']
        conn.commit()
        
        return jsonify({
            'success': True,
            'simulation_id': simulation_id,
            'message': 'Simulación guardada exitosamente'
        })
        
    except Exception as e:
        app.logger.error(f"Error guardando simulación: {str(e)}")
        return jsonify({
            'error': 'Error al guardar la simulación',
            'details': str(e),
            'trace': traceback.format_exc()
        }), 500
    finally:
        if 'conn' in locals():
            conn.close()
            
# Añade esta ruta al app.py, antes del if __name__ == '__main__':
@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        # Obtener parámetros de paginación (valores por defecto: page=1, limit=10)
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        search_term = request.args.get('search', '').strip()
        
        # Calcular offset
        offset = (page - 1) * limit
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Consulta base
        query = """
            SELECT 
                id, clinic_id, full_name, id_number as rut, age, 
                to_char(created_at, 'DD/MM/YYYY') as evaluation_date,
                risk_level as risk,
                probability
            FROM patient_simulations
        """
        
        # Consulta para contar el total de registros (con filtro de búsqueda si existe)
        count_query = "SELECT COUNT(*) as total FROM patient_simulations"
        
        # Añadir condiciones de búsqueda si hay término
        where_clause = ""
        params = []
        if search_term:
            where_clause = """
                WHERE full_name ILIKE %s 
                OR id_number ILIKE %s 
                OR clinic_id ILIKE %s
            """
            search_param = f"%{search_term}%"
            params = [search_param, search_param, search_param]
        
        # Consulta para los registros paginados
        paginated_query = f"""
            {query}
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        
        # Consulta para el total de registros
        total_query = f"{count_query} {where_clause}"
        
        # Ejecutar consulta paginada
        cursor.execute(paginated_query, params + [limit, offset])
        records = cursor.fetchall()
        
        # Ejecutar consulta para total
        cursor.execute(total_query, params)
        total = cursor.fetchone()['total']
        
        return jsonify({
            'success': True,
            'records': records,
            'total': total,
            'page': page,
            'limit': limit,
            'total_pages': (total + limit - 1) // limit  # Cálculo de páginas totales
        })
        
    except Exception as e:
        app.logger.error(f"Error obteniendo historial: {str(e)}")
        return jsonify({
            'error': 'Error al obtener historial',
            'details': str(e)
        }), 500
    finally:
        if 'conn' in locals():
            conn.close()
            
# Añade esta ruta al app.py
@app.route('/delete_record/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM patient_simulations WHERE id = %s", (record_id,))
        conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Registro eliminado correctamente'
        })
        
    except Exception as e:
        app.logger.error(f"Error eliminando registro: {str(e)}")
        return jsonify({
            'error': 'Error al eliminar registro',
            'details': str(e)
        }), 500
    finally:
        if 'conn' in locals():
            conn.close()
            
@app.route('/get_record_details/<int:record_id>')
def get_record_details(record_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM patient_simulations 
            WHERE id = %s
        """, (record_id,))
        
        record = cursor.fetchone()
        
        if not record:
            return jsonify({'error': 'Registro no encontrado'}), 404
            
        return jsonify({
            'success': True,
            'record': record
        })
        
    except Exception as e:
        app.logger.error(f"Error obteniendo detalles: {str(e)}")
        return jsonify({
            'error': 'Error al obtener detalles',
            'details': str(e)
        }), 500
    finally:
        if 'conn' in locals():
            conn.close()
            
# Variables globales para monitoreo
SERVICE_START_TIME = time.time()
REQUEST_COUNTER = 0
PERFORMANCE_METRICS = {
    'cpu': [],
    'memory': [],
    'response_times': []
}

# Cargar modelo
MODEL_PATH = 'models/endometriosis_model_optimized.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Ejecuta primero train_model.py")

model = joblib.load(MODEL_PATH)
FEATURES = model.feature_names_in_

class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.true_labels = []
        self.calibration_fig = None
        
    def update(self, y_true, y_pred):
        self.predictions.extend(y_pred)
        if y_true is not None:
            self.true_labels.extend(y_true)
        
    def get_calibration_plot(self):
        if len(self.true_labels) == 0:
            return None
            
        prob_true, prob_pred = calibration_curve(
            self.true_labels, self.predictions, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', label='Modelo')
        plt.plot([0, 1], [0, 1], '--', label='Perfecto')
        plt.xlabel('Probabilidad Predicha')
        plt.ylabel('Probabilidad Real')
        plt.title('Curva de Calibración')
        plt.legend()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')

monitor = ModelMonitor()

def generate_performance_metrics():
    """Simula métricas de rendimiento en segundo plano"""
    while True:
        PERFORMANCE_METRICS['cpu'].append(
            np.clip(20 + 10 * np.random.randn(), 5, 95))
        PERFORMANCE_METRICS['memory'].append(
            np.clip(300 + 200 * np.random.rand(), 100, 800))
        PERFORMANCE_METRICS['response_times'].append(
            np.clip(50 + 100 * np.random.rand(), 10, 500))
        
        # Mantener solo últimos 60 valores
        for k in PERFORMANCE_METRICS:
            if len(PERFORMANCE_METRICS[k]) > 60:
                PERFORMANCE_METRICS[k].pop(0)
        
        time.sleep(5)

# Iniciar hilo de métricas
metrics_thread = threading.Thread(target=generate_performance_metrics, daemon=True)
metrics_thread.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    start_time = time.time()
    
    try:
        # 1. Validar entrada
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Datos no proporcionados'}), 400
        
        # 2. Preparar datos para el modelo
        input_df = prepare_input_data(data)
        
        # 3. Hacer predicción
        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba > 0.5)
        
        # 4. Generar explicación
        explanation = generate_explanation(input_df.iloc[0], proba)
        
        # 5. Registrar para monitoreo (sin true_label en producción)
        monitor.update(None, [proba])
        
        # 6. Métricas de rendimiento
        response_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'prediction': prediction,
            'probability': float(proba),
            'risk_level': explanation['risk_level'],
            'key_factors': explanation['key_factors'],
            'recommendations': explanation['recommendations'],
            'model_info': {
                'version': 'v4.1',
                'features_used': FEATURES.tolist(),
                'response_time_ms': round(response_time, 2)
            }
        })
    
    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            'error': 'Error procesando la solicitud',
            'details': str(e)
        }), 500

def prepare_input_data(raw_data):
    """Prepara y valida los datos de entrada con todas las variables"""
    default_values = {
        'age': 25,
        'bmi': 23.5,
        'menarche_age': 12,
        'cycle_length': 28,
        'period_duration': 5,
        'pain_level': 5,
        'ca125': 20.0,
        'crp': 3.0,
        'dysmenorrhea': 0,
        'dyspareunia': 0,
        'chronic_pelvic_pain': 0,
        'infertility': 0,
        'family_history': 0,
        'pain_during_sex': 0,
        'bowel_symptoms': 0,
        'urinary_symptoms': 0,
        'fatigue': 0,
        'gynecological_surgery': 0,
        'pelvic_inflammatory': 0,
        'ovarian_cysts': 0,
        'family_endometriosis': 0,
        'family_autoimmune': 0,
        'family_cancer': 0,
        'comorbidity_autoimmune': 0,
        'comorbidity_thyroid': 0,
        'comorbidity_ibs': 0,
        'pain_premenstrual': 0,
        'pain_menstrual': 0,
        'pain_ovulation': 0,
        'pain_chronic': 0
    }
    
    # Aplicar valores por defecto y validar tipos
    processed_data = {}
    for feature in FEATURES:
        value = raw_data.get(feature, default_values[feature])
        
        # Conversión de tipos
        if feature in ['age', 'menarche_age', 'cycle_length', 
                      'period_duration', 'pain_level']:
            processed_data[feature] = int(value)
        else:
            processed_data[feature] = float(value)
    
    return pd.DataFrame([processed_data])

def generate_explanation(input_data, probability):
    """Genera una explicación comprensible de la predicción"""
    factors = []
    medical_terms = {
        'dysmenorrhea': 'Dolor menstrual severo',
        'dyspareunia': 'Dolor durante relaciones',
        'chronic_pelvic_pain': 'Dolor pélvico crónico',
        'family_history': 'Historia familiar de endometriosis'
    }
    
    # Identificar factores clave
    if input_data['pain_level'] >= 7:
        factors.append(f"Dolor intenso ({input_data['pain_level']}/10)")
    if input_data['ca125'] > 35:
        factors.append(f"CA-125 elevado ({input_data['ca125']} U/mL)")
    if input_data['crp'] > 5:
        factors.append(f"Inflamación elevada (CRP: {input_data['crp']} mg/L)")
    
    for feature, term in medical_terms.items():
        if input_data[feature] == 1:
            factors.append(term)
    
    # Determinar nivel de riesgo
    if probability >= 0.7:
        risk_level = "ALTO"
        recommendations = [
            "Consulta inmediata con ginecólogo especializado en endometriosis",
            "Ecografía transvaginal con protocolo ESUR para caracterización de implantes",
            "Perfil de marcadores inflamatorios (CA-125, IL-6, PCR ultrasensible)",
            "Resonancia magnética pélvica con protocolo de endometriosis",
            "Evaluación multidisciplinaria (ginecología, gastroenterología, urología según sintomatología)"
        ]
    elif probability >= 0.4:
        risk_level = "MODERADO" 
        recommendations = [
            "Evaluación ginecológica con escala visual analógica (EVA) para cuantificación del dolor",
            "Terapia analgésica escalonada (AINES como primera línea, considerando opioides débiles si EVA >7)",
            "Estudio hormonal básico (FSH, LH, estradiol) y marcador CA-125",
            "Ecografía pélvica transvaginal de alta resolución",
            "Seguimiento clínico a 8-12 semanas con reevaluación de score de riesgo"
        ]
    else:
        risk_level = "BAJO"
        recommendations = [
            "Registro estructurado de síntomas mediante diario menstrual (frecuencia, intensidad, dismenorrea)",
            "Analgesia con AINES selectivos (ej. celecoxib 200mg/12h) según patrón dolor",
            "Educación sobre signos de alarma (disquecia, dispareunia profunda, hematuria cíclica)",
            "Suplementación con ácidos grasos omega-3 (1000mg/día) como modulador inflamatorio",
            "Control anual con evaluación de progresión sintomática"
        ]
    
    return {
        'risk_level': risk_level,
        'key_factors': factors,
        'recommendations': recommendations
    }

@app.route('/generate_clinical_record', methods=['POST'])
def generate_clinical_record():
    try:
        data = request.get_json()
        
        # Configuración del documento
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              leftMargin=0.5*inch,
                              rightMargin=0.5*inch,
                              topMargin=0.5*inch,
                              bottomMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        
        # Estilos personalizados
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Heading1'],
            fontSize=12,
            alignment=1,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            spaceAfter=4
        )
        
        bold_style = ParagraphStyle(
            'Bold',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            spaceAfter=4,
            fontName='Helvetica-Bold'
        )
        
        # Contenido del PDF
        elements = []
        
        # Logo
        logo_path = "static/assets/logo.png"
        try:
            logo = Image(logo_path, width=1.5*inch, height=0.4*inch)
            logo.hAlign = 'LEFT'
            elements.append(logo)
            elements.append(Spacer(1, 12))
        except Exception as e:
            app.logger.error(f"No se pudo cargar el logo: {str(e)}")
        
        # 1. Encabezado
        elements.append(Paragraph("BONO DE ATENCIÓN MÉDICA", header_style))
        elements.append(Paragraph("INFORMACIÓN DE LA PACIENTE:", subtitle_style))
        elements.append(Spacer(1, 12))
        
        # 2. Información del paciente con negritas en indicadores
        beneficiary_data = [
            [Paragraph("<b>N° Bono:</b>", bold_style), Paragraph(f"END-{datetime.now().strftime('%Y%m%d%H%M')}", normal_style)],
            [Paragraph("<b>RUT Beneficiario:</b>", bold_style), Paragraph(data['personal']['id_number'], normal_style)],
            [Paragraph("<b>Nombre:</b>", bold_style), Paragraph(data['personal']['full_name'], normal_style)],
            [Paragraph("<b>Edad:</b>", bold_style), Paragraph(f"{data['personal']['age']} años", normal_style)],
            [Paragraph("<b>Previsión:</b>", bold_style), Paragraph(data['personal']['insurance'], normal_style)]
        ]
        
        beneficiary_table = Table(beneficiary_data, colWidths=[120, 300])
        beneficiary_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(beneficiary_table)
        elements.append(Spacer(1, 12))
        
        # 3. Tabla de servicios con encabezados en negrita
        service_header_style = ParagraphStyle(
            'ServiceHeader',
            parent=styles['Normal'],
            fontSize=8,
            fontName='Helvetica-Bold'
        )
        
        service_data = [
            [Paragraph("<b>Código</b>", service_header_style),
            Paragraph("<b>Descripción</b>", service_header_style),
            Paragraph("<b>Fecha</b>", service_header_style),
            Paragraph("<b>Valor</b>", service_header_style),
            Paragraph("<b>Copago</b>", service_header_style),
            Paragraph("<b>A Pagar</b>", service_header_style)],
            [Paragraph("END-001", normal_style),
            Paragraph("Evaluación Endometriosis", normal_style),
            Paragraph(datetime.now().strftime('%d/%m/%Y'), normal_style),
            Paragraph("$15.780", normal_style),
            Paragraph("$7.560", normal_style),
            Paragraph("$8.220", normal_style)]
        ]
        
        service_table = Table(service_data, colWidths=[60, 130, 60, 60, 60, 60])
        service_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(service_table)
        elements.append(Spacer(1, 12))
        
        # 4. Totales
        total_data = [
            ["TOTAL A PAGAR:", "$8.220"],
            ["IVA (19%):", "$1.564"], 
            ["TOTAL CON IVA:", "$9.784"]
        ]

        total_table = Table(total_data, colWidths=[300, 60])  # Ajusté el ancho de la primera columna
        total_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),  # Alinea las etiquetas a la derecha
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),  # Alinea los valores a la derecha
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Centra verticalmente
            ('LEFTPADDING', (0, 0), (0, -1), 10),  # Espacio izquierdo para etiquetas
            ('RIGHTPADDING', (0, 0), (0, -1), 5),  # Espacio derecho para etiquetas
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),  # Espacio inferior
            ('TOPPADDING', (0, 0), (-1, -1), 3),  # Espacio superior
        ]))

        elements.append(total_table)
        elements.append(Spacer(1, 12))
        
        # 5. Información profesional con indicadores en negrita y respuestas en normal_style
        professional_data = [
            [Paragraph("<b>Médico tratante:</b>", bold_style), Paragraph("Dr. John Doe", normal_style)],
            [Paragraph("<b>RUT:</b>", bold_style), Paragraph("12.345.678-9", normal_style)],
            [Paragraph("<b>Fecha atención:</b>", bold_style), Paragraph(datetime.now().strftime('%d/%m/%Y'), normal_style)]
        ]

        professional_table = Table(professional_data, colWidths=[120, 300])
        professional_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(professional_table)
        elements.append(Spacer(1, 24))
        
        # 6. Resultados de la evaluación de riesgo
        elements.append(Paragraph("RESULTADOS DE LA EVALUACIÓN:", subtitle_style))
        elements.append(Spacer(1, 6))

        # Hacer predicción para incluir en el PDF
        input_data = {
            'age': int(data['personal']['age']),
            'menarche_age': int(data['menstrual']['menarche_age']),
            'cycle_length': int(data['menstrual']['cycle_length']),
            'period_duration': int(data['menstrual']['period_duration']),
            'pain_level': int(data['menstrual']['pain_level']),
            'pain_during_sex': 1 if data['symptoms']['pain_during_sex'] else 0,
            'family_history': 1 if data['history']['family_endometriosis'] else 0,
            'bowel_symptoms': 1 if data['symptoms']['bowel_symptoms'] else 0,
            'urinary_symptoms': 1 if data['symptoms']['urinary_symptoms'] else 0,
            'fatigue': 1 if data['symptoms']['fatigue'] else 0,
            'infertility': 1 if data['symptoms']['infertility'] else 0,
            'ca125': float(data['biomarkers']['ca125']) if data['biomarkers']['ca125'] is not None else 20.0,
            'crp': float(data['biomarkers']['crp']) if data['biomarkers']['crp'] is not None else 3.0
        }

        input_df = prepare_input_data(input_data)
        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba > 0.5)
        explanation = generate_explanation(input_df.iloc[0], proba)

        # Convertir probabilidad a porcentaje
        probability_percent = round(proba * 100, 1)

        # Determinar nivel de riesgo y color
        if proba >= 0.7:
            risk_level = "ALTO"
        elif proba >= 0.4:
            risk_level = "MODERADO"
        else:
            risk_level = "BAJO"

        # Crear tabla para los resultados con el mismo formato que información del paciente
        results_data = [
            [Paragraph("<b>Probabilidad:</b>", bold_style), 
            Paragraph(f"{probability_percent}%", normal_style)],
            
            [Paragraph("<b>Nivel de Riesgo:</b>", bold_style), 
            Paragraph(risk_level, normal_style)],
            
            [Paragraph("<b>Factores Clave:</b>", bold_style), 
            Paragraph(', '.join(explanation['key_factors']) or 'No identificados', normal_style)]
        ]

        results_table = Table(results_data, colWidths=[120, 300])
        results_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(results_table)
        elements.append(Spacer(1, 6))

        # Estilo para las recomendaciones (añadir junto a los otros estilos)
        recommendation_style = ParagraphStyle(
            'Recommendation',
            parent=styles['Normal'],
            fontSize=9,
            leading=10,
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=10,
            textColor=colors.black,
            bulletIndent=5
        )

        # Sección de recomendaciones mejorada
        recommendations_title = Paragraph("<b>RECOMENDACIONES:</b>", subtitle_style)
        elements.append(recommendations_title)
        elements.append(Spacer(1, 4))

        # Lista de recomendaciones con formato mejorado
        recommendation_items = []
        for recommendation in explanation['recommendations']:
            recommendation_items.append(Paragraph(f"• {recommendation}", recommendation_style)
            )
            recommendation_items.append(Spacer(1, 2))  # Espacio entre items

        elements.extend(recommendation_items)
        elements.append(Spacer(1, 12))  # Espacio final después de la sección
        
        # 7. Firmas
        signature_data = [
            ["", ""],
            ["__________________________", "__________________________"],
            ["Firma Beneficiario", "Firma Profesional/Institución"]
        ]
        
        signature_table = Table(signature_data, colWidths=[210, 210])
        signature_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(signature_table)
        
        # Generar PDF
        doc.build(elements)
        
        # Preparar respuesta
        buffer.seek(0)
        filename = f"{data['personal']['full_name'].replace(' ', '_')}_END_{datetime.now().strftime('%Y%m%d')}.pdf"
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'

        return response
        
    except Exception as e:
        app.logger.error(f"Error generando bono: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Verificar que exista el modelo
    if not os.path.exists(MODEL_PATH):
        print("Error: Modelo no encontrado. Ejecuta train_model.py primero.")
        exit(1)
    
    # Crear directorios necesarios
    os.makedirs('static/plots', exist_ok=True)
    
    # Iniciar servidor
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), 
           threaded=True)