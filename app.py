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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Configuración inicial
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

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
    """Prepara y valida los datos de entrada"""
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
        'family_history': 0
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
        risk_level = "alto"
        recommendations = [
            "Consulta urgente con especialista",
            "Ecografía transvaginal",
            "Análisis de marcadores inflamatorios"
        ]
    elif probability >= 0.4:
        risk_level = "moderado"
        recommendations = [
            "Evaluación ginecológica",
            "Control del dolor",
            "Seguimiento en 3 meses"
        ]
    else:
        risk_level = "bajo"
        recommendations = [
            "Monitoreo de síntomas",
            "Analgesia según necesidad",
            "Educación sobre endometriosis"
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
        
        # Configuración con dos columnas
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
            parent=styles['Heading2'],
            fontSize=10,
            leading=12,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        item_style = ParagraphStyle(
            'Item',
            parent=styles['Normal'],
            fontSize=8,
            leading=9,
            spaceAfter=2,
            leftIndent=0
        )
        
        # Función para crear secciones
        def create_section(title, items):
            content = []
            content.append(Paragraph(title, header_style))
            for item in items:
                content.append(Paragraph(item, item_style))
            content.append(Spacer(1, 12))
            return content
        
        # Preparar contenido para dos columnas
        col1 = []
        col2 = []
        
        # Columna 1
        # 1. Datos personales
        personal_items = [
            f"<b>Nombre:</b> {data['personal']['full_name']}",
            f"<b>RUT:</b> {data['personal']['id_number']}",
            f"<b>Edad:</b> {data['personal']['age']} años",
            f"<b>Previsión:</b> {data['personal']['insurance']}",
            f"<b>Tipo sangre:</b> {data['personal']['blood_type']}"
        ]
        col1.extend(create_section("DATOS PERSONALES", personal_items))
        
        # 2. Antecedentes médicos
        history_items = [
            f"<b>Cirugías ginecológicas:</b> {'Sí' if data['history']['gynecological_surgery'] else 'No'}",
            f"<b>Enf. inflamatoria pélvica:</b> {'Sí' if data['history']['pelvic_inflammatory'] else 'No'}",
            f"<b>Quistes ováricos:</b> {'Sí' if data['history']['ovarian_cysts'] else 'No'}",
            f"<b>Familiar endometriosis:</b> {'Sí' if data['history']['family_endometriosis'] else 'No'}",
            f"<b>Familiar autoinmunes:</b> {'Sí' if data['history']['family_autoimmune'] else 'No'}",
            f"<b>Comorbilidades:</b> {'Sí' if data['history']['comorbidity_autoimmune'] or data['history']['comorbidity_thyroid'] else 'No'}",
            f"<b>Medicamentos:</b> {data['history']['medications'] or 'Ninguno'}"
        ]
        col1.extend(create_section("ANTECEDENTES MÉDICOS", history_items))
        
        # 3. Datos menstruales
        menstrual_items = [
            f"<b>Menarquia:</b> {data['menstrual']['menarche_age']} años",
            f"<b>Ciclo:</b> {data['menstrual']['cycle_length']} días",
            f"<b>Duración:</b> {data['menstrual']['period_duration']} días",
            f"<b>Última regla:</b> {data['menstrual']['last_period']}",
            f"<b>Dolor:</b> {data['menstrual']['pain_level']}/10",
            f"<b>Dolor crónico:</b> {'Sí' if data['menstrual']['pain_chronic'] else 'No'}"
        ]
        col1.extend(create_section("DATOS MENSTRUALES", menstrual_items))
        
        # Columna 2
        # 1. Síntomas
        symptoms_items = [
            f"<b>Dolor durante relaciones:</b> {'Sí' if data['symptoms']['pain_during_sex'] else 'No'}",
            f"<b>Síntomas intestinales:</b> {'Sí' if data['symptoms']['bowel_symptoms'] else 'No'}",
            f"<b>Síntomas urinarios:</b> {'Sí' if data['symptoms']['urinary_symptoms'] else 'No'}",
            f"<b>Fatiga:</b> {'Sí' if data['symptoms']['fatigue'] else 'No'}",
            f"<b>Infertilidad:</b> {'Sí' if data['symptoms']['infertility'] else 'No'}",
            f"<b>Otros síntomas:</b> {data['symptoms']['other_symptoms'] or 'Ninguno'}"
        ]
        col2.extend(create_section("SÍNTOMAS", symptoms_items))
        
        # 2. Biomarcadores
        biomarkers_items = [
            f"<b>CA-125:</b> {data['biomarkers']['ca125'] or 'No medido'}",
            f"<b>IL-6:</b> {data['biomarkers']['il6'] or 'No medido'}",
            f"<b>TNF-α:</b> {data['biomarkers']['tnf_alpha'] or 'No medido'}",
            f"<b>VEGF:</b> {data['biomarkers']['vegf'] or 'No medido'}",
            f"<b>AMH:</b> {data['biomarkers']['amh'] or 'No medido'}",
            f"<b>PCR:</b> {data['biomarkers']['crp'] or 'No medido'}"
        ]
        col2.extend(create_section("BIOMARCADORES", biomarkers_items))
        
        # 3. Examen físico
        exam_items = [
            f"<b>IMC:</b> {data['examination']['bmi'] or 'No calculado'}",
            f"<b>Ex. pélvico:</b> {data['examination']['pelvic_exam'] or 'No realizado'}",
            f"<b>Ex. vaginal:</b> {data['examination']['vaginal_exam'] or 'No realizado'}",
            f"<b>Notas clínicas:</b> {data['examination']['clinical_notes'] or 'Ninguna'}"
        ]
        col2.extend(create_section("EXAMEN FÍSICO", exam_items))
        
        # Bono de atención (ancho completo)
        bono_items = [
            "BONO DE ATENCIÓN AMBULATORIA",
            f"<b>Paciente:</b> {data['personal']['full_name']}",
            f"<b>RUT:</b> {data['personal']['id_number']}",
            f"<b>Previsión:</b> {data['personal']['insurance']}",
            f"<b>Fecha emisión:</b> {datetime.now().strftime('%d/%m/%Y')}",
            f"<b>Válido hasta:</b> {(datetime.now() + timedelta(days=30)).strftime('%d/%m/%Y')}",
            "Autoriza atención especializada en endometriosis",
            "Firma profesional: ________________________"
        ]
        
        # Organizar en dos columnas
        frame1 = Frame(doc.leftMargin, doc.bottomMargin + doc.height/2, 
                      doc.width/2 - 6, doc.height/2 - 20, id='col1')
        frame2 = Frame(doc.leftMargin + doc.width/2, doc.bottomMargin + doc.height/2, 
                      doc.width/2 - 6, doc.height/2 - 20, id='col2')
        frame_bono = Frame(doc.leftMargin, doc.bottomMargin, 
                         doc.width, doc.height/2 - 20, id='bono')
        
        # Construir el documento
        doc.addPageTemplates([
            PageTemplate(id='TwoColumns', frames=[frame1, frame2, frame_bono])
        ])
        
        story = []
        story.extend(col1)
        story.extend(col2)
        story.extend(create_section("", bono_items))
        
        doc.build(story)
        
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=ficha_{data["personal"]["full_name"].replace(" ", "_")}.pdf'
        
        return response
        
    except Exception as e:
        app.logger.error(f"Error generando ficha: {str(e)}")
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