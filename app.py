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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
        
        # Crear el PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Estilo para el título
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=14,
            alignment=1,
            spaceAfter=12
        )
        
        # Contenido del PDF
        elements = []
        
        # 1. Encabezado
        elements.append(Paragraph("FICHA CLÍNICA - ENDOMETRIOSIS", title_style))
        elements.append(Spacer(1, 12))
        
        # 2. Datos personales
        personal_data = [
            ["Nombre:", data['personal']['full_name']],
            ["RUT:", data['personal']['id_number']],
            ["Fecha Nacimiento:", data['personal']['birth_date']],
            ["Edad:", f"{data['personal']['age']} años"],
            ["Tipo Sangre:", data['personal']['blood_type']],
            ["Previsión:", data['personal']['insurance']]
        ]
        
        personal_table = Table(personal_data, colWidths=[100, 300])
        personal_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ]))
        elements.append(personal_table)
        elements.append(Spacer(1, 12))
        
        # 3. Antecedentes médicos
        elements.append(Paragraph("ANTECEDENTES MÉDICOS", styles['Heading2']))
        
        history_data = [
            ["Cirugías ginecológicas:", "Sí" if data['history']['gynecological_surgery'] else "No"],
            ["Enfermedad inflamatoria pélvica:", "Sí" if data['history']['pelvic_inflammatory'] else "No"],
            ["Quistes ováricos:", "Sí" if data['history']['ovarian_cysts'] else "No"],
            ["Antecedentes familiares:", 
             f"Endometriosis: {'Sí' if data['history']['family_endometriosis'] else 'No'}, " +
             f"Autoinmunes: {'Sí' if data['history']['family_autoimmune'] else 'No'}, " +
             f"Cáncer: {'Sí' if data['history']['family_cancer'] else 'No'}"],
            ["Comorbilidades:", 
             f"Autoinmunes: {'Sí' if data['history']['comorbidity_autoimmune'] else 'No'}, " +
             f"Tiroides: {'Sí' if data['history']['comorbidity_thyroid'] else 'No'}, " +
             f"SII: {'Sí' if data['history']['comorbidity_ibs'] else 'No'}"],
            ["Medicamentos:", data['history']['medications'] or "Ninguno"]
        ]
        
        history_table = Table(history_data, colWidths=[150, 250])
        history_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(history_table)
        elements.append(Spacer(1, 12))
        
        # 4. Datos menstruales
        elements.append(Paragraph("DATOS MENSTRUALES", styles['Heading2']))
        
        menstrual_data = [
            ["Edad menarquia:", f"{data['menstrual']['menarche_age']} años"],
            ["Duración ciclo:", f"{data['menstrual']['cycle_length']} días"],
            ["Duración período:", f"{data['menstrual']['period_duration']} días"],
            ["Última menstruación:", data['menstrual']['last_period']],
            ["Nivel dolor:", f"{data['menstrual']['pain_level']}/10"],
            ["Dolor premenstrual:", "Sí" if data['menstrual']['pain_premenstrual'] else "No"],
            ["Dolor menstrual:", "Sí" if data['menstrual']['pain_menstrual'] else "No"],
            ["Dolor ovulación:", "Sí" if data['menstrual']['pain_ovulation'] else "No"],
            ["Dolor crónico:", "Sí" if data['menstrual']['pain_chronic'] else "No"]
        ]
        
        menstrual_table = Table(menstrual_data, colWidths=[150, 150, 150])
        menstrual_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(menstrual_table)
        elements.append(Spacer(1, 12))
        
        # 5. Síntomas
        elements.append(Paragraph("SÍNTOMAS", styles['Heading2']))
        
        symptoms_data = [
            ["Dolor durante relaciones:", "Sí" if data['symptoms']['pain_during_sex'] else "No"],
            ["Síntomas intestinales:", "Sí" if data['symptoms']['bowel_symptoms'] else "No"],
            ["Síntomas urinarios:", "Sí" if data['symptoms']['urinary_symptoms'] else "No"],
            ["Fatiga:", "Sí" if data['symptoms']['fatigue'] else "No"],
            ["Infertilidad:", "Sí" if data['symptoms']['infertility'] else "No"],
            ["Otros síntomas:", data['symptoms']['other_symptoms'] or "Ninguno"]
        ]
        
        symptoms_table = Table(symptoms_data, colWidths=[150, 150, 150])
        symptoms_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(symptoms_table)
        elements.append(Spacer(1, 12))
        
        # 6. Biomarcadores
        elements.append(Paragraph("BIOMARCADORES Y EXÁMENES", styles['Heading2']))
        
        biomarkers_data = [
            ["CA-125:", data['biomarkers']['ca125'] or "No medido"],
            ["IL-6:", data['biomarkers']['il6'] or "No medido"],
            ["TNF-α:", data['biomarkers']['tnf_alpha'] or "No medido"],
            ["VEGF:", data['biomarkers']['vegf'] or "No medido"],
            ["AMH:", data['biomarkers']['amh'] or "No medido"],
            ["PCR:", data['biomarkers']['crp'] or "No medido"],
            ["Imágenes:", data['biomarkers']['imaging'] or "No realizado"],
            ["Hallazgos:", data['biomarkers']['imaging_details'] or "No especificado"]
        ]
        
        biomarkers_table = Table(biomarkers_data, colWidths=[100, 100, 100, 100])
        biomarkers_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(biomarkers_table)
        elements.append(Spacer(1, 12))
        
        # 7. Examen físico
        elements.append(Paragraph("EXAMEN FÍSICO", styles['Heading2']))
        
        exam_data = [
            ["IMC:", data['examination']['bmi'] or "No calculado"],
            ["Examen pélvico:", data['examination']['pelvic_exam'] or "No realizado"],
            ["Examen vaginal:", data['examination']['vaginal_exam'] or "No realizado"],
            ["Notas clínicas:", data['examination']['clinical_notes'] or "Ninguna"]
        ]
        
        exam_table = Table(exam_data, colWidths=[100, 400])
        exam_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(exam_table)
        elements.append(Spacer(1, 24))
        
        # 8. Bono de atención ambulatoria
        elements.append(Paragraph("BONO DE ATENCIÓN AMBULATORIA", title_style))
        elements.append(Spacer(1, 12))
        
        bono_data = [
            ["Nombre del Paciente:", data['personal']['full_name']],
            ["RUT:", data['personal']['id_number']],
            ["Previsión:", data['personal']['insurance']],
            ["Fecha de Emisión:", datetime.now().strftime("%d/%m/%Y")],
            ["Válido hasta:", (datetime.now() + timedelta(days=30)).strftime("%d/%m/%Y")],
            ["", ""],
            ["Este bono autoriza al paciente a recibir atención ambulatoria especializada en endometriosis dentro del período de validez."],
            ["", ""],
            ["Firma del profesional:", "__________________________"]
        ]
        
        bono_table = Table(bono_data, colWidths=[150, 350])
        bono_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('SPAN', (0, 5), (1, 5)),
            ('SPAN', (0, 6), (1, 6)),
            ('SPAN', (0, 7), (1, 7)),
            ('SPAN', (0, 8), (1, 8)),
            ('ALIGN', (0, 6), (0, 6), 'CENTER'),
        ]))
        elements.append(bono_table)
        
        # Construir el PDF
        doc.build(elements)
        
        # Preparar la respuesta
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=ficha_clinica_{data["personal"]["full_name"].replace(" ", "_")}.pdf'
        
        return response
        
    except Exception as e:
        app.logger.error(f"Error generando ficha clínica: {str(e)}")
        return jsonify({
            'error': 'Error generando el documento',
            'details': str(e)
        }), 500

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