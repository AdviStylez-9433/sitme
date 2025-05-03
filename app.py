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
        
        # Crear el PDF con márgenes más pequeños
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              leftMargin=0.5*inch,
                              rightMargin=0.5*inch,
                              topMargin=0.5*inch,
                              bottomMargin=0.5*inch)
        
        # Estilos
        styles = getSampleStyleSheet()
        
        # Estilo para el título (más pequeño)
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=12,
            alignment=1,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        # Estilo para subtítulos
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Heading2'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            leftIndent=0
        )
        
        # Estilo para texto normal (más pequeño)
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=8,
            spaceAfter=6,
            leading=10,
            alignment=0,  # 0=left, 1=center, 2=right
            leftIndent=0
        )
        
        # Contenido del PDF
        elements = []
        
        # 1. Encabezado compacto
        elements.append(Paragraph("FICHA CLÍNICA - ENDOMETRIOSIS", title_style))
        elements.append(Spacer(1, 6))
        
        # 2. Datos personales en 2 columnas
        personal_data = [
            ["<b>Nombre:</b>", data['personal']['full_name'],
             "<b>Fecha Nac.:</b>", data['personal']['birth_date']],
            ["<b>RUT:</b>", data['personal']['id_number'],
             "<b>Edad:</b>", f"{data['personal']['age']} años"],
            ["<b>Tipo Sangre:</b>", data['personal']['blood_type'],
             "<b>Previsión:</b>", data['personal']['insurance']]
        ]
        
        personal_table = Table(personal_data, colWidths=[60, 120, 60, 120])
        personal_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(personal_table)
        elements.append(Spacer(1, 6))
        
        # 3. Antecedentes médicos compactos
        elements.append(Paragraph("ANTECEDENTES MÉDICOS", subtitle_style))
        
        history_data = [
            ["<b>Cirugías ginecológicas:</b>", "Sí" if data['history']['gynecological_surgery'] else "No",
             "<b>Enf. inflamatoria pélvica:</b>", "Sí" if data['history']['pelvic_inflammatory'] else "No"],
            ["<b>Quistes ováricos:</b>", "Sí" if data['history']['ovarian_cysts'] else "No",
             "<b>Familiar endometriosis:</b>", "Sí" if data['history']['family_endometriosis'] else "No"],
            ["<b>Familiar autoinmunes:</b>", "Sí" if data['history']['family_autoimmune'] else "No",
             "<b>Familiar cáncer:</b>", "Sí" if data['history']['family_cancer'] else "No"],
            ["<b>Comorbilidades autoinmunes:</b>", "Sí" if data['history']['comorbidity_autoimmune'] else "No",
             "<b>Comorbilidades tiroides:</b>", "Sí" if data['history']['comorbidity_thyroid'] else "No"],
            ["<b>SII:</b>", "Sí" if data['history']['comorbidity_ibs'] else "No",
             "<b>Medicamentos:</b>", data['history']['medications'] or "Ninguno"]
        ]
        
        history_table = Table(history_data, colWidths=[90, 50, 90, 50])
        history_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        elements.append(history_table)
        elements.append(Spacer(1, 6))
        
        # 4. Datos menstruales compactos
        elements.append(Paragraph("DATOS MENSTRUALES", subtitle_style))
        
        menstrual_data = [
            ["<b>Menarquia:</b>", f"{data['menstrual']['menarche_age']} años",
             "<b>Ciclo:</b>", f"{data['menstrual']['cycle_length']} días",
             "<b>Duración:</b>", f"{data['menstrual']['period_duration']} días"],
            ["<b>Última regla:</b>", data['menstrual']['last_period'],
             "<b>Dolor:</b>", f"{data['menstrual']['pain_level']}/10",
             "<b>Dolor crónico:</b>", "Sí" if data['menstrual']['pain_chronic'] else "No"],
            ["<b>Dolor premenstrual:</b>", "Sí" if data['menstrual']['pain_premenstrual'] else "No",
             "<b>Dolor menstrual:</b>", "Sí" if data['menstrual']['pain_menstrual'] else "No",
             "<b>Dolor ovulación:</b>", "Sí" if data['menstrual']['pain_ovulation'] else "No"]
        ]
        
        menstrual_table = Table(menstrual_data, colWidths=[60, 40, 50, 40, 50, 40])
        menstrual_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        elements.append(menstrual_table)
        elements.append(Spacer(1, 6))
        
        # 5. Síntomas compactos
        elements.append(Paragraph("SÍNTOMAS", subtitle_style))
        
        symptoms_data = [
            ["<b>Dolor relaciones:</b>", "Sí" if data['symptoms']['pain_during_sex'] else "No",
             "<b>Síntomas intestinales:</b>", "Sí" if data['symptoms']['bowel_symptoms'] else "No"],
            ["<b>Síntomas urinarios:</b>", "Sí" if data['symptoms']['urinary_symptoms'] else "No",
             "<b>Fatiga:</b>", "Sí" if data['symptoms']['fatigue'] else "No"],
            ["<b>Infertilidad:</b>", "Sí" if data['symptoms']['infertility'] else "No",
             "<b>Otros síntomas:</b>", data['symptoms']['other_symptoms'] or "Ninguno"]
        ]
        
        symptoms_table = Table(symptoms_data, colWidths=[80, 40, 80, 40])
        symptoms_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        elements.append(symptoms_table)
        elements.append(Spacer(1, 6))
        
        # 6. Biomarcadores compactos
        elements.append(Paragraph("BIOMARCADORES", subtitle_style))
        
        biomarkers_data = [
            ["<b>CA-125:</b>", data['biomarkers']['ca125'] or "No",
             "<b>IL-6:</b>", data['biomarkers']['il6'] or "No",
             "<b>TNF-α:</b>", data['biomarkers']['tnf_alpha'] or "No"],
            ["<b>VEGF:</b>", data['biomarkers']['vegf'] or "No",
             "<b>AMH:</b>", data['biomarkers']['amh'] or "No",
             "<b>PCR:</b>", data['biomarkers']['crp'] or "No"],
            ["<b>Imágenes:</b>", data['biomarkers']['imaging'] or "No",
             "<b>Hallazgos:</b>", data['biomarkers']['imaging_details'] or "No"]
        ]
        
        biomarkers_table = Table(biomarkers_data, colWidths=[50, 40, 50, 40, 50, 40])
        biomarkers_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        elements.append(biomarkers_table)
        elements.append(Spacer(1, 6))
        
        # 7. Examen físico compacto
        elements.append(Paragraph("EXAMEN FÍSICO", subtitle_style))
        
        exam_data = [
            ["<b>IMC:</b>", data['examination']['bmi'] or "No",
             "<b>Ex. pélvico:</b>", data['examination']['pelvic_exam'] or "No"],
            ["<b>Ex. vaginal:</b>", data['examination']['vaginal_exam'] or "No",
             "<b>Notas:</b>", data['examination']['clinical_notes'] or "No"]
        ]
        
        exam_table = Table(exam_data, colWidths=[50, 40, 60, 100])
        exam_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        elements.append(exam_table)
        elements.append(Spacer(1, 6))
        
        # 8. Bono de atención ambulatoria compacto
        elements.append(Paragraph("BONO DE ATENCIÓN AMBULATORIA", subtitle_style))
        
        bono_data = [
            ["<b>Paciente:</b>", data['personal']['full_name']],
            ["<b>RUT:</b>", data['personal']['id_number']],
            ["<b>Previsión:</b>", data['personal']['insurance']],
            ["<b>Fecha:</b>", datetime.now().strftime("%d/%m/%Y")],
            ["<b>Válido hasta:</b>", (datetime.now() + timedelta(days=30)).strftime("%d/%m/%Y")],
            ["", ""],
            ["Autoriza atención especializada en endometriosis dentro del período de validez."],
            ["", ""],
            ["<b>Firma profesional:</b>", "__________________________"]
        ]
        
        bono_table = Table(bono_data, colWidths=[60, 200])
        bono_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('SPAN', (0, 5), (1, 5)),
            ('SPAN', (0, 6), (1, 6)),
            ('SPAN', (0, 7), (1, 7)),
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