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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame, PageTemplate, PageBreak
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
            alignment=1,  # Centrado
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
        
        # Contenido del PDF
        elements = []
        
        # 1. Encabezado del bono
        elements.append(Paragraph("BONO DE ATENCIÓN MÉDICA", header_style))
        elements.append(Paragraph("Evaluación de Endometriosis", subtitle_style))
        elements.append(Spacer(1, 12))
        
        # 2. Información del beneficiario (en tabla)
        beneficiary_data = [
            ["N° Bono:", f"END-{datetime.now().strftime('%Y%m%d%H%M')}"],
            ["Fecha Emisión:", datetime.now().strftime('%d/%m/%Y')],
            ["RUT Beneficiario:", data['personal']['id_number']],
            ["Nombre:", data['personal']['full_name']],
            ["Edad:", f"{data['personal']['age']} años"],
            ["Previsión:", data['personal']['insurance']],
            ["Nivel:", "3"]  # Nivel de atención
        ]
        
        beneficiary_table = Table(beneficiary_data, colWidths=[120, 300])
        beneficiary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (0, -1), 0),
            ('RIGHTPADDING', (0, 0), (0, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(beneficiary_table)
        elements.append(Spacer(1, 12))
        
        # 3. Detalle de la prestación (tabla con formato similar al ejemplo)
        service_data = [
            # Encabezados
            ["Código", "Descripción", "Fecha", "Valor", "Bonificación", "A Pagar"],
            # Datos
            ["END-001", "Evaluación Endometriosis", datetime.now().strftime('%d/%m/%Y'), "$15.780", "$7.560", "$8.220"]
        ]
        
        service_table = Table(service_data, colWidths=[60, 120, 60, 60, 60, 60])
        service_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ]))
        elements.append(service_table)
        elements.append(Spacer(1, 12))
        
        # 4. Totales
        total_data = [
            ["TOTAL A PAGAR:", "$7.000"]
        ]
        
        total_table = Table(total_data, colWidths=[360, 60])
        total_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))
        elements.append(total_table)
        elements.append(Spacer(1, 12))
        
        # 5. Información del profesional
        professional_data = [
            ["Profesional/Institución:", "Centro Médico SITME"],
            ["RUT:", "76.549.770-1"],
            ["Médico tratante:", "Dr. John Doe"],
            ["Fecha atención:", datetime.now().strftime('%d/%m/%Y')]
        ]
        
        professional_table = Table(professional_data, colWidths=[120, 300])
        professional_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(professional_table)
        elements.append(Spacer(1, 24))
        
        # 6. Firmas
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
        
                # --- SEGUNDA PÁGINA (RESULTADOS) ---
        elements.append(PageBreak())
        elements.append(Paragraph("RESULTADOS DE EVALUACIÓN", header_style))
        elements.append(Spacer(1, 12))
        
        # 1. Resumen de riesgo
        risk_percentage = int(float(data['probability']) * 100)
        elements.append(Paragraph(f"Riesgo estimado: {risk_percentage}%", subtitle_style))
        elements.append(Paragraph(f"Nivel de riesgo: {data['risk_level'].capitalize()}", normal_style))
        elements.append(Spacer(1, 8))
        
        # 2. Factores clave
        elements.append(Paragraph("FACTORES CLAVE IDENTIFICADOS", subtitle_style))
        for factor in data['key_factors']:
            elements.append(Paragraph(f"• {factor}", normal_style))
        elements.append(Spacer(1, 8))
        
        # 3. Recomendaciones
        elements.append(Paragraph("RECOMENDACIONES", subtitle_style))
        for recommendation in data['recommendations']:
            elements.append(Paragraph(f"• {recommendation}", normal_style))
        
        # Construir el PDF
        doc.build(elements)
        
        # Preparar la respuesta
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=bono_endometriosis_{data["personal"]["full_name"].replace(" ", "_")}.pdf'
        
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