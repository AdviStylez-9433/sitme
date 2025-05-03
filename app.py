from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
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
import re
import logging

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
        # Validar que se recibió JSON
        if not request.is_json:
            return jsonify({'error': 'Se requiere Content-Type: application/json'}), 400

        data = request.get_json()
        
        # Validar estructura básica
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Datos no válidos'}), 400

        # Validar campos obligatorios
        required_structure = {
            'personal': {
                'full_name': str,
                'id_number': str,
                'birth_date': str,
                'age': int,
                'insurance': str
            },
            'riskTitle': str,
            'probability': (float, int),
            'riskDescription': str,
            'riskFactors': list,
            'recommendations': list
        }

        for field, field_type in required_structure.items():
            if field not in data:
                return jsonify({'error': f'Campo obligatorio faltante: {field}'}), 400
            
            # Validar tipos para campos específicos
            if isinstance(field_type, tuple):
                if not isinstance(data[field], field_type):
                    return jsonify({'error': f'Campo {field} debe ser numérico'}), 400
            elif not isinstance(data[field], field_type):
                return jsonify({'error': f'Campo {field} debe ser {field_type.__name__}'}), 400

        # Validar valores específicos
        if not 0 <= float(data['probability']) <= 1:
            return jsonify({'error': 'Probabilidad debe estar entre 0 y 1'}), 400

        # Validar y limpiar nombre para el archivo
        clean_name = re.sub(r'[^\w\-_]', '_', data["personal"]["full_name"])[:50]  # Limitar longitud

        # --- CONSTRUCCIÓN DEL PDF ---
        buffer = io.BytesIO()
        
        # Configurar documento
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,  # Más margen para mejor legibilidad
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Estilos personalizados
        styles.add(ParagraphStyle(
            name='Header',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,
            spaceAfter=12,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='Subtitle',
            parent=styles['Normal'],
            fontSize=11,
            alignment=1,
            spaceAfter=24,
            textColor=colors.HexColor('#666666'),
            fontName='Helvetica'
        ))
        
        styles.add(ParagraphStyle(
            name='Section',
            parent=styles['Heading2'],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor('#5e35b1'),
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1,
            textColor=colors.HexColor('#999999'),
            fontName='Helvetica-Oblique'
        ))
        
        # Contenido del PDF
        elements = []
        
        # Encabezado
        elements.append(Paragraph("COMPROBANTE DE ATENCIÓN MÉDICA", styles['Header']))
        elements.append(Paragraph("SISTEMA INTEGRAL DE TAMIZAJE PARA ENDOMETRIOSIS", styles['Subtitle']))
        
        # Línea divisoria
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Table(
            [[""]], 
            colWidths=[7*inch], 
            style=[
                ('LINEABOVE', (0,0), (-1,-1), 1.5, colors.HexColor('#5e35b1')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8)
            ]
        ))
        
        # Datos del paciente
        patient_data = [
            ["<b>Nombre</b>", data['personal']['full_name']],
            ["<b>RUT</b>", data['personal']['id_number']],
            ["<b>Fecha Nac.</b>", data['personal']['birth_date']],
            ["<b>Edad</b>", f"{data['personal']['age']} años"],
            ["<b>Previsión</b>", data['personal']['insurance']],
            ["<b>Médico Tratante</b>", "ESPECIALISTA EN GINECOLOGÍA"]
        ]
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("DATOS DEL PACIENTE", styles['Section']))
        elements.append(Table(
            patient_data, 
            colWidths=[2*inch, 5*inch], 
            style=[
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 11),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (0,-1), 'LEFT'),
                ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#666666')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEADING', (0,0), (-1,-1), 14)
            ]
        ))
        
        # Resultados de evaluación
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("RESULTADOS DE EVALUACIÓN", styles['Section']))
        
        risk_data = [
            ["<b>Nivel de Riesgo</b>", data['riskTitle']],
            ["<b>Probabilidad</b>", f"{int(float(data['probability']))*100}%"],
            ["<b>Descripción</b>", data['riskDescription']]
        ]
        
        elements.append(Table(
            risk_data, 
            colWidths=[2*inch, 5*inch], 
            style=[
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 11),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (0,-1), 'LEFT'),
                ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#666666')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#FAFAFA'), colors.white]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#EEEEEE'))
            ]
        ))
        
        # Factores de riesgo
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("FACTORES DE RIESGO IDENTIFICADOS", styles['Section']))
        
        if data['riskFactors']:
            risk_factors = [[f"• {factor}"] for factor in data['riskFactors']]
            elements.append(Table(
                risk_factors, 
                colWidths=[7*inch], 
                style=[
                    ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('VALIGN', (0,0), (-1,-1), 'TOP')
                ]
            ))
        else:
            elements.append(Paragraph("No se identificaron factores de riesgo relevantes.", styles['Normal']))
        
        # Recomendaciones
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("RECOMENDACIONES CLÍNICAS", styles['Section']))
        
        if data['recommendations']:
            recommendations = [[f"• {rec}"] for rec in data['recommendations']]
            elements.append(Table(
                recommendations, 
                colWidths=[7*inch], 
                style=[
                    ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                    ('FONTSIZE', (0,0), (-1,-1), 10),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('VALIGN', (0,0), (-1,-1), 'TOP')
                ]
            ))
        else:
            elements.append(Paragraph("No se generaron recomendaciones específicas.", styles['Normal']))
        
        # Pie de página
        elements.append(Spacer(1, 0.4*inch))
        elements.append(Table(
            [[""]], 
            colWidths=[7*inch], 
            style=[
                ('LINEABOVE', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4)
            ]
        ))
        
        footer_text = f"Documento generado automáticamente el {datetime.now().strftime('%d-%m-%Y %H:%M')} - Sistema Integral de Tamizaje para Endometriosis"
        elements.append(Paragraph(footer_text, styles['Footer']))
        
        # Generar PDF
        doc.build(elements)
        
        # Preparar respuesta
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=ficha_clinica_{clean_name}.pdf'
        
        return response
        
    except Exception as e:
        logging.exception("Error al generar ficha clínica")
        return jsonify({
            'error': 'Error interno al generar el documento',
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