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
        logo_path = "static/logo.png"
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