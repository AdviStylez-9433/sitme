import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
from streamlit.components.v1 import html
import time

# Configuración de la página
st.set_page_config(
    page_title="SITME - Sistema Integral de Tamizaje Multimodal para Endometriosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS y Animaciones ---
def load_css():
    st.markdown("""
    <style>
        :root {
            --primary: #005f73;
            --secondary: #0a9396;
            --accent: #94d2bd;
            --light: #e9d8a6;
            --warning: #ee9b00;
            --danger: #bb3e03;
            --dark: #001219;
            --white: #ffffff;
            --light-gray: #f8f9fa;
        }
        
        .stApp {
            background-color: var(--light-gray);
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
        }
        
        /* Header con gradiente profesional */
        .header-container {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 2rem;
            border-radius: 0 0 15px 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .header-container::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
            transform: rotate(30deg);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
            animation: fadeInDown 0.8s ease-out;
        }
        
        .header-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 0;
            position: relative;
            animation: fadeIn 1s ease-out 0.3s both;
        }
        
        .header-footer {
            font-size: 0.85rem;
            opacity: 0.7;
            margin-top: 1rem;
            position: relative;
            animation: fadeIn 1s ease-out 0.6s both;
        }
        
        /* Tarjetas de sección con animación */
        .section {
            background-color: var(--white);
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 5px solid var(--primary);
            transition: all 0.3s ease;
            animation: slideInUp 0.5s ease-out;
        }
        
        .section:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        }
        
        .section-title {
            color: var(--primary);
            border-bottom: 1px solid #eee;
            padding-bottom: 0.8rem;
            margin-bottom: 1.2rem;
            font-weight: 600;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
        }
        
        .section-title i {
            margin-right: 10px;
            font-size: 1.2rem;
        }
        
        /* Indicadores de riesgo */
        .risk-high {
            background-color: rgba(187, 62, 3, 0.08);
            border-left: 5px solid var(--danger) !important;
        }
        
        .risk-moderate {
            background-color: rgba(238, 155, 0, 0.08);
            border-left: 5px solid var(--warning) !important;
        }
        
        .risk-low {
            background-color: rgba(56, 142, 60, 0.08);
            border-left: 5px solid var(--secondary) !important;
        }
        
        /* Botones con efecto */
        .stButton button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 0.7rem 1.8rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,95,115,0.2);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(0,95,115,0.3);
        }
        
        /* Campos de formulario */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            border-radius: 8px !important;
            border: 1px solid #ddd !important;
            transition: all 0.3s;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(0,95,115,0.2) !important;
        }
        
        /* Efectos de carga */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .pulse-animation {
            animation: pulse 1.5s infinite;
        }
        
        /* Animaciones generales */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Efecto para resultados */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            transition: all 0.5s ease-out;
            opacity: 0;
            transform: translateY(20px);
            animation: slideInUp 0.6s ease-out forwards;
        }
        
        /* Barra de progreso personalizada */
        .progress-container {
            width: 100%;
            height: 12px;
            background-color: #f0f0f0;
            border-radius: 6px;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            border-radius: 6px;
            transition: width 1s ease-out;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: var(--dark);
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
            font-weight: normal;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header-title {
                font-size: 2rem;
            }
            
            .section {
                padding: 1.2rem;
            }
        }
    </style>
    
    <!-- Font Awesome para iconos -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Montserrat:wght@500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# JavaScript para animaciones
def load_js():
    js_code = """
    <script>
    // Animación para los elementos de resultados
    function animateResults() {
        const resultCards = document.querySelectorAll('.result-card');
        resultCards.forEach((card, index) => {
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 150);
        });
    }
    
    // Observador para animaciones al hacer scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.section').forEach(section => {
        observer.observe(section);
    });
    
    // Animación de la barra de progreso
    function animateProgressBar() {
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            const width = progressBar.getAttribute('data-width');
            progressBar.style.width = width + '%';
        }
    }
    
    // Ejecutar cuando el DOM esté listo
    document.addEventListener('DOMContentLoaded', function() {
        animateResults();
        setTimeout(animateProgressBar, 500);
    });
    
    // Actualizar para Streamlit
    document.addEventListener('streamlit:render', function() {
        animateResults();
        setTimeout(animateProgressBar, 500);
    });
    </script>
    """
    html(js_code)

load_css()
load_js()

# Cargar modelo (sin cambios)
@st.cache_resource
def load_model():
    MODEL_PATH = 'models/endometriosis_model_optimized.pkl'
    return joblib.load(MODEL_PATH)

model = load_model()
FEATURES = model.feature_names_in_

# --- Funciones del backend (sin cambios) ---
def prepare_input_data(raw_data):
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
    
    processed_data = {}
    for feature in FEATURES:
        value = raw_data.get(feature, default_values[feature])
        processed_data[feature] = int(value) if feature in ['age', 'menarche_age', 'cycle_length', 'period_duration', 'pain_level'] else float(value)
    
    return pd.DataFrame([processed_data])

def generate_explanation(input_data, probability):
    factors = []
    medical_terms = {
        'dysmenorrhea': 'Dolor menstrual severo',
        'dyspareunia': 'Dolor durante relaciones',
        'chronic_pelvic_pain': 'Dolor pélvico crónico',
        'family_history': 'Historia familiar de endometriosis'
    }
    
    if input_data['pain_level'] >= 7:
        factors.append(f"Dolor intenso ({input_data['pain_level']}/10)")
    if input_data['ca125'] > 35:
        factors.append(f"CA-125 elevado ({input_data['ca125']} U/mL)")
    if input_data['crp'] > 5:
        factors.append(f"Inflamación elevada (CRP: {input_data['crp']} mg/L)")
    
    for feature, term in medical_terms.items():
        if input_data[feature] == 1:
            factors.append(term)
    
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

# --- Interfaz de Streamlit Mejorada ---
def main():
    # Encabezado profesional con animación
    st.markdown("""
    <div class="header-container">
        <div class="header-title">
            <i class="fas fa-clinic-medical" style="margin-right: 15px;"></i>SITME
        </div>
        <div class="header-subtitle">
            Sistema Integral de Tamizaje Multimodal para Endometriosis
        </div>
        <div class="header-footer">
            <i class="fas fa-badge-check" style="margin-right: 5px;"></i>Herramienta clínica validada - v2.1
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Barra de estado del sistema
    with st.container():
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #e8f5e9; border-radius: 8px;">
                <i class="fas fa-shield-alt" style="color: #388e3c;"></i> <strong>Sistema</strong>: Operativo
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #fff8e1; border-radius: 8px;">
                <i class="fas fa-database" style="color: #ffa000;"></i> <strong>Modelo</strong>: EndoPredict v3.2
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background: #e3f2fd; border-radius: 8px;">
                <i class="fas fa-user-md" style="color: #1565c0;"></i> <strong>Uso</strong>: Profesional
            </div>
            """, unsafe_allow_html=True)
    
    # Formulario con diseño mejorado
    with st.expander("📋 FORMULARIO DE EVALUACIÓN CLÍNICA - ENDOMETRIOSIS", expanded=True):
        with st.form("endometriosis_form"):
            # Sección 1: Datos de Identificación
            st.markdown("""
            <div class="section">
                <h3 class="section-title"><i class="fas fa-id-card"></i> Datos de Identificación</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                full_name = st.text_input("Nombre completo*", key="full_name", help="Nombre y apellidos completos del paciente")
                birth_date = st.date_input("Fecha de nacimiento*", value=datetime(1990, 1, 1), help="Seleccione la fecha de nacimiento del paciente")
                age = datetime.now().year - birth_date.year
                st.text_input("Edad", value=f"{age} años", disabled=True)
                
            with col2:
                rut = st.text_input("RUT (Ej: 12345678-9)*", key="rut", help="Ingrese el RUT con guión y dígito verificador")
                blood_type = st.selectbox("Tipo de sangre", 
                                        ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                        help="Seleccione el grupo sanguíneo y factor Rh")
                
            with col3:
                insurance = st.radio("Previsión*", ["Fonasa", "Isapre", "Otro"], horizontal=True, help="Seleccione el sistema de salud del paciente")
                id_number = st.text_input("N° Identificación Clínica", disabled=True, 
                                         value=f"ENDO-{np.random.randint(1000, 9999)}",
                                         help="Identificador único generado por el sistema")
            
            # Sección 2: Antecedentes Médicos
            st.markdown("""
            <div class="section">
                <h3 class="section-title"><i class="fas fa-medical-record"></i> Antecedentes Médicos</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col4, col5 = st.columns(2)
            with col4:
                st.markdown("**Antecedentes ginecológicos**")
                gynecological_surgery = st.checkbox("Cirugías ginecológicas", help="Histerectomía, miomectomía, cirugía de ovarios, etc.")
                pelvic_inflammatory = st.checkbox("Enfermedad inflamatoria pélvica")
                ovarian_cysts = st.checkbox("Quistes ováricos recurrentes")
                
            with col5:
                st.markdown("**Antecedentes familiares**")
                family_endometriosis = st.checkbox("Endometriosis", help="Historia familiar de endometriosis en madre o hermanas")
                family_autoimmune = st.checkbox("Enfermedades autoinmunes")
                family_cancer = st.checkbox("Cáncer ginecológico", help="Cáncer de ovario, endometrio o mama en familiares de primer grado")
            
            medications = st.text_area("Medicamentos actuales*", 
                                     placeholder="Lista de medicamentos, dosis y frecuencia...",
                                     help="Incluya anticonceptivos, analgésicos, hormonas, etc.")
            
            # Sección 3: Historia Menstrual con iconos
            st.markdown("""
            <div class="section">
                <h3 class="section-title"><i class="fas fa-calendar-week"></i> Historia Menstrual</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col6, col7, col8 = st.columns(3)
            with col6:
                menarche_age = st.number_input("Edad de la menarquia (años)*", min_value=8, max_value=20, value=12,
                                             help="Edad al primer período menstrual")
                st.caption("<i class='fas fa-info-circle'></i> Rango normal: 10-14 años", unsafe_allow_html=True)
                
            with col7:
                cycle_length = st.number_input("Duración del ciclo (días)*", min_value=15, max_value=45, value=28,
                                            help="Días entre el inicio de un período y el siguiente")
                st.caption("<i class='fas fa-info-circle'></i> Rango normal: 21-35 días", unsafe_allow_html=True)
                
            with col8:
                period_duration = st.number_input("Duración del sangrado (días)*", min_value=1, max_value=15, value=5,
                                               help="Duración típica del sangrado menstrual")
                st.caption("<i class='fas fa-info-circle'></i> Rango normal: 3-7 días", unsafe_allow_html=True)
            
            last_period = st.date_input("Última menstruación*", value=datetime.now(),
                                      help="Fecha del primer día del último período menstrual")
            
            st.markdown("**Dolor menstrual (Escala 1-10)**")
            pain_level = st.slider("", 1, 10, 5, label_visibility="collapsed",
                                 help="0 = Sin dolor, 10 = Máximo dolor imaginable")
            
            st.markdown("**Características del dolor**")
            pain_col1, pain_col2, pain_col3 = st.columns(3)
            with pain_col1:
                pain_premenstrual = st.checkbox("Premenstrual", help="Dolor antes del inicio del sangrado")
                pain_menstrual = st.checkbox("Durante menstruación", help="Dolor durante los días de sangrado")
            with pain_col2:
                pain_ovulation = st.checkbox("Durante ovulación", help="Dolor a mitad del ciclo")
                pain_chronic = st.checkbox("Crónico/pélvico", help="Dolor constante en región pélvica")
            
            # Sección 4: Síntomas Actuales con iconos
            st.markdown("""
            <div class="section">
                <h3 class="section-title"><i class="fas fa-heartbeat"></i> Síntomas Actuales</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Síntomas característicos**")
            col9, col10 = st.columns(2)
            with col9:
                pain_during_sex = st.radio("Dolor durante/después de relaciones sexuales*", ["Sí", "No"], horizontal=True,
                                         help="Dolor profundo durante o después del coito")
                bowel_symptoms = st.radio("Síntomas intestinales cíclicos*", ["Sí", "No"], horizontal=True,
                                        help="Diarrea, estreñimiento o dolor al defecar asociado al ciclo menstrual")
                
            with col10:
                urinary_symptoms = st.radio("Síntomas urinarios cíclicos*", ["Sí", "No"], horizontal=True,
                                          help="Dolor al orinar o aumento de frecuencia urinaria asociado al ciclo")
                fatigue = st.radio("Fatiga crónica*", ["Sí", "No"], horizontal=True,
                                 help="Cansancio persistente no relacionado con actividad física")
            
            infertility = st.radio("Dificultades para concebir*", ["Sí", "No", "No aplica"], horizontal=True,
                                 help="Problemas de fertilidad o intentos fallidos de concepción >1 año")
            other_symptoms = st.text_area("Otros síntomas relevantes", 
                                        placeholder="Descripción detallada de otros síntomas...",
                                        help="Por ejemplo: náuseas, mareos, dolor lumbar, etc.")
            
            # Sección 5: Biomarcadores y Exámenes
            st.markdown("""
            <div class="section">
                <h3 class="section-title"><i class="fas fa-flask"></i> Biomarcadores y Exámenes</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Resultados de laboratorio**")
            col11, col12, col13 = st.columns(3)
            with col11:
                ca125 = st.number_input("CA-125 (U/mL)*", min_value=0.0, value=20.0, step=0.1,
                                      help="Marcador tumoral asociado a endometriosis")
                st.caption("<i class='fas fa-info-circle'></i> Normal: <35 U/mL", unsafe_allow_html=True)
                
            with col12:
                crp = st.number_input("Proteína C Reactiva (mg/L)*", min_value=0.0, value=3.0, step=0.1,
                                    help="Marcador de inflamación sistémica")
                st.caption("<i class='fas fa-info-circle'></i> Normal: <10 mg/L", unsafe_allow_html=True)
                
            with col13:
                il6 = st.number_input("IL-6 (pg/mL)", min_value=0.0, value=0.0, step=0.1,
                                    help="Interleucina 6 - Marcador inflamatorio")
                st.caption("<i class='fas fa-info-circle'></i> Normal: <5 pg/mL", unsafe_allow_html=True)
            
            imaging = st.selectbox("Resultados de imágenes*", 
                                 ["", "Normal", "Endometrioma(s) ovárico(s)", 
                                  "Adenomiosis sospechosa", "Endometriosis profunda sospechosa", "Otros hallazgos"],
                                 help="Hallazgos relevantes en ecografía o resonancia magnética")
            
            imaging_details = st.text_area("Detalles de estudios por imágenes", 
                                         placeholder="Descripción detallada de ecografía, resonancia u otros estudios...",
                                         help="Incluya tamaño de endometriomas, localización de implantes, etc.")
            
            # Botón de envío con icono animado
            submitted = st.form_submit_button("🚀 Evaluar Riesgo de Endometriosis", 
                                            use_container_width=True,
                                            type="primary",
                                            help="Haga clic para procesar la información y generar el análisis de riesgo")
    
    # Procesamiento de resultados con animaciones
    if submitted:
        if not full_name:
            st.error("❌ Por favor ingrese el nombre completo del paciente")
        else:
            with st.spinner("🔍 Analizando datos clínicos..."):
                # Simular procesamiento con barra de progreso
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                
                input_data = {
                    'full_name': full_name,
                    'age': age,
                    'bmi': 23.5,  # Podrías calcularlo si añades peso/talla
                    'menarche_age': menarche_age,
                    'cycle_length': cycle_length,
                    'period_duration': period_duration,
                    'pain_level': pain_level,
                    'ca125': ca125,
                    'crp': crp,
                    'dysmenorrhea': 1 if pain_menstrual else 0,
                    'chronic_pelvic_pain': int(pain_chronic),
                    'infertility': 1 if infertility == "Sí" else 0,
                    'family_history': int(family_endometriosis),
                    'gynecological_surgery': gynecological_surgery,
                    'pelvic_inflammatory': pelvic_inflammatory,
                    'ovarian_cysts': ovarian_cysts,
                    'family_autoimmune': family_autoimmune,
                    'family_cancer': family_cancer,
                    'bowel_symptoms': 1 if bowel_symptoms == "Sí" else 0,
                    'urinary_symptoms': 1 if urinary_symptoms == "Sí" else 0,
                    'fatigue': 1 if fatigue == "Sí" else 0,
                    'other_symptoms': other_symptoms,
                    'imaging': imaging,
                    'imaging_details': imaging_details,
                    'medications': medications
                }
                
                input_df = prepare_input_data(input_data)
                proba = model.predict_proba(input_df)[0][1]
                explanation = generate_explanation(input_df.iloc[0], proba)
                
                # Mostrar resultados con animaciones
                display_results(input_data, proba, explanation)

def display_results(input_data, probability, explanation):
    """Muestra los resultados de la predicción con diseño mejorado"""
    risk_class = {
        "ALTO": "risk-high",
        "MODERADO": "risk-moderate",
        "BAJO": "risk-low"
    }[explanation['risk_level']]
    
    # Resultado principal con animación
    st.markdown(f"""
    <div class="section {risk_class}">
        <h3 class="section-title"><i class="fas fa-chart-line"></i> Resultados de la Evaluación</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Tarjeta de probabilidad con animación
    prob_percent = round(probability * 100, 1)
    risk_color = {
        "ALTO": "#d32f2f",
        "MODERADO": "#ffa000",
        "BAJO": "#388e3c"
    }[explanation['risk_level']]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <div class="result-card" style="text-align: center; padding: 1.5rem; border-radius: 12px; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 2.5rem; font-weight: 700; color: {risk_color}; margin-bottom: 0.5rem;">
                {prob_percent}%
            </div>
            <div style="font-size: 1rem; color: #666; margin-bottom: 1rem;">
                Probabilidad de endometriosis
            </div>
            <div style="margin-bottom: 1rem;">
                <span style="background-color: {risk_color}; color: white; 
                            padding: 0.5rem 1.5rem; border-radius: 20px; 
                            font-weight: 600; display: inline-block;">
                    <i class="fas fa-{'exclamation-triangle' if explanation['risk_level'] == 'ALTO' else 'info-circle'}"></i> 
                    Riesgo {explanation['risk_level']}
                </span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" data-width="{prob_percent}" style="background-color: {risk_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: var(--primary); border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                <i class="fas fa-search-plus"></i> Factores Clave Identificados
            </h4>
        """, unsafe_allow_html=True)
        
        if explanation['key_factors']:
            for factor in explanation['key_factors']:
                st.markdown(f"<p style='margin-bottom: 0.5rem;'><i class='fas fa-circle' style='font-size: 0.5rem; color: {risk_color}; vertical-align: middle; margin-right: 0.5rem;'></i> {factor}</p>", unsafe_allow_html=True)
        else:
            st.info("No se identificaron factores de riesgo significativos")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-card">
            <h4 style="color: var(--primary); border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                <i class="fas fa-clipboard-check"></i> Recomendaciones Clínicas
            </h4>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(explanation['recommendations']):
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 0.8rem; align-items: flex-start;">
                <div style="background-color: {risk_color}; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.8rem; flex-shrink: 0;">
                    {i+1}
                </div>
                <div style="flex-grow: 1;">
                    {rec}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sección de resumen del paciente con animación
    st.markdown("""
    <div class="section">
        <h3 class="section-title"><i class="fas fa-file-medical-alt"></i> Resumen Clínico del Paciente</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: var(--primary); border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                <i class="fas fa-user-circle"></i> Datos Personales
            </h4>
            <p><strong>Nombre:</strong> {}</p>
            <p><strong>Edad:</strong> {} años</p>
            <p><strong>Menarquia:</strong> {} años</p>
            <p><strong>Previsión:</strong> {}</p>
        </div>
        """.format(input_data['full_name'], input_data['age'], input_data['menarche_age'], input_data.get('insurance', 'No especificado')), unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: var(--primary); border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                <i class="fas fa-calendar-alt"></i> Historia Menstrual
            </h4>
            <p><strong>Ciclo:</strong> {} días</p>
            <p><strong>Sangrado:</strong> {} días</p>
            <p><strong>Dolor:</strong> {}/10</p>
            <p><strong>Última menstruación:</strong> {}</p>
        </div>
        """.format(input_data['cycle_length'], input_data['period_duration'], input_data['pain_level'], input_data.get('last_period', 'No especificado')), unsafe_allow_html=True)
        
    with col5:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: var(--primary); border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                <i class="fas fa-flask"></i> Biomarcadores
            </h4>
            <p><strong>CA-125:</strong> {} U/mL</p>
            <p><strong>PCR:</strong> {} mg/L</p>
            <p><strong>IL-6:</strong> {} pg/mL</p>
            <p><strong>Historia familiar:</strong> {}</p>
        </div>
        """.format(input_data['ca125'], input_data['crp'], input_data.get('il6', 'No especificado'), 
                  "Positiva" if input_data.get('family_endometriosis') else "Negativa"), unsafe_allow_html=True)
    
    # Botón para generar PDF con animación
    st.markdown("""
    <div class="section">
        <h3 class="section-title"><i class="fas fa-file-pdf"></i> Generar Informe Clínico</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🖨️ Generar Informe en PDF", use_container_width=True, key="generate_pdf"):
        with st.spinner("Generando informe PDF..."):
            generate_pdf(input_data, probability, explanation)
            st.success("Informe generado correctamente")
            st.balloons()  # Efecto de confeti al completar

# Función para generar PDF (sin cambios)
def generate_pdf(input_data, probability, explanation):
    """Genera el PDF clínico profesional"""
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
    
    elements = []
    
    # 1. Encabezado
    elements.append(Paragraph("INFORME CLÍNICO - EVALUACIÓN DE ENDOMETRIOSIS", header_style))
    elements.append(Paragraph("SITME - Sistema Integral de Tamizaje Multimodal para Endometriosis", subtitle_style))
    elements.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
    elements.append(Spacer(1, 12))
    
    # 2. Información del paciente
    elements.append(Paragraph("INFORMACIÓN DE LA PACIENTE:", subtitle_style))
    
    patient_data = [
        [Paragraph("<b>Nombre:</b>", bold_style), Paragraph(input_data['full_name'], normal_style)],
        [Paragraph("<b>Edad:</b>", bold_style), Paragraph(f"{input_data['age']} años", normal_style)],
        [Paragraph("<b>Menarquia:</b>", bold_style), Paragraph(f"{input_data['menarche_age']} años", normal_style)],
        [Paragraph("<b>Ciclo menstrual:</b>", bold_style), Paragraph(f"{input_data['cycle_length']} días", normal_style)],
        [Paragraph("<b>Duración período:</b>", bold_style), Paragraph(f"{input_data['period_duration']} días", normal_style)],
        [Paragraph("<b>Dolor menstrual:</b>", bold_style), Paragraph(f"{input_data['pain_level']}/10", normal_style)]
    ]
    
    patient_table = Table(patient_data, colWidths=[120, 300])
    patient_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 12))
    
    # 3. Resultados
    elements.append(Paragraph("RESULTADOS DE LA EVALUACIÓN:", subtitle_style))
    
    results_data = [
        [Paragraph("<b>Probabilidad:</b>", bold_style), Paragraph(f"{round(probability*100, 1)}%", normal_style)],
        [Paragraph("<b>Nivel de Riesgo:</b>", bold_style), Paragraph(explanation['risk_level'], normal_style)],
        [Paragraph("<b>Factores Clave:</b>", bold_style), 
         Paragraph(', '.join(explanation['key_factors']) or 'No identificados', normal_style)]
    ]
    
    results_table = Table(results_data, colWidths=[120, 300])
    results_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 6))
    
    # 4. Recomendaciones
    elements.append(Paragraph("<b>RECOMENDACIONES:</b>", subtitle_style))
    for recommendation in explanation['recommendations']:
        elements.append(Paragraph(f"• {recommendation}", normal_style))
        elements.append(Spacer(1, 2))
    
    elements.append(Spacer(1, 12))
    
    # 5. Notas
    elements.append(Paragraph("<b>NOTAS CLÍNICAS:</b>", subtitle_style))
    elements.append(Paragraph("Este informe ha sido generado automáticamente por el sistema SITME y debe ser interpretado por un profesional médico calificado.", normal_style))
    elements.append(Paragraph("Los resultados se basan en algoritmos predictivos validados pero no sustituyen el juicio clínico profesional.", normal_style))
    
    # Generar PDF
    doc.build(elements)
    
    # Descargar
    st.download_button(
        label="⬇️ Descargar Informe Completo",
        data=buffer.getvalue(),
        file_name=f"informe_endometriosis_{input_data['full_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        key="download_pdf"
    )

if __name__ == "__main__":
    main()