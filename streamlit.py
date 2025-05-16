import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import base64

# Configuración inicial
st.set_page_config(
    page_title="SITME - Sistema de Tamizaje para Endometriosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("static/style.css")

# Colores personalizados basados en tu CSS
PRIMARY = "#143f6a"
PRIMARY_LIGHT = "#4d7bb6"
SECONDARY = "#26a69a"
DANGER = "#d32f2f"
WARNING = "#ffa000"
SUCCESS = "#388e3c"
INFO = "#1976d2"
LIGHT = "#f5f5f5"
DARK = "#212121"
GRAY = "#757575"
TEXT = "#424242"
BORDER = "#e0e0e0"
BACKGROUND = "#f9f6fc"

# Generar ID clínico aleatorio
def generate_clinic_id():
    return f"ENDO-{random.randint(1000, 9999)}"

# Validación de RUT (para Chile)
def validate_rut(rut):
    rut = rut.upper().replace(".", "").replace("-", "")
    if not rut or len(rut) < 2:
        return False
    
    body = rut[:-1]
    dv = rut[-1]
    
    try:
        body = int(body)
    except ValueError:
        return False
    
    suma = 0
    multiplier = 2
    
    for c in reversed(str(body)):
        suma += int(c) * multiplier
        multiplier = multiplier + 1 if multiplier < 7 else 2
    
    resto = suma % 11
    computed_dv = {10: "K", 0: "0"}.get(11 - resto, str(11 - resto))
    
    return computed_dv == dv

# Calcular edad desde fecha de nacimiento
def calculate_age(birth_date):
    today = datetime.now()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

# Calcular IMC
def calculate_bmi(height, weight):
    if height is None or weight is None:
        return None  # or raise an error, or return a default value
    if height > 0 and weight > 0:
        return weight / (height / 100) ** 2  # assuming height is in cm
    return None

# Simular predicción (para desarrollo)
def simulate_prediction(form_data):
    probability = 0.2
    
    # Factores de riesgo simulados
    if form_data.get('pain_level', 0) >= 7:
        probability += 0.25
    if form_data.get('pain_during_sex', False):
        probability += 0.15
    if form_data.get('family_endometriosis', False):
        probability += 0.1
    if form_data.get('bowel_symptoms', False):
        probability += 0.1
    if form_data.get('urinary_symptoms', False):
        probability += 0.1
    if form_data.get('fatigue', False):
        probability += 0.05
    if form_data.get('infertility', False):
        probability += 0.05
    
    # Ajustar según biomarcadores
    if form_data.get('ca125', 0) > 35:
        probability += 0.1
    if form_data.get('il6', 0) > 5:
        probability += 0.05
    if form_data.get('crp', 0) > 10:
        probability += 0.05
    
    probability = min(probability, 0.95)
    
    risk_level = "high" if probability > 0.7 else "moderate" if probability > 0.4 else "low"
    
    return {
        "probability": probability,
        "risk_level": risk_level,
        "risk_factors": [
            "Dolor menstrual severo" if form_data.get('pain_level', 0) >= 7 else None,
            "Dolor durante relaciones sexuales" if form_data.get('pain_during_sex', False) else None,
            "Antecedentes familiares" if form_data.get('family_endometriosis', False) else None,
            "Síntomas intestinales" if form_data.get('bowel_symptoms', False) else None,
            "CA-125 elevado" if form_data.get('ca125', 0) > 35 else None
        ]
    }

# Mostrar resultados
def display_results(results, form_data):
    probability_percent = round(results['probability'] * 100)
    
    with st.container():
        st.header("Resultados de la Evaluación")
        
        # Mostrar nivel de riesgo
        risk_colors = {
            "high": DANGER,
            "moderate": WARNING,
            "low": SUCCESS
        }
        
        st.markdown(f"""
        <div class="risk-header" style="border-left: 5px solid {risk_colors[results['risk_level']]}; padding-left: 15px;">
            <h2 style="margin: 0;">Riesgo {'Alto' if results['risk_level'] == 'high' else 'Moderado' if results['risk_level'] == 'moderate' else 'Bajo'} de Endometriosis</h2>
            <p>{'Los síntomas sugieren alta probabilidad de endometriosis. Se recomienda evaluación especializada urgente.' if results['risk_level'] == 'high' 
               else 'Presenta varios indicadores de endometriosis que justifican mayor investigación.' if results['risk_level'] == 'moderate' 
               else 'Los síntomas actuales no sugieren endometriosis como diagnóstico principal.'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico de probabilidad
        cols = st.columns([1, 3])
        with cols[0]:
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="width: 150px; height: 150px; margin: 0 auto; border-radius: 50%; 
                            border: 10px solid {risk_colors[results['risk_level']]}; 
                            display: flex; align-items: center; justify-content: center;
                            font-size: 2rem; font-weight: bold;">
                    {probability_percent}%
                </div>
                <p style="text-align: center; margin-top: 10px; font-weight: 500;">
                    Probabilidad de Endometriosis
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            # Factores de riesgo
            st.subheader("Factores de Riesgo Identificados")
            risk_factors = [f for f in results['risk_factors'] if f is not None]
            
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="color: {risk_colors[results['risk_level']]}; margin-right: 8px;">•</span>
                        <span>{factor}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No se identificaron factores de riesgo significativos.")
        
        # Recomendaciones
        st.subheader("Recomendaciones Clínicas")
        
        if results['risk_level'] == "high":
            recommendations = [
                "Consulta urgente con especialista en endometriosis",
                "Considerar laparoscopia diagnóstica",
                "Ecografía transvaginal especializada",
                "Evaluación multidisciplinaria (dolor, fertilidad)"
            ]
        elif results['risk_level'] == "moderate":
            recommendations = [
                "Consulta con ginecólogo",
                "Prueba de tratamiento médico de 3-6 meses",
                "Considerar imagenología avanzada",
                "Seguimiento estrecho de síntomas"
            ]
        else:
            recommendations = [
                "Manejo conservador con seguimiento",
                "Educación sobre síntomas de alerta",
                "Analgesia según necesidad",
                "Reevaluar si síntomas progresan"
            ]
            
        for rec in recommendations:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="color: {risk_colors[results['risk_level']]}; margin-right: 8px;">•</span>
                <span>{rec}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Resumen del paciente
        st.subheader("Resumen Clínico del Paciente")
        
        summary_data = {
            "Nombre": form_data.get('full_name', ''),
            "Edad": f"{form_data.get('age', '')} años" if form_data.get('age') else "",
            "Menarquia": f"{form_data.get('menarche_age', '')} años" if form_data.get('menarche_age') else "",
            "Ciclo menstrual": f"{form_data.get('cycle_length', '')} días" if form_data.get('cycle_length') else "",
            "Duración período": f"{form_data.get('period_duration', '')} días" if form_data.get('period_duration') else "",
            "Dolor menstrual": f"{form_data.get('pain_level', '')}/10" if form_data.get('pain_level') else "",
            "Dispareunia": "Sí" if form_data.get('pain_during_sex', False) else "No",
            "Antecedentes familiares": "Sí" if form_data.get('family_endometriosis', False) else "No",
            "CA-125": f"{form_data['ca125']} U/mL" if form_data.get('ca125') is not None else "No medido",
            "IMC": f"{form_data['bmi']:.1f}" if form_data.get('bmi') is not None else "No calculado"
        }
        
        # Mostrar en columnas
        cols = st.columns(3)
        for i, (key, value) in enumerate(summary_data.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background: {LIGHT}; padding: 12px; border-radius: 8px; margin-bottom: 10px;
                            border-left: 3px solid {PRIMARY};">
                    <div style="font-size: 0.85rem; color: {GRAY}; margin-bottom: 4px;">{key}</div>
                    <div style="font-weight: 500; color: {DARK};">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Guías clínicas
        st.subheader("Guías Clínicas Aplicables")
        
        if results['risk_level'] == "high":
            guidelines = {
                "ASRM": "Paciente cumple criterios para evaluación laparoscópica diagnóstica según ASRM.",
                "ESHRE": "Derivación a unidad especializada en endometriosis. Considerar tratamiento médico agresivo.",
                "NICE": "Evaluación multidisciplinaria (ginecólogo, especialista en dolor, fertilidad)."
            }
        elif results['risk_level'] == "moderate":
            guidelines = {
                "ASRM": "Paciente puede beneficiarse de tratamiento médico empírico según ASRM.",
                "ESHRE": "Prueba de tratamiento médico de 3-6 meses. Si no mejora, considerar evaluación quirúrgica.",
                "NICE": "Manejo inicial con AINEs y terapia hormonal. Evaluar respuesta en 3 meses."
            }
        else:
            guidelines = {
                "ASRM": "ASRM sugiere manejo conservador con seguimiento. Educación sobre síntomas de alerta.",
                "ESHRE": "Manejo sintomático. Reevaluar si síntomas progresan o cambian.",
                "NICE": "Educación y analgesia según necesidad. Seguimiento anual o ante nuevos síntomas."
            }
            
        for org, guideline in guidelines.items():
            with st.expander(org):
                st.write(guideline)
        
        # Botón para descargar PDF (simulado)
        if st.button("📄 Descargar Ficha Clínica", use_container_width=True,
                    help="Generar un reporte en formato PDF con los resultados"):
            st.success("Generando documento... (simulación)")

# Diseño de la aplicación
def main():
    # Encabezado
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_LIGHT}); 
                padding: 30px; border-radius: 10px; color: white; 
                text-align: center; margin-bottom: 30px;">
        <h1 style="margin: 0;">SITME</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Sistema Integral de Tamizaje Multimodal para Endometriosis</p>
        <div style="background: rgba(0,0,0,0.1); display: inline-block; 
                    padding: 5px 15px; border-radius: 20px; margin-top: 15px;">
            <i class="fas fa-id-card" style="margin-right: 8px;"></i>
            ID Clínico: {generate_clinic_id()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar form_data en session_state si no existe
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Formulario en pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Identificación", 
        "🏥 Antecedentes", 
        "📅 Menstrual", 
        "🩺 Síntomas", 
        "🧪 Biomarcadores", 
        "🔍 Evaluación"
    ])
    
    with tab1:
        st.subheader("Datos de Identificación")
        
        cols = st.columns(2)
        with cols[0]:
            st.session_state.form_data['full_name'] = st.text_input(
                "Nombre completo*", 
                key="full_name",
                placeholder="Ingrese nombre completo",
                value=""
            )
            st.session_state.form_data['rut'] = st.text_input(
                "RUT", 
                placeholder="12345678-9", 
                key="rut",
                value=""
            )
            if st.session_state.form_data['rut'] and not validate_rut(st.session_state.form_data['rut']):
                st.error("RUT inválido. Verifique el dígito verificador.")
            
        with cols[1]:
            birth_date = st.date_input(
                "Fecha de nacimiento*", 
                max_value=datetime.now(), 
                key="birth_date",
                value=None
            )
            st.session_state.form_data['age'] = calculate_age(birth_date) if birth_date else None
            st.text_input(
                "Edad", 
                value=st.session_state.form_data['age'] if st.session_state.form_data.get('age') else "", 
                disabled=True, 
                key="age"
            )
            
        cols = st.columns(2)
        with cols[0]:
            st.session_state.form_data['blood_type'] = st.selectbox(
                "Tipo de sangre",
                ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                key="blood_type",
                index=0
            )
            
        with cols[1]:
            st.session_state.form_data['insurance'] = st.radio(
                "Previsión*",
                ["Fonasa", "Isapre"],
                horizontal=True,
                key="insurance",
                index=None  # Ninguna opción seleccionada por defecto
            )
    
    with tab2:
        st.subheader("Antecedentes Médicos")
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Antecedentes ginecológicos**")
            st.session_state.form_data['gynecological_surgery'] = st.checkbox(
                "Cirugías ginecológicas", 
                key="gynecological_surgery",
                value=False
            )
            st.session_state.form_data['pelvic_inflammatory'] = st.checkbox(
                "Enfermedad inflamatoria pélvica", 
                key="pelvic_inflammatory",
                value=False
            )
            st.session_state.form_data['ovarian_cysts'] = st.checkbox(
                "Quistes ováricos recurrentes", 
                key="ovarian_cysts",
                value=False
            )
            
        with cols[1]:
            st.markdown("**Antecedentes familiares**")
            st.session_state.form_data['family_endometriosis'] = st.checkbox(
                "Endometriosis", 
                key="family_endometriosis",
                value=False
            )
            st.session_state.form_data['family_autoimmune'] = st.checkbox(
                "Enfermedades autoinmunes", 
                key="family_autoimmune",
                value=False
            )
            st.session_state.form_data['family_cancer'] = st.checkbox(
                "Cáncer ginecológico", 
                key="family_cancer",
                value=False
            )
            
        with cols[2]:
            st.markdown("**Comorbilidades**")
            st.session_state.form_data['comorbidity_autoimmune'] = st.checkbox(
                "Enfermedad autoinmune", 
                key="comorbidity_autoimmune",
                value=False
            )
            st.session_state.form_data['comorbidity_thyroid'] = st.checkbox(
                "Trastorno tiroideo", 
                key="comorbidity_thyroid",
                value=False
            )
            st.session_state.form_data['comorbidity_ibs'] = st.checkbox(
                "Síndrome de intestino irritable", 
                key="comorbidity_ibs",
                value=False
            )
            
        st.session_state.form_data['medications'] = st.text_area(
            "Medicamentos actuales",
            placeholder="Lista de medicamentos, dosis y frecuencia",
            key="medications",
            value=""
        )
    
    with tab3:
        st.subheader("Historial Menstrual")
        
        cols = st.columns(3)
        with cols[0]:
            st.session_state.form_data['menarche_age'] = st.number_input(
                "Edad de la menarquia*", 
                min_value=8, 
                max_value=20, 
                step=1, 
                key="menarche_age",
                value=None
            )
            st.caption("Rango normal: 10-14 años")
            
            st.session_state.form_data['cycle_length'] = st.number_input(
                "Duración del ciclo (días)*", 
                min_value=15, 
                max_value=45, 
                step=1, 
                key="cycle_length",
                value=None
            )
            st.caption("Rango normal: 21-35 días")
            
        with cols[1]:
            st.session_state.form_data['period_duration'] = st.number_input(
                "Duración del sangrado (días)*", 
                min_value=1, 
                max_value=15, 
                step=1, 
                key="period_duration",
                value=None
            )
            st.caption("Rango normal: 3-7 días")
            
            st.session_state.form_data['last_period'] = st.date_input(
                "Última menstruación*", 
                key="last_period",
                value=None
            )
            
        with cols[2]:
            st.session_state.form_data['pain_level'] = st.slider(
                "Dolor menstrual (1-10)*", 
                min_value=1, 
                max_value=10, 
                value=1,  # Valor mínimo, no preseleccionado
                key="pain_level"
            )
            
            st.markdown("**Características del dolor**")
            st.session_state.form_data['pain_premenstrual'] = st.checkbox(
                "Premenstrual", 
                key="pain_premenstrual",
                value=False
            )
            st.session_state.form_data['pain_menstrual'] = st.checkbox(
                "Durante menstruación", 
                key="pain_menstrual",
                value=False
            )
            st.session_state.form_data['pain_ovulation'] = st.checkbox(
                "Durante ovulación", 
                key="pain_ovulation",
                value=False
            )
            st.session_state.form_data['pain_chronic'] = st.checkbox(
                "Crónico/pélvico", 
                key="pain_chronic",
                value=False
            )
    
    with tab4:
        st.subheader("Síntomas Actuales")
        
        cols = st.columns(2)
        with cols[0]:
            st.session_state.form_data['pain_during_sex'] = st.radio(
                "Dolor durante/después de relaciones sexuales*",
                ["Sí", "No"],
                horizontal=True,
                key="pain_during_sex",
                index=None  # Ninguna opción seleccionada
            ) == "Sí"
            
            st.session_state.form_data['bowel_symptoms'] = st.radio(
                "Síntomas intestinales cíclicos*",
                ["Sí", "No"],
                horizontal=True,
                key="bowel_symptoms",
                index=None
            ) == "Sí"
            
            st.session_state.form_data['urinary_symptoms'] = st.radio(
                "Síntomas urinarios cíclicos*",
                ["Sí", "No"],
                horizontal=True,
                key="urinary_symptoms",
                index=None
            ) == "Sí"
            
        with cols[1]:
            st.session_state.form_data['fatigue'] = st.radio(
                "Fatiga crónica*",
                ["Sí", "No"],
                horizontal=True,
                key="fatigue",
                index=None
            ) == "Sí"
            
            st.session_state.form_data['infertility'] = st.radio(
                "Dificultades para concebir",
                ["Sí", "No", "No aplica"],
                horizontal=True,
                key="infertility",
                index=None
            ) == "Sí"
            
        st.session_state.form_data['other_symptoms'] = st.text_area(
            "Otros síntomas",
            placeholder="Descripción de otros síntomas relevantes",
            key="other_symptoms",
            value=""
        )
    
    with tab5:
        st.subheader("Biomarcadores y Exámenes")
        
        cols = st.columns(3)
        with cols[0]:
            st.session_state.form_data['ca125'] = st.number_input(
                "CA-125 (U/mL)", 
                min_value=0.0, 
                step=0.1, 
                key="ca125",
                value=None,
                placeholder="Ej: 25.5"
            )
            st.caption("Normal: <35 U/mL")
            
            st.session_state.form_data['il6'] = st.number_input(
                "IL-6 (pg/mL)", 
                min_value=0.0, 
                step=0.1, 
                key="il6",
                value=None,
                placeholder="Ej: 3.2"
            )
            st.caption("Normal: <5 pg/mL")
            
        with cols[1]:
            st.session_state.form_data['tnf_alpha'] = st.number_input(
                "TNF-α (pg/mL)", 
                min_value=0.0, 
                step=0.1, 
                key="tnf_alpha",
                value=None,
                placeholder="Ej: 10.0"
            )
            st.caption("Normal: <15 pg/mL")
            
            st.session_state.form_data['vegf'] = st.number_input(
                "VEGF (pg/mL)", 
                min_value=0.0, 
                step=0.1, 
                key="vegf",
                value=None,
                placeholder="Ej: 350.0"
            )
            st.caption("Normal: <500 pg/mL")
            
        with cols[2]:
            st.session_state.form_data['amh'] = st.number_input(
                "Hormona Antimulleriana (ng/mL)", 
                min_value=0.0, 
                step=0.1, 
                key="amh",
                value=None,
                placeholder="Ej: 2.5"
            )
            st.caption("Normal: 1.0-4.0 ng/mL")
            
            st.session_state.form_data['crp'] = st.number_input(
                "Proteína C Reactiva (mg/L)", 
                min_value=0.0, 
                step=0.1, 
                key="crp",
                value=None,
                placeholder="Ej: 5.0"
            )
            st.caption("Normal: <10 mg/L")
            
        st.session_state.form_data['imaging'] = st.selectbox(
            "Resultados de imágenes",
            ["", "Normal", "Endometrioma(s) ovárico(s)", "Adenomiosis sospechosa", 
             "Endometriosis profunda sospechosa", "Otros hallazgos"],
            key="imaging",
            index=0
        )
        
        st.session_state.form_data['imaging_details'] = st.text_area(
            "Detalles de estudios por imágenes",
            placeholder="Descripción de ecografía, resonancia u otros estudios",
            key="imaging_details",
            value=""
        )
    
    with tab6:
        st.subheader("Evaluación Clínica")
        
        cols = st.columns(3)
        with cols[0]:
            height = st.number_input(
                "Estatura (cm)", 
                min_value=100, 
                max_value=250, 
                step=1, 
                key="height",
                value=None,
                placeholder="Ej: 165"
            )
            
            weight = st.number_input(
                "Peso (kg)", 
                min_value=30.0, 
                max_value=200.0, 
                step=0.1, 
                key="weight",
                value=None,
                placeholder="Ej: 65.5"
            )
            
            st.session_state.form_data['bmi'] = calculate_bmi(height, weight)
            st.text_input(
                "IMC (Índice de Masa Corporal)", 
                value=f"{st.session_state.form_data['bmi']:.1f}" if st.session_state.form_data.get('bmi') is not None else "", 
                disabled=True, 
                key="bmi"
            )
            st.caption("Normal: <18.5, >24.9")
            
        with cols[1]:
            st.session_state.form_data['pelvic_exam'] = st.selectbox(
                "Examen pélvico",
                ["", "Normal", "Dolor a la movilización uterina", 
                 "Nódulos en ligamentos uterosacros", "Útero fijo/retroverso", "Masa anexial"],
                key="pelvic_exam",
                index=0
            )
            
        with cols[2]:
            st.session_state.form_data['vaginal_exam'] = st.selectbox(
                "Examen vaginal",
                ["", "Normal", "Dolor en fondos de saco", 
                 "Nódulos palpables", "Otros hallazgos"],
                key="vaginal_exam",
                index=0
            )
            
        st.session_state.form_data['clinical_notes'] = st.text_area(
            "Notas clínicas",
            placeholder="Observaciones relevantes del examen físico",
            key="clinical_notes",
            value=""
        )
        
        # Botones de acción SOLO EN LA ÚLTIMA PESTAÑA
        cols = st.columns(2)
        with cols[0]:
            if st.button("🔄 Limpiar Todo", 
                        use_container_width=True,
                        help="Borrar todos los datos del formulario"):
                st.session_state.form_data = {}
                st.experimental_rerun()
        
        with cols[1]:
            if st.button("📊 Evaluar Riesgo de Endometriosis", 
                        type="primary", 
                        use_container_width=True,
                        help="Analizar los datos ingresados para evaluar riesgo de endometriosis"):
                # Validar campos requeridos
                required_fields = [
                    ('full_name', "Nombre completo"),
                    ('birth_date', "Fecha de nacimiento"),
                    ('menarche_age', "Edad de la menarquia"),
                    ('cycle_length', "Duración del ciclo"),
                    ('period_duration', "Duración del sangrado"),
                    ('last_period', "Última menstruación"),
                    ('insurance', "Previsión")
                ]
                
                missing_fields = []
                for field, name in required_fields:
                    if not st.session_state.form_data.get(field):
                        missing_fields.append(name)
                
                if missing_fields:
                    st.error(f"Por favor complete los siguientes campos requeridos: {', '.join(missing_fields)}")
                else:
                    # Simular predicción (en producción, llamar a la API)
                    results = simulate_prediction(st.session_state.form_data)
                    display_results(results, st.session_state.form_data)

if __name__ == "__main__":
    main()