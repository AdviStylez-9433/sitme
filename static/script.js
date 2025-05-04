// Generar ID clínico aleatorio
document.getElementById('clinicId').textContent = Math.floor(1000 + Math.random() * 9000);

document.addEventListener('DOMContentLoaded', function () {
    const rutInput = document.getElementById('rut');
    const rutError = document.getElementById('rut-error');

    // Formateo automático mientras escribe
    rutInput.addEventListener('input', function (e) {
        // Limpiar error si está visible
        rutError.style.display = 'none';
        rutInput.classList.remove('invalid');

        // Formateo del RUT
        let value = e.target.value.replace(/[^\dkK-]/gi, '');

        if (value.length > 1) {
            value = value.replace(/-/g, '');
            if (value.length > 7) {
                value = value.substring(0, value.length - 1) + '-' + value.slice(-1);
            }
            if (value.includes('-')) {
                const parts = value.split('-');
                value = parts[0] + '-' + parts[1].toUpperCase();
            }
        }

        e.target.value = value;
    });

    // Validar al perder foco
    rutInput.addEventListener('blur', function (e) {
        if (e.target.value && !validarRUT(e.target.value)) {
            rutInput.classList.add('invalid');
            rutError.textContent = 'RUT inválido. Verifique el dígito verificador.';
            rutError.style.display = 'block';
        }
    });

    function validarRUT(rut) {
        rut = rut.replace(/[^\dkK-]/gi, '');
        if (!/^\d{7,8}-[\dkK]$/i.test(rut)) return false;

        const [numero, dv] = rut.split('-');
        return calcularDV(numero) === dv.toUpperCase();
    }

    function calcularDV(numero) {
        let suma = 0;
        let multiplo = 2;

        for (let i = numero.length - 1; i >= 0; i--) {
            suma += parseInt(numero.charAt(i)) * multiplo;
            multiplo = multiplo === 7 ? 2 : multiplo + 1;
        }

        const resto = suma % 11;
        return resto === 0 ? '0' : resto === 1 ? 'K' : (11 - resto).toString();
    }
});

// Calcular edad automáticamente desde fecha de nacimiento
document.getElementById('birth_date').addEventListener('change', function () {
    const birthDate = new Date(this.value);
    const ageDifMs = Date.now() - birthDate.getTime();
    const ageDate = new Date(ageDifMs);
    const age = Math.abs(ageDate.getUTCFullYear() - 1970);
    document.getElementById('age').value = age;
});

document.addEventListener('DOMContentLoaded', function () {
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiInput = document.getElementById('bmi');

    function calculateBMI() {
        const height = parseFloat(heightInput.value) / 100; // Convertir cm a m
        const weight = parseFloat(weightInput.value);

        if (height && weight) {
            const bmi = weight / (height * height);
            bmiInput.value = bmi.toFixed(1);
        } else {
            bmiInput.value = '';
        }
    }

    heightInput.addEventListener('input', calculateBMI);
    weightInput.addEventListener('input', calculateBMI);
});

// Manejar envío del formulario
document.getElementById('endometriosisForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const button = this.querySelector('button');
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';

    // Validar campos requeridos
    const requiredFields = ['full_name', 'birth_date', 'menarche_age', 'cycle_length', 'period_duration', 'last_period'];
    for (const field of requiredFields) {
        if (!document.getElementById(field).value) {
            alert(`Por favor complete el campo requerido: "${document.querySelector(`label[for="${field}"]`).textContent.replace(' *', '')}"`);
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-heartbeat"></i> Evaluar Riesgo de Endometriosis';
            return;
        }
    }

    // Recoger todos los datos del formulario
    const formData = {
        personal: {
            full_name: document.getElementById('full_name').value,
            id_number: document.getElementById('rut').value,
            birth_date: document.getElementById('birth_date').value,
            age: document.getElementById('age').value,
            blood_type: document.getElementById('blood_type').value,
            insurance: document.getElementById('insurance').value
        },
        history: {
            gynecological_surgery: document.getElementById('gynecological_surgery').checked,
            pelvic_inflammatory: document.getElementById('pelvic_inflammatory').checked,
            ovarian_cysts: document.getElementById('ovarian_cysts').checked,
            family_endometriosis: document.getElementById('family_endometriosis').checked,
            family_autoimmune: document.getElementById('family_autoimmune').checked,
            family_cancer: document.getElementById('family_cancer').checked,
            comorbidity_autoimmune: document.getElementById('comorbidity_autoimmune').checked,
            comorbidity_thyroid: document.getElementById('comorbidity_thyroid').checked,
            comorbidity_ibs: document.getElementById('comorbidity_ibs').checked,
            medications: document.getElementById('medications').value
        },
        menstrual: {
            menarche_age: document.getElementById('menarche_age').value,
            cycle_length: document.getElementById('cycle_length').value,
            period_duration: document.getElementById('period_duration').value,
            last_period: document.getElementById('last_period').value,
            pain_level: document.getElementById('pain_level'

            ).value,
            pain_premenstrual: document.getElementById('pain_premenstrual').checked,
            pain_menstrual: document.getElementById('pain_menstrual').checked,
            pain_ovulation: document.getElementById('pain_ovulation').checked,
            pain_chronic: document.getElementById('pain_chronic').checked
        },
        symptoms: {
            pain_during_sex: document.querySelector('input[name="pain_during_sex"]:checked')?.value === '1',
            bowel_symptoms: document.querySelector('input[name="bowel_symptoms"]:checked')?.value === '1',
            urinary_symptoms: document.querySelector('input[name="urinary_symptoms"]:checked')?.value === '1',
            fatigue: document.querySelector('input[name="fatigue"]:checked')?.value === '1',
            infertility: document.querySelector('input[name="infertility"]:checked')?.value === '1',
            other_symptoms: document.getElementById('other_symptoms').value
        },
        biomarkers: {
            ca125: document.getElementById('ca125').value ? parseFloat(document.getElementById('ca125').value) : null,
            il6: document.getElementById('il6').value ? parseFloat(document.getElementById('il6').value) : null,
            tnf_alpha: document.getElementById('tnf_alpha').value ? parseFloat(document.getElementById('tnf_alpha').value) : null,
            vegf: document.getElementById('vegf').value ? parseFloat(document.getElementById('vegf').value) : null,
            amh: document.getElementById('amh').value ? parseFloat(document.getElementById('amh').value) : null,
            crp: document.getElementById('crp').value ? parseFloat(document.getElementById('crp').value) : null,
            imaging: document.getElementById('imaging').value,
            imaging_details: document.getElementById('imaging_details').value
        },
        examination: {
            bmi: document.getElementById('bmi').value ? parseFloat(document.getElementById('bmi').value) : null,
            pelvic_exam: document.getElementById('pelvic_exam').value,
            vaginal_exam: document.getElementById('vaginal_exam').value,
            clinical_notes: document.getElementById('clinical_notes').value
        }
    };

    // Hacer la petición al backend
    fetch('https://sitme.onrender.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            // Datos para el modelo predictivo
            age: formData.personal.age,
            menarche_age: formData.menstrual.menarche_age,
            cycle_length: formData.menstrual.cycle_length,
            period_duration: formData.menstrual.period_duration,
            pain_level: formData.menstrual.pain_level,
            pain_during_sex: formData.symptoms.pain_during_sex ? 1 : 0,
            family_history: formData.history.family_endometriosis ? 1 : 0,
            bowel_symptoms: formData.symptoms.bowel_symptoms ? 1 : 0,
            urinary_symptoms: formData.symptoms.urinary_symptoms ? 1 : 0,
            fatigue: formData.symptoms.fatigue ? 1 : 0,
            infertility: formData.symptoms.infertility ? 1 : 0,
            ca125: formData.biomarkers.ca125,
            il6: formData.biomarkers.il6,
            tnf_alpha: formData.biomarkers.tnf_alpha,
            vegf: formData.biomarkers.vegf,
            amh: formData.biomarkers.amh,
            crp: formData.biomarkers.crp,
            // Datos adicionales para el reporte
            full_form_data: formData
        })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error del servidor: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Mostrar resultados
            displayResults({
                probability: data.probability,
                riskLevel: data.risk_level,
                riskTitle: getRiskTitle(data.risk_level),
                riskDescription: getRiskDescription(data.risk_level),
                riskIcon: getRiskIcon(data.risk_level),
                recommendations: data.recommendations,
                riskFactors: mapRiskFactors(data.risk_factors || [], formData),
                formData: formData,
                guidelines: getClinicalGuidelines(data.risk_level)
            });
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message);

            // Opción de simulación solo en desarrollo
            if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                if (confirm('Error al conectar con el servidor. ¿Desea ver una simulación local?')) {
                    showSimulation(formData);
                }
            }
        })
        .finally(() => {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-heartbeat"></i> Evaluar Riesgo de Endometriosis';
        });
});

// Función para mostrar resultados
function displayResults(data) {
    const resultContainer = document.getElementById('resultContainer');
    const probabilityPercent = Math.ceil(data.probability * 100);

    // Configurar el resultado principal
    document.getElementById('riskTitle').textContent = data.riskTitle;
    document.getElementById('riskDescription').textContent = data.riskDescription;
    document.getElementById('riskIcon').className = data.riskIcon;

    // Configurar el círculo de probabilidad
    const probabilityCircle = document.getElementById('probabilityCircle');
    probabilityCircle.textContent = `${probabilityPercent}%`;

    // Configurar el texto de probabilidad
    document.getElementById('probabilityText').textContent =
        `El sistema ha calculado un ${probabilityPercent}% de probabilidad de endometriosis basado en los síntomas y marcadores proporcionados.`;

    // Configurar factores de riesgo
    const riskFactorsList = document.getElementById('riskFactorsList');
    riskFactorsList.innerHTML = '';

    data.riskFactors.forEach(factor => {
        const factorElement = document.createElement('div');
        factorElement.className = 'risk-factor';
        factorElement.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${factor}`;
        riskFactorsList.appendChild(factorElement);
    });

    // Configurar recomendaciones
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';

    const recommendations = Array.isArray(data.recommendations)
        ? data.recommendations
        : (data.recommendation || '').split('\n').filter(r => r.trim() !== '');

    recommendations.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        recommendationsList.appendChild(li);
    });

    // Configurar resumen del paciente CON TOOLTIPS Y FACTORES CRÍTICOS
    const patientSummary = document.getElementById('patientSummary');
    patientSummary.innerHTML = '';

    // Mapeo de tooltips para factores críticos (rojos) y no críticos (verdes)
    const tooltipMap = {
        'Edad': (value, isCritical) => {
            const age = parseInt(value.split(' ')[0]);
            if (isCritical) return 'Edad <30 años: Mayor prevalencia de endometriosis según estudios poblacionales';
            if (age >= 40) return 'Edad ≥40 años: Menor prevalencia pero mayor probabilidad de formas avanzadas';
            return '';
        },
        'Dolor menstrual': (value, isCritical) => {
            const level = parseInt(value);
            if (isCritical) return 'Dolor severo (≥7/10) tiene alta correlación con endometriosis profunda (82% VPP)';
            if (level < 4) return 'Dolor leve (<4/10) tiene menor correlación pero no descarta endometriosis';
            return '';
        },
        'Dispareunia': (value, isCritical) => {
            if (isCritical) return 'Dolor durante relaciones sugiere implantes en ligamentos uterosacros';
            if (value === 'No') return 'Ausencia de dispareunia reduce probabilidad de endometriosis profunda';
            return '';
        },
        'Antecedentes familiares': (value, isCritical) => {
            if (isCritical) return 'Riesgo aumentado 6-9x según guías ASRM 2022';
            if (value === 'No') return 'Sin antecedentes familiares reduce riesgo relativo a ~1.5x';
            return '';
        },
        'CA-125': (value, isCritical) => {
            if (value === 'No medido') return '';
            const num = parseFloat(value.split(' ')[0]);
            if (isCritical) return 'Niveles elevados (>35 U/mL) en 72% de endometriosis estadio III-IV';
            if (num <= 20) return 'Valor normal tiene bajo valor predictivo negativo (45%)';
            return '';
        },
        'PCR': (value, isCritical) => {
            if (value === 'No medido') return '';
            const num = parseFloat(value.split(' ')[0]);
            if (isCritical) return 'Inflamación sistémica (PCR >10) asociada a progresión de enfermedad';
            if (num <= 3) return 'PCR normal sugiere ausencia de inflamación sistémica significativa';
            return '';
        },
        'Menarquia': (value, isCritical) => {
            const age = parseInt(value.split(' ')[0]);
            if (isCritical) return 'Menarquia temprana (<12 años) es factor de riesgo significativo';
            if (age >= 14) return 'Menarquia tardía (≥14 años) asociada a menor riesgo relativo';
            return '';
        },
        'Ciclo menstrual': (value, isCritical) => {
            const days = parseInt(value.split(' ')[0]);
            if (isCritical) return 'Ciclos cortos (<25 días) asociados a mayor actividad estrogénica';
            if (days >= 30) return 'Ciclos largos (≥30 días) pueden indicar anovulación';
            return '';
        },
        'Duración período': (value, isCritical) => {
            const days = parseInt(value.split(' ')[0]);
            if (isCritical) return 'Sangrado prolongado (>7 días) puede indicar adenomiosis coexistente';
            if (days <= 3) return 'Sangrado breve (≤3 días) tiene menor asociación con endometriosis';
            return '';
        },
        'IMC': (value, isCritical) => {
            if (value === 'No calculado') return '';
            const bmi = parseFloat(value);
            if (bmi < 18.5) return 'IMC bajo (<18.5) asociado a mayor riesgo de endometriosis';
            if (bmi >= 25) return 'Sobrepeso puede enmascarar síntomas pero no reduce riesgo';
            return 'IMC normal (18.5-24.9) no muestra correlación clara con endometriosis';
        },
        'Examen pélvico': (value, isCritical) => {
            if (isCritical) return 'Hallazgos anormales aumentan probabilidad de endometriosis';
            if (value === 'normal') return 'Examen normal no descarta endometriosis (especificidad 58%)';
            return '';
        }
    };

    const summaryData = [
        {
            label: 'Nombre',
            value: data.formData.personal.full_name,
            critical: false
        },
        {
            label: 'Edad',
            value: `${data.formData.personal.age} años`,
            critical: data.formData.personal.age < 30
        },
        {
            label: 'Menarquia',
            value: `${data.formData.menstrual.menarche_age} años`,
            critical: data.formData.menstrual.menarche_age < 12
        },
        {
            label: 'Ciclo menstrual',
            value: `${data.formData.menstrual.cycle_length} días`,
            critical: data.formData.menstrual.cycle_length < 25
        },
        {
            label: 'Duración período',
            value: `${data.formData.menstrual.period_duration} días`,
            critical: data.formData.menstrual.period_duration > 7
        },
        {
            label: 'Dolor menstrual',
            value: `${data.formData.menstrual.pain_level}/10`,
            critical: data.formData.menstrual.pain_level >= 7
        },
        {
            label: 'Dispareunia',
            value: data.formData.symptoms.pain_during_sex ? 'Sí' : 'No',
            critical: data.formData.symptoms.pain_during_sex
        },
        {
            label: 'Antecedentes familiares',
            value: data.formData.history.family_endometriosis ? 'Sí' : 'No',
            critical: data.formData.history.family_endometriosis
        },
        {
            label: 'CA-125',
            value: data.formData.biomarkers.ca125 !== null ? `${data.formData.biomarkers.ca125} U/mL` : 'No medido',
            critical: data.formData.biomarkers.ca125 > 35
        },
        {
            label: 'PCR',
            value: data.formData.biomarkers.crp !== null ? `${data.formData.biomarkers.crp} mg/L` : 'No medido',
            critical: data.formData.biomarkers.crp > 10
        },
        {
            label: 'IMC',
            value: data.formData.examination.bmi !== null ? data.formData.examination.bmi : 'No calculado',
            critical: false
        },
        {
            label: 'Examen pélvico',
            value: data.formData.examination.pelvic_exam || 'No registrado',
            critical: ['tenderness', 'nodules'].includes(data.formData.examination.pelvic_exam)
        }
    ];

    summaryData.forEach(item => {
        const summaryItem = document.createElement('div');
        summaryItem.className = `summary-item ${item.critical ? 'critical' : ''}`;

        const tooltip = tooltipMap[item.label] ? tooltipMap[item.label](item.value, item.critical) : '';

        summaryItem.innerHTML = `
        <div class="summary-label" ${tooltip ? `data-tooltip="${tooltip}" data-tooltip-color="${item.critical ? 'red' : 'green'}"` : ''}>
            ${item.label}
            ${tooltip ? '<i class="fas fa-info-circle tooltip-icon"></i>' : ''}
        </div>
        <div class="summary-value">${item.value}</div>
    `;
        patientSummary.appendChild(summaryItem);
    });

    // Configurar guías clínicas
    if (data.guidelines) {
        document.getElementById('asrmGuideline').textContent = data.guidelines.asrm;
        document.getElementById('eshreGuideline').textContent = data.guidelines.eshre;
        document.getElementById('niceGuideline').textContent = data.guidelines.nice;
    }

    // Mostrar el contenedor de resultados con la clase de riesgo adecuada
    resultContainer.className = `result-container ${data.riskLevel}-risk`;
    resultContainer.style.display = 'block';

    // Desplazarse a los resultados
    resultContainer.scrollIntoView({ behavior: 'smooth' });

    // Añadir botón de descarga de ficha clínica
    const downloadButton = document.createElement('button');
    downloadButton.className = 'download-button';
    downloadButton.innerHTML = '<i class="fas fa-file-pdf"></i> Descargar Ficha Clínica y Bono';
    downloadButton.onclick = () => downloadClinicalRecord(data.formData);

    const recommendationsSection = document.querySelector('.recommendations');
    if (!document.querySelector('.download-button')) {
        recommendationsSection.appendChild(downloadButton);
    }
}

// Funciones auxiliares
function getRiskTitle(riskLevel) {
    const titles = {
        'high': 'Riesgo Alto de Endometriosis',
        'moderate': 'Riesgo Moderado de Endometriosis',
        'low': 'Riesgo Bajo de Endometriosis'
    };
    return titles[riskLevel] || 'Resultado de la Evaluación';
}

function getRiskDescription(riskLevel) {
    const descriptions = {
        'high': 'Los síntomas, marcadores y hallazgos clínicos sugieren una alta probabilidad de endometriosis. Se recomienda evaluación especializada urgente.',
        'moderate': 'Presenta varios indicadores de endometriosis que justifican mayor investigación y seguimiento cercano.',
        'low': 'Los síntomas actuales no sugieren endometriosis como diagnóstico principal, pero se recomienda monitorear cualquier cambio.'
    };
    return descriptions[riskLevel] || 'Por favor consulte los resultados detallados.';
}

function getRiskIcon(riskLevel) {
    const icons = {
        'high': 'fas fa-exclamation-triangle',
        'moderate': 'fas fa-exclamation-circle',
        'low': 'fas fa-check-circle'
    };
    return icons[riskLevel] || 'fas fa-info-circle';
}

function mapRiskFactors(riskFactors, formData) {
    const factorMap = {
        'dolor_severo': `Dolor menstrual severo (nivel ${formData.menstrual.pain_level}/10)`,
        'dispareunia': 'Dolor durante relaciones sexuales',
        'ca125_elevado': formData.biomarkers.ca125 ? `CA-125 elevado (${formData.biomarkers.ca125} U/mL)` : 'Marcadores inflamatorios elevados',
        'historia_familiar': 'Antecedentes familiares de endometriosis',
        'sintomas_intestinales': 'Síntomas intestinales cíclicos',
        'sintomas_urinarios': 'Síntomas urinarios cíclicos',
        'fatiga_cronica': 'Fatiga crónica',
        'problemas_fertilidad': 'Dificultades para concebir',
        'menarquia_temprana': `Menarquia temprana (${formData.menstrual.menarche_age} años)`,
        'ciclos_cortos': `Ciclos menstruales cortos (${formData.menstrual.cycle_length} días)`,
        'sangrado_prolongado': `Sangrado menstrual prolongado (${formData.menstrual.period_duration} días)`
    };

    // Añadir factores basados en los datos del formulario
    if (formData.history.gynecological_surgery) {
        riskFactors.push('cirugias_ginecologicas');
        factorMap['cirugias_ginecologicas'] = 'Historial de cirugías ginecológicas';
    }

    if (formData.biomarkers.crp > 10) {
        riskFactors.push('pcr_elevada');
        factorMap['pcr_elevada'] = `PCR elevada (${formData.biomarkers.crp} mg/L)`;
    }

    if (formData.examination.pelvic_exam === 'tenderness' || formData.examination.pelvic_exam === 'nodules') {
        riskFactors.push('hallazgos_pelvicos');
        factorMap['hallazgos_pelvicos'] = 'Hallazgos anormales en examen pélvico';
    }

    return riskFactors.map(factor => factorMap[factor] || factor);
}

function getClinicalGuidelines(riskLevel) {
    const guidelines = {
        high: {
            asrm: "Paciente cumple criterios para evaluación laparoscópica diagnóstica según ASRM. Considerar estadificación quirúrgica.",
            eshre: "Recomendación ESHRE: Derivación a unidad especializada en endometriosis. Considerar tratamiento médico agresivo y evaluación quirúrgica.",
            nice: "Guía NICE: Paciente de alto riesgo requiere evaluación multidisciplinaria (ginecólogo, especialista en dolor, fertilidad)."
        },
        moderate: {
            asrm: "Paciente puede beneficiarse de tratamiento médico empírico según ASRM. Considerar imagenología avanzada antes de cirugía.",
            eshre: "Recomendación ESHRE: Prueba de tratamiento médico de 3-6 meses. Si no mejora, considerar evaluación quirúrgica.",
            nice: "Guía NICE: Manejo inicial con AINEs y terapia hormonal. Evaluar respuesta en 3 meses."
        },
        low: {
            asrm: "ASRM sugiere manejo conservador con seguimiento. Educación sobre síntomas de alerta.",
            eshre: "Recomendación ESHRE: Manejo sintomático. Reevaluar si síntomas progresan o cambian.",
            nice: "Guía NICE: Educación y analgesia según necesidad. Seguimiento anual o ante nuevos síntomas."
        }
    };

    return guidelines[riskLevel] || {
        asrm: "Consulte las guías ASRM más recientes para recomendaciones específicas.",
        eshre: "Ver directrices ESHRE actualizadas para el manejo de casos individuales.",
        nice: "Referir a las guías NICE completas para protocolos de tratamiento."
    };
}

function showError(message) {
    const errorContainer = document.createElement('div');
    errorContainer.className = 'error-message';
    errorContainer.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
    `;
    errorContainer.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #ffebee;
        color: #c62828;
        padding: 15px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        z-index: 1000;
    `;

    document.body.appendChild(errorContainer);

    setTimeout(() => {
        errorContainer.style.transition = 'opacity 1s';
        errorContainer.style.opacity = '0';
        setTimeout(() => errorContainer.remove(), 1000);
    }, 5000);
}

// Función de simulación solo para desarrollo
function showSimulation(formData) {
    let probability = 0.2;

    // Factores de riesgo simulados
    if (formData.menstrual.pain_level >= 7) probability += 0.25;
    if (formData.symptoms.pain_during_sex) probability += 0.15;
    if (formData.history.family_endometriosis) probability += 0.1;
    if (formData.symptoms.bowel_symptoms) probability += 0.1;
    if (formData.symptoms.urinary_symptoms) probability += 0.1;
    if (formData.symptoms.fatigue) probability += 0.05;
    if (formData.symptoms.infertility) probability += 0.05;

    // Ajustar según biomarcadores
    if (formData.biomarkers.ca125 > 35) probability += 0.1;
    if (formData.biomarkers.il6 > 5) probability += 0.05;
    if (formData.biomarkers.crp > 10) probability += 0.05;

    probability = Math.min(probability, 0.95);

    const riskLevel = probability > 0.7 ? 'high' : probability > 0.4 ? 'moderate' : 'low';

    displayResults({
        probability: probability,
        riskLevel: riskLevel,
        riskTitle: getRiskTitle(riskLevel),
        riskDescription: getRiskDescription(riskLevel),
        riskIcon: getRiskIcon(riskLevel),
        recommendation: 'Esta es una simulación local basada en patrones típicos.\nLos resultados reales requieren conexión con el servidor y evaluación médica profesional.\n\nRecomendaciones:\n1. Consulta con ginecólogo\n2. Pruebas complementarias\n3. Seguimiento estrecho',
        riskFactors: mapRiskFactors([
            ...(formData.menstrual.pain_level >= 7 ? ['dolor_severo'] : []),
            ...(formData.symptoms.pain_during_sex ? ['dispareunia'] : []),
            ...(formData.history.family_endometriosis ? ['historia_familiar'] : []),
            ...(formData.biomarkers.ca125 > 35 ? ['ca125_elevado'] : []),
            ...(formData.biomarkers.crp > 10 ? ['pcr_elevada'] : []),
            ...(formData.symptoms.bowel_symptoms ? ['sintomas_intestinales'] : [])
        ], formData),
        formData: formData,
        guidelines: getClinicalGuidelines(riskLevel)
    });
}

// Actualizar visualización del nivel de dolor
document.getElementById('pain_level').addEventListener('input', function () {
    const value = this.value;
    const min = this.min || 1;
    const max = this.max || 10;
    const normalizedValue = (value - min) / (max - min); // Esto da un valor entre 0 y 1
    const percentage = Math.round(normalizedValue * 100); // Convertir a porcentaje (0-100)
    this.style.background = `linear-gradient(to right, var(--primary) 0%, var(--primary) ${percentage}%, #e0e0e0 ${percentage}%, #e0e0e0 100%)`;
});

// Inicializar el control deslizante
document.getElementById('pain_level').dispatchEvent(new Event('input'));

function downloadClinicalRecord(formData) {
    const button = document.querySelector('.download-button') || document.createElement('button');
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-file-pdf fa-spin"></i> Generando documento...';

    fetch('https://sitme.onrender.com/generate_clinical_record', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
        .then(response => {
            if (!response.ok) throw new Error('Error al generar el documento');
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ficha_clinica_${formData.personal.full_name.replace(' ', '_')}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            showError('Error al generar el documento: ' + error.message);
        })
        .finally(() => {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-file-pdf"></i> Descargar Ficha Clínica y Bono';
        });
}