// Función para mostrar/ocultar formularios de login
function toggleAuthForms() {
    const loginForm = document.getElementById('loginFormContainer');
    const appContent = document.getElementById('mainAppContent');
    const userPanel = document.querySelector('.user-panel');
    const logo = document.querySelector('.login-logo');

    
    const isLoggedIn = localStorage.getItem('medicoToken') !== null;
    
    loginForm.style.display = isLoggedIn ? 'none' : 'block';
    appContent.style.display = isLoggedIn ? 'block' : 'none';
    userPanel.style.display = isLoggedIn ? 'flex' : 'none';
    logo.style.display = isLoggedIn ? 'none' : 'block';
    
    if (isLoggedIn) {
        loadMedicoData();
    }
}

// Función para manejar el login
async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const loginBtn = document.getElementById('loginBtn');
    const loginError = document.getElementById('loginError');
    
    // Mostrar estado de carga
    loginBtn.disabled = true;
    loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Iniciando sesión...';
    loginError.textContent = '';
    
    try {
        const response = await fetch('https://sitme-api.onrender.com/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Credenciales incorrectas. Por favor intente nuevamente.');
        }
        
        // Guardar datos en localStorage
        localStorage.setItem('medicoToken', data.token);
        localStorage.setItem('medicoData', JSON.stringify(data.user));
        
        // Actualizar UI
        updateMedicoBanner(data.user);
        toggleAuthForms();
        showSuccessNotification(`Bienvenido, ${data.user.nombre.split(' ')[0]}`);

    } catch (error) {
        console.error('Login error:', error);
        loginError.textContent = error.message;
        showErrorNotification('Error al iniciar sesión');
    } finally {
        // Restaurar botón
        loginBtn.disabled = false;
        loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Iniciar sesión';
    }
}

// Función para actualizar el panel de usuario
function updateMedicoBanner(medicoData) {
    if (!medicoData) return;
    
    // Actualizar avatar (iniciales)
    const avatar = document.getElementById('medicoAvatar');
    if (avatar) {
        const initials = medicoData.nombre.split(' ')
                            .filter(name => name.length > 0)
                            .map(name => name[0])
                            .join('')
                            .toUpperCase();
        avatar.textContent = initials.slice(0, 2);
    }
    
    // Actualizar información
    const nombreElement = document.getElementById('medicoNombre');
    const emailElement = document.getElementById('medicoEmail');
    
    if (nombreElement) nombreElement.textContent = `${medicoData.nombre}`;
    if (emailElement) emailElement.textContent = medicoData.email;
}

// Función para cargar datos del médico
function loadMedicoData() {
    const medicoData = JSON.parse(localStorage.getItem('medicoData'));
    if (medicoData) {
        updateMedicoBanner(medicoData);
    }
}

// Función para cerrar sesión
function handleLogout() {
    localStorage.removeItem('medicoToken');
    localStorage.removeItem('medicoData');
    toggleAuthForms();
    showSuccessNotification('Sesión cerrada exitosamente');
    // Opcional: Redirigir a la página de inicio
    // window.location.href = '/';
}

// Inicialización al cargar la página
document.addEventListener('DOMContentLoaded', () => {
    // Configurar event listeners
    const loginForm = document.getElementById('loginForm');
    const logoutBtn = document.getElementById('logoutBtn');
    
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
    
    // Verificar estado de autenticación
    toggleAuthForms();
});

// Funciones de notificación (ejemplo básico)
function showSuccessNotification(message) {
    console.log('Éxito:', message);
    // Aquí puedes integrar Toastify, SweetAlert, etc.
    alert(message); // Solo para ejemplo, reemplazar con tu sistema de notificaciones
}

function showErrorNotification(message) {
    console.error('Error:', message);
    alert(message); // Solo para ejemplo, reemplazar con tu sistema de notificaciones
}

// JavaScript para manejar el menú desplegable
document.addEventListener('DOMContentLoaded', function() {
    const userMenuTrigger = document.getElementById('userMenuTrigger');
    const userMenu = document.getElementById('userMenu');
    
    // Alternar menú al hacer clic
    userMenuTrigger.addEventListener('click', function(e) {
        e.stopPropagation();
        userMenu.classList.toggle('active');
        userMenuTrigger.classList.toggle('active');
    });
    
    // Cerrar menú al hacer clic fuera
    document.addEventListener('click', function() {
        userMenu.classList.remove('active');
        userMenuTrigger.classList.remove('active');
    });
    
    // Evitar que el menú se cierre al hacer clic en él
    userMenu.addEventListener('click', function(e) {
        e.stopPropagation();
    });
});

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

    const submitButton = this.querySelector('button[type="submit"], .submit-btn');
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';

    // Validar campos requeridos
    const requiredFields = ['full_name', 'birth_date', 'menarche_age', 'cycle_length', 'period_duration', 'last_period'];
    for (const field of requiredFields) {
        if (!document.getElementById(field).value) {
            alert(`Por favor complete el campo requerido: "${document.querySelector(`label[for="${field}"]`).textContent.replace(' *', '')}"`);
            button.disabled = false;
            //button.innerHTML = '<i class="fas fa-heartbeat"></i> Evaluar Riesgo de Endometriosis';
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
    fetch('https://sitme-api.onrender.com/predict', {
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
                guidelines: getClinicalGuidelines(data.probability)
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
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-heartbeat"></i> Evaluar Riesgo';
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

    // Mapeo de tooltips para factores críticos
    const tooltipMap = {
        'Edad': (value) => {
            const age = parseInt(value.split(' ')[0]);
            if (age < 30) return 'Edad <30 años: Mayor prevalencia de endometriosis según estudios poblacionales';
            return '';
        },
        'Dolor menstrual': (value) => {
            const level = parseInt(value);
            if (level >= 7) return 'Dolor severo (≥7/10) tiene alta correlación con endometriosis profunda (82% VPP)';
            if (level >= 4) return 'Dolor moderado puede indicar endometriosis temprana o adenomiosis';
            return '';
        },
        'Dolor menstrual': (value) => {
            const level = parseInt(value);
            if (level >= 7) return 'Dolor severo (≥7/10) tiene alta correlación con endometriosis profunda (82% VPP)';
            if (level >= 4) return 'Dolor moderado puede indicar endometriosis temprana o adenomiosis';
            return '';
        },
        'Dispareunia': (value) => value === 'Sí'
            ? 'Dolor durante relaciones sugiere implantes en ligamentos uterosacros'
            : '',
        'Antecedentes familiares': (value) => value === 'Sí'
            ? 'Riesgo aumentado 6-9x según guías ASRM 2022'
            : '',
        'CA-125': (value) => {
            if (value === 'No medido') return '';
            const num = parseFloat(value.split(' ')[0]);
            if (num > 35) return 'Niveles elevados (>35 U/mL) en 72% de endometriosis estadio III-IV';
            if (num > 20) return 'Valor limítrofe puede requerir seguimiento';
            return '';
        },
        'PCR': (value) => {
            if (value === 'No medido') return '';
            const num = parseFloat(value.split(' ')[0]);
            if (num > 10) return 'Inflamación sistémica (PCR >10) asociada a progresión de enfermedad';
            return '';
        },
        'Menarquia': (value) => {
            const age = parseInt(value.split(' ')[0]);
            if (age < 12) return 'Menarquia temprana (<12 años) es factor de riesgo significativo';
            return '';
        },
        'Ciclo menstrual': (value) => {
            const days = parseInt(value.split(' ')[0]);
            if (days < 25) return 'Ciclos cortos (<25 días) asociados a mayor actividad estrogénica';
            return '';
        },
        'Duración período': (value) => {
            const days = parseInt(value.split(' ')[0]);
            if (days > 7) return 'Sangrado prolongado (>7 días) puede indicar adenomiosis coexistente';
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

        const tooltip = tooltipMap[item.label] ? tooltipMap[item.label](item.value) : '';

        summaryItem.innerHTML = `
            <div class="summary-label" ${tooltip ? `data-tooltip="${tooltip}"` : ''}>
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
        document.getElementById('minsalGuideline').textContent = data.guidelines.minsal;
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
    // Añadir botón de guardar simulación
    const saveButton = document.createElement('button');
    saveButton.className = 'save-button';
    saveButton.innerHTML = '<i class="fas fa-save"></i> Guardar Simulación';
    saveButton.onclick = () => saveSimulationToDB(data);

    if (!document.querySelector('.save-button')) {
        recommendationsSection.appendChild(saveButton);
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

function getClinicalGuidelines(probability) {
    // Convertir probability a número si es necesario
    const prob = typeof probability === 'number' ? probability : parseFloat(probability);

    // Definir las guías basadas en rangos de probabilidad
    if (prob >= 0.7) { // Alto riesgo (>70%)
        return {
            asrm: "Paciente cumple criterios para evaluación laparoscópica diagnóstica según ASRM. Considerar estadificación quirúrgica.",
            eshre: "Recomendación ESHRE: Derivación a unidad especializada en endometriosis. Considerar tratamiento médico agresivo y evaluación quirúrgica.",
            nice: "Guía NICE: Paciente de alto riesgo requiere evaluación multidisciplinaria (ginecólogo, especialista en dolor, fertilidad).",
            minsal: "Guía Chilena (GES 12): Derivación urgente a especialista. Laparoscopia diagnóstica/terapéutica prioritaria (AUGE). Tratamiento hormonal postquirúrgico obligatorio."
        };
    } else if (prob >= 0.4) { // Riesgo moderado (40-69%)
        return {
            asrm: "Paciente puede beneficiarse de tratamiento médico empírico según ASRM. Considerar imagenología avanzada antes de cirugía.",
            eshre: "Recomendación ESHRE: Prueba de tratamiento médico de 3-6 meses. Si no mejora, considerar evaluación quirúrgica.",
            nice: "Guía NICE: Manejo inicial con AINEs y terapia hormonal. Evaluar respuesta en 3 meses.",
            minsal: "Guía Chilena (GES 12): Iniciar tratamiento hormonal (anticonceptivos orales/progestágenos) + AINEs. Ecografía transvaginal. Si no mejora en 6 meses, derivar a especialista."
        };
    } else { // Bajo riesgo (<40%)
        return {
            asrm: "ASRM sugiere manejo conservador con seguimiento. Educación sobre síntomas de alerta.",
            eshre: "Recomendación ESHRE: Manejo sintomático. Reevaluar si síntomas progresan o cambian.",
            nice: "Guía NICE: Educación y analgesia según necesidad. Seguimiento anual o ante nuevos síntomas.",
            minsal: "Guía Chilena (GES 12): Educación en síntomas y control anual. Analgesia con AINEs. Considerar anticonceptivos si dismenorrea. Derivar si empeora."
        };
    }
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
        guidelines: getClinicalGuidelines(probability)
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
    const button = document.querySelector('.download-button');
    if (!button) return;

    // Guardar el contenido original para restaurarlo después
    const originalContent = button.innerHTML;

    // Estado de carga
    button.disabled = true;
    button.innerHTML = `
        <div class="spinner-container">
            <div class="loading-spinner"></div>
            <span>Generando documento...</span>
        </div>
    `;

    // Añadir clase de loading para posibles estilos adicionales
    button.classList.add('loading');

    // Generar nombre de archivo
    const today = new Date().toLocaleDateString('es-CL', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    }).replace(/\//g, '-');

    const patientName = formData.personal.full_name
        .trim()
        .toLowerCase()
        .replace(/\s+/g, '_')
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-z0-9_]/g, '');

    const fileName = `ficha_${patientName}_${today}.pdf`;

    // Solicitud de generación del PDF
    fetch('https://sitme-api.onrender.com/generate_clinical_record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
        .then(response => {
            if (!response.ok) throw new Error(`Error ${response.status}: ${response.statusText}`);
            return response.blob();
        })
        .then(blob => {
            // Descarga del archivo
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();

            // Limpieza
            setTimeout(() => {
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }, 100);
        })
        .catch(error => {
            console.error('Error:', error);
            showError(`Error al generar el documento: ${error.message}`);
        })
        .finally(() => {
            // Restaurar estado normal
            button.disabled = false;
            button.innerHTML = originalContent;
            button.classList.remove('loading');
        });
}

function saveSimulationToDB(simulationData) {
    const saveButton = document.querySelector('.save-button');
    if (!saveButton) return;

    // Guardar el contenido original para restaurarlo después
    const originalContent = saveButton.innerHTML;

    // Estado de carga
    saveButton.disabled = true;
    saveButton.innerHTML = `
        <div class="spinner-container">
            <div class="loading-spinner"></div>
            <span>Guardando...</span>
        </div>
    `;

    // Recoger TODOS los datos del formulario
    const formData = {
        personal: {
            full_name: document.getElementById('full_name').value,
            id_number: document.getElementById('rut').value,
            birth_date: document.getElementById('birth_date').value,
            age: document.getElementById('age').value,
            blood_type: document.getElementById('blood_type').value,
            insurance: document.querySelector('input[name="insurance"]:checked')?.value
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
            pain_level: document.getElementById('pain_level').value,
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
            height: document.getElementById('height').value ? parseFloat(document.getElementById('height').value) : null,
            weight: document.getElementById('weight').value ? parseFloat(document.getElementById('weight').value) : null,
            bmi: document.getElementById('bmi').value ? parseFloat(document.getElementById('bmi').value) : null,
            pelvic_exam: document.getElementById('pelvic_exam').value,
            vaginal_exam: document.getElementById('vaginal_exam').value,
            clinical_notes: document.getElementById('clinical_notes').value
        }
    };

    // Preparamos los datos para enviar
    const dataToSend = {
        form_data: formData,
        prediction: {
            probability: simulationData.probability,
            risk_level: simulationData.riskLevel || simulationData.risk_level,
            recommendations: simulationData.recommendations,
            model_version: simulationData.model_info?.version || 'v4.1'
        },
        clinic_id: document.getElementById('clinicId').textContent
    };

    console.log('Data enviada:', dataToSend);

    // Solicitud para guardar la simulación
    fetch('https://sitme-api.onrender.com/save_simulation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw new Error(`Error ${response.status}: ${errData.error || response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) throw new Error(data.error);

            // Mostrar notificación de éxito
            showSuccessNotification('Simulación guardada exitosamente en la base de datos');
        })
        .catch(error => {
            console.error('Error:', error);
            showError(`Error al guardar: ${error.message}`);
        })
        .finally(() => {
            // Restaurar estado normal
            saveButton.disabled = false;
            saveButton.innerHTML = originalContent;
        });
}

// Agrega esta función al script.js para cargar el historial
function loadHistoryData(searchTerm = '') {
    const historyTableBody = document.getElementById('historyTableBody');
    const noHistoryMsg = document.getElementById('noHistoryMessage');

    // Mostrar estado de carga
    historyTableBody.innerHTML = '<tr><td colspan="7" class="loading-row"><div class="spinner-container"><div class="loading-spinner"></div><span>Cargando historial...</span></div></td></tr>';

    // Construir URL con parámetro de búsqueda si existe
    let url = 'https://sitme-api.onrender.com/get_history';
    if (searchTerm) {
        url += `?search=${encodeURIComponent(searchTerm)}`;
    }

    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error('Error al cargar el historial');
            return response.json();
        })
        .then(data => {
            if (!data.records || data.records.length === 0) {
                historyTableBody.innerHTML = '';
                noHistoryMsg.style.display = 'block';
                return;
            }

            historyTableBody.innerHTML = '';
            noHistoryMsg.style.display = 'none';

            data.records.forEach(record => {
                const row = document.createElement('tr');

                // Determinar clase de riesgo basada en la probabilidad
                const probability = Math.round((record.probability || 0) * 100);
                let riskClass = '';
                let riskText = '';

                if (probability >= 70) {
                    riskClass = 'high-risk';
                    riskText = 'Alto';
                } else if (probability >= 40) {
                    riskClass = 'moderate-risk';
                    riskText = 'Moderado';
                } else {
                    riskClass = 'low-risk';
                    riskText = 'Bajo';
                }

                row.innerHTML = `
                    <td>${record.clinic_id || 'ENDO-' + record.id.toString().padStart(4, '0')}</td>
                    <td>${record.full_name || 'No registrado'}</td>
                    <td>${record.rut || 'No registrado'}</td>
                    <td>${record.age || 'N/A'}</td>
                    <td>${record.evaluation_date || 'N/A'}</td>
                    <td class="${riskClass}">${riskText}</td>
                    <td class="history-actions">
                        <button class="history-btn view-btn" data-id="${record.id}">
                            <i class="fas fa-eye"></i> Ver
                        </button>
                        <button class="history-btn delete-btn" data-id="${record.id}">
                            <i class="fas fa-trash"></i> Eliminar
                        </button>
                    </td>
                `;

                historyTableBody.appendChild(row);
            });

            // Agregar eventos a los botones
            addHistoryButtonEvents();
        })
        .catch(error => {
            console.error('Error:', error);
            historyTableBody.innerHTML = '<tr><td colspan="7" class="error-row">Error al cargar el historial</td></tr>';
        });
}

let currentPage = 1;
const recordsPerPage = 10;
let totalRecords = 0;

function loadHistoryData(searchTerm = '', page = 1) {
    const historyTableBody = document.getElementById('historyTableBody');
    const noHistoryMsg = document.getElementById('noHistoryMessage');
    const paginationContainer = document.getElementById('paginationContainer');

    // Mostrar estado de carga
    historyTableBody.innerHTML = '<tr><td colspan="7" class="loading-row"><div class="spinner-container"><div class="loading-spinner"></div><span>Cargando historial...</span></div></td></tr>';
    paginationContainer.innerHTML = '';

    // Construir URL con parámetros de búsqueda y paginación
    let url = `https://sitme-api.onrender.com/get_history?page=${page}&limit=${recordsPerPage}`;
    if (searchTerm) {
        url += `&search=${encodeURIComponent(searchTerm)}`;
    }

    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error('Error al cargar el historial');
            return response.json();
        })
        .then(data => {
            if (!data.records || data.records.length === 0) {
                historyTableBody.innerHTML = '';
                noHistoryMsg.style.display = 'block';
                paginationContainer.innerHTML = '';
                return;
            }

            historyTableBody.innerHTML = '';
            noHistoryMsg.style.display = 'none';
            totalRecords = data.total || data.records.length;

            data.records.forEach(record => {
                const row = document.createElement('tr');

                // Determinar clase de riesgo basada en la probabilidad
                const probability = Math.round((record.probability || 0) * 100);
                let riskClass = '';
                let riskText = '';

                if (probability >= 70) {
                    riskClass = 'high-risk';
                    riskText = 'Alto';
                } else if (probability >= 40) {
                    riskClass = 'moderate-risk';
                    riskText = 'Moderado';
                } else {
                    riskClass = 'low-risk';
                    riskText = 'Bajo';
                }

                row.innerHTML = `
                    <td>${record.clinic_id || 'ENDO-' + record.id.toString().padStart(4, '0')}</td>
                    <td>${record.full_name || 'No registrado'}</td>
                    <td>${record.rut || 'No registrado'}</td>
                    <td>${record.age || 'N/A'}</td>
                    <td>${record.evaluation_date || 'N/A'}</td>
                    <td class="${riskClass}">${riskText}</td>
                    <td class="history-actions">
                        <button class="history-btn view-btn" data-id="${record.id}">
                            <i class="fas fa-eye"></i> Ver
                        </button>
                        <button class="history-btn delete-btn" data-id="${record.id}">
                            <i class="fas fa-trash"></i> Eliminar
                        </button>
                    </td>
                `;

                historyTableBody.appendChild(row);
            });

            // Agregar eventos a los botones
            addHistoryButtonEvents();

            // Mostrar paginación si hay más de una página
            if (totalRecords > recordsPerPage) {
                renderPagination(totalRecords, page, searchTerm);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            historyTableBody.innerHTML = '<tr><td colspan="7" class="error-row">Error al cargar el historial</td></tr>';
        });
}

function renderPagination(totalRecords, currentPage, searchTerm) {
    const paginationContainer = document.getElementById('paginationContainer');
    const totalPages = Math.ceil(totalRecords / recordsPerPage);
    
    paginationContainer.innerHTML = '';
    
    if (totalPages <= 1) return;

    const pagination = document.createElement('div');
    pagination.className = 'pagination';

    // Botón Anterior
    if (currentPage > 1) {
        const prevBtn = document.createElement('button');
        prevBtn.className = 'pagination-btn';
        prevBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
        prevBtn.addEventListener('click', () => {
            loadHistoryData(searchTerm, currentPage - 1);
        });
        pagination.appendChild(prevBtn);
    }

    // Números de página
    const maxVisiblePages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

    if (endPage - startPage + 1 < maxVisiblePages) {
        startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    if (startPage > 1) {
        const firstPageBtn = document.createElement('button');
        firstPageBtn.className = 'pagination-btn';
        firstPageBtn.textContent = '1';
        firstPageBtn.addEventListener('click', () => {
            loadHistoryData(searchTerm, 1);
        });
        pagination.appendChild(firstPageBtn);

        if (startPage > 2) {
            const ellipsis = document.createElement('span');
            ellipsis.className = 'pagination-ellipsis';
            ellipsis.textContent = '...';
            pagination.appendChild(ellipsis);
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        const pageBtn = document.createElement('button');
        pageBtn.className = `pagination-btn ${i === currentPage ? 'active' : ''}`;
        pageBtn.textContent = i;
        pageBtn.addEventListener('click', () => {
            loadHistoryData(searchTerm, i);
        });
        pagination.appendChild(pageBtn);
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const ellipsis = document.createElement('span');
            ellipsis.className = 'pagination-ellipsis';
            ellipsis.textContent = '...';
            pagination.appendChild(ellipsis);
        }

        const lastPageBtn = document.createElement('button');
        lastPageBtn.className = 'pagination-btn';
        lastPageBtn.textContent = totalPages;
        lastPageBtn.addEventListener('click', () => {
            loadHistoryData(searchTerm, totalPages);
        });
        pagination.appendChild(lastPageBtn);
    }

    // Botón Siguiente
    if (currentPage < totalPages) {
        const nextBtn = document.createElement('button');
        nextBtn.className = 'pagination-btn';
        nextBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
        nextBtn.addEventListener('click', () => {
            loadHistoryData(searchTerm, currentPage + 1);
        });
        pagination.appendChild(nextBtn);
    }

    // Contador de registros
    const startRecord = (currentPage - 1) * recordsPerPage + 1;
    const endRecord = Math.min(currentPage * recordsPerPage, totalRecords);
    const counter = document.createElement('div');
    counter.className = 'pagination-counter';
    counter.textContent = `Mostrando ${startRecord}-${endRecord} de ${totalRecords} registros`;
    
    paginationContainer.appendChild(counter);
    paginationContainer.appendChild(pagination);
}

// Función para agregar eventos a los botones de la tabla
function addHistoryButtonEvents() {
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const recordId = this.getAttribute('data-id');
            viewRecordDetails(recordId);
        });
    });

    document.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const recordId = this.getAttribute('data-id');
            deleteRecord(recordId);
        });
    });
}

// Función para ver detalles de un registro
function viewRecordDetails(recordId) {
    fetch(`https://sitme-api.onrender.com/get_record_details/${recordId}`)
        .then(response => {
            if (!response.ok) throw new Error('Error al cargar detalles');
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showPatientModal(data.record);
            } else {
                throw new Error(data.error || 'Error desconocido');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('No se pudieron cargar los detalles del paciente');
        });
}

// Inicializar eventos del modal cuando se cargue el DOM
document.addEventListener('DOMContentLoaded', setupModalEvents);

// Función para eliminar un registro
function deleteRecord(recordId) {
    if (confirm('¿Está seguro que desea eliminar este registro permanentemente?')) {
        fetch(`https://sitme-api.onrender.com/delete_record/${recordId}`, {
            method: 'DELETE'
        })
            .then(response => {
                if (!response.ok) throw new Error('Error al eliminar registro');
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    showSuccessNotification('Registro eliminado correctamente');
                    loadHistoryData(); // Recargar la tabla
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showError('No se pudo eliminar el registro');
            });
    }
}

// Eventos de búsqueda mejorados
function setupSearchEvents() {
    const performSearch = () => {
        const searchInput = document.getElementById('historySearch');
        const searchTerm = searchInput ? searchInput.value.trim() : '';
        currentPage = 1; // Reiniciar a la primera página
        loadHistoryData(searchTerm, currentPage);
    };

    // Botón de búsqueda
    const searchBtn = document.getElementById('searchHistoryBtn');
    if (searchBtn) {
        searchBtn.addEventListener('click', performSearch);
    }

    // Buscar al presionar Enter
    const searchInput = document.getElementById('historySearch');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
}

// Cargar datos cuando se muestra la pestaña
document.addEventListener('DOMContentLoaded', function () {
    // Cargar historial si estamos en esa pestaña
    if (document.querySelector('#main-tab-history').classList.contains('active')) {
        loadHistoryData();
    }

    // Evento para cambiar entre pestañas
    document.querySelectorAll('.main-tab-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            if (this.getAttribute('data-tab') === 'main-tab-history' &&
                !this.classList.contains('active')) {
                loadHistoryData();
            }
        });
    });
});

// Agrega esto al final del DOMContentLoaded en script.js
document.addEventListener('DOMContentLoaded', function () {
    // ... código existente ...

    // Cargar historial cuando se muestre la pestaña
    document.querySelector('.main-tab-btn[data-tab="main-tab-history"]').addEventListener('click', function () {
        if (!this.classList.contains('active')) {
            loadHistoryData();
        }
    });

    // También cargar al inicio si estamos en la pestaña de historial
    if (document.querySelector('#main-tab-history').classList.contains('active')) {
        loadHistoryData();
    }
});

// Función para mostrar notificación de éxito
function showSuccessNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'success-notification';
    notification.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 15px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        z-index: 1000;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.transition = 'opacity 1s';
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 1000);
    }, 5000);
}

document.addEventListener('DOMContentLoaded', function () {
    // Manejar pestañas principales
    const mainTabButtons = document.querySelectorAll('.main-tab-btn');
    const mainTabContents = document.querySelectorAll('.main-tab-content');

    mainTabButtons.forEach(button => {
        button.addEventListener('click', function () {
            const tabId = this.getAttribute('data-tab');

            // Ocultar todos los contenidos de pestaña principal
            mainTabContents.forEach(content => {
                content.classList.remove('active');
            });

            // Mostrar el contenido de la pestaña principal seleccionada
            document.getElementById(tabId).classList.add('active');

            // Actualizar botones activos
            mainTabButtons.forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    // Manejar pestañas del formulario
    const formTabButtons = document.querySelectorAll('.form-tab-btn');
    const formTabContents = document.querySelectorAll('.form-tab-content');
    const nextButtons = document.querySelectorAll('.nav-btn.next');
    const prevButtons = document.querySelectorAll('.nav-btn.prev');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    // Manejar clic en las pestañas del formulario
    formTabButtons.forEach(button => {
        button.addEventListener('click', function () {
            const tabId = this.getAttribute('data-tab');

            // Ocultar todos los contenidos de pestaña
            formTabContents.forEach(content => {
                content.classList.remove('active');
            });

            // Mostrar el contenido de la pestaña seleccionada
            document.getElementById(tabId).classList.add('active');

            // Actualizar botones activos
            formTabButtons.forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');

            // Actualizar barra de progreso
            updateProgressBar(tabId);
        });
    });

    // Manejar botones de siguiente
    nextButtons.forEach(button => {
        button.addEventListener('click', function () {
            if (this.classList.contains('submit-btn')) return;

            const nextTabId = this.getAttribute('data-next');
            const currentTab = this.closest('.form-tab-content').id;

            // Validar campos requeridos antes de avanzar
            if (!validateTab(currentTab)) {
                return;
            }

            // Cambiar a la siguiente pestaña
            document.querySelector(`.form-tab-btn[data-tab="${nextTabId}"]`).click();

            // Desplazar hacia arriba para mejor experiencia de usuario
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });

    // Manejar botones de anterior
    prevButtons.forEach(button => {
        button.addEventListener('click', function () {
            if (this.disabled) return;

            const prevTabId = this.getAttribute('data-prev');

            // Cambiar a la pestaña anterior
            document.querySelector(`.form-tab-btn[data-tab="${prevTabId}"]`).click();

            // Desplazar hacia arriba para mejor experiencia de usuario
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });

    // Función para validar campos requeridos en la pestaña actual
    function validateTab(tabId) {
        const tab = document.getElementById(tabId);
        const requiredInputs = tab.querySelectorAll('[required]');
        let isValid = true;

        requiredInputs.forEach(input => {
            if (!input.value) {
                input.classList.add('error');
                isValid = false;

                // Mostrar mensaje de error
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Este campo es obligatorio';
                errorMsg.style.color = 'red';
                errorMsg.style.fontSize = '0.8em';

                // Insertar después del campo si no existe ya
                if (!input.nextElementSibling || !input.nextElementSibling.classList.contains('error-message')) {
                    input.insertAdjacentElement('afterend', errorMsg);
                }
            } else {
                input.classList.remove('error');
                if (input.nextElementSibling && input.nextElementSibling.classList.contains('error-message')) {
                    input.nextElementSibling.remove();
                }
            }
        });

        if (!isValid) {
            // Desplazar al primer error
            const firstError = tab.querySelector('.error');
            if (firstError) {
                firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        return isValid;
    }

    // Función para actualizar la barra de progreso
    function updateProgressBar(tabId) {
        const tabIndex = Array.from(formTabContents).findIndex(tab => tab.id === tabId);
        const progress = ((tabIndex + 1) / formTabContents.length) * 100;

        progressBar.style.width = `${progress}%`;
        progressText.textContent = `Paso ${tabIndex + 1} de ${formTabContents.length}`;
    }

    // Inicializar barra de progreso
    updateProgressBar('tab1');

    // Manejar búsqueda en historial
    const searchInput = document.querySelector('.search-input');
    const searchBtn = document.querySelector('.search-btn');
    const historyTable = document.querySelector('.history-table');
    const noHistoryMsg = document.querySelector('.no-history');

    searchBtn.addEventListener('click', function () {
        const searchTerm = searchInput.value.toLowerCase();
        const rows = historyTable.querySelectorAll('tbody tr');
        let hasResults = false;

        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            let rowMatches = false;

            cells.forEach(cell => {
                if (cell.textContent.toLowerCase().includes(searchTerm)) {
                    rowMatches = true;
                }
            });

            if (rowMatches) {
                row.style.display = '';
                hasResults = true;
            } else {
                row.style.display = 'none';
            }
        });

        // Mostrar mensaje si no hay resultados
        if (!hasResults) {
            historyTable.style.display = 'none';
            noHistoryMsg.style.display = 'block';
        } else {
            historyTable.style.display = 'table';
            noHistoryMsg.style.display = 'none';
        }
    });

    // Manejar botones de acciones en historial
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            alert('Visualizando registro del paciente');
            // Aquí iría la lógica para mostrar los detalles completos del paciente
        });
    });

    document.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            if (confirm('¿Está seguro que desea eliminar este registro?')) {
                const row = this.closest('tr');
                row.remove();

                // Verificar si quedan registros
                const remainingRows = historyTable.querySelectorAll('tbody tr');
                if (remainingRows.length === 0) {
                    historyTable.style.display = 'none';
                    noHistoryMsg.style.display = 'block';
                }
            }
        });
    });
});

// Función para limpiar solo la pestaña actual
function clearCurrentTab() {
    const currentTab = document.querySelector('.form-tab-content.active');

    if (currentTab && confirm('¿Estás seguro que deseas limpiar todos los campos de esta pestaña?')) {
        const formElements = currentTab.querySelectorAll('input, select, textarea');

        formElements.forEach(element => {
            // No limpiar botones
            if (element.type !== 'button' && element.type !== 'submit') {
                element.value = '';

                // Para checkboxes y radios
                if (element.type === 'checkbox' || element.type === 'radio') {
                    element.checked = false;
                }

                // Para selects
                if (element.tagName === 'SELECT') {
                    element.selectedIndex = 0;
                }
            }
        });

        console.log(`Pestaña ${currentTab.id} limpiada`);
    }
}

// Asignar evento a los botones de limpiar
document.addEventListener('DOMContentLoaded', function () {
    const clearButtons = document.querySelectorAll('.clear-tab-btn');

    clearButtons.forEach(button => {
        button.addEventListener('click', clearCurrentTab);
    });
});

// Función para mostrar el modal con los detalles del paciente
function showPatientModal(patientData) {
    const modal = document.getElementById('patientModal');
    const modalPatientName = document.getElementById('modalPatientName');

    // Establecer nombre del paciente
    modalPatientName.textContent = patientData.full_name || 'Paciente sin nombre';

    // Llenar información básica
    fillPersonalInfo(patientData);
    fillMenstrualInfo(patientData);

    // Llenar historial médico
    fillMedicalHistory(patientData);
    fillMedicationsInfo(patientData);

    // Llenar resultados
    fillExamResults(patientData);
    fillBiomarkersInfo(patientData);
    fillImagingInfo(patientData);

    // Manejar el botón de descarga
    const downloadBtn = document.getElementById('downloadPdfBtn');
    if (downloadBtn) {
        // Si el botón ya existe, solo actualizamos su manejador de eventos
        downloadBtn.onclick = null; // Eliminar manejador anterior
        downloadBtn.onclick = () => generatePatientPDF(patientData);
    } else {
        // Si no existe, lo creamos
        const newDownloadBtn = document.createElement('button');
        newDownloadBtn.id = 'downloadPdfBtn';
        newDownloadBtn.className = 'modal-btn download-btn';
        newDownloadBtn.innerHTML = '<i class="fas fa-file-pdf"></i> Descargar PDF';
        newDownloadBtn.onclick = () => generatePatientPDF(patientData);
        
        const modalFooter = document.querySelector('.modal-footer');
        modalFooter.insertBefore(newDownloadBtn, modalFooter.querySelector('.close-btn'));
    }

    // Mostrar modal
    modal.style.display = 'block';

    // Configurar eventos de las pestañas
    setupModalTabs();
}

function generatePatientPDF(patientData) {
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        let yPosition = 20; // Posición vertical inicial

        // Configuración inicial
        doc.setFont('helvetica');
        doc.setFontSize(10);

        // ========== ENCABEZADO ==========
        doc.setFontSize(18);
        doc.setTextColor(33, 150, 243); // Azul
        doc.text('FICHA CLÍNICA - ENDOMETRIOSIS', 105, yPosition, { align: 'center' });
        yPosition += 10;

        doc.setFontSize(10);
        doc.setTextColor(100);
        doc.text(`ID Clínico: ${patientData.clinic_id || 'N/A'} | Generado: ${new Date().toLocaleDateString('es-CL')}`, 105, yPosition, { align: 'center' });
        yPosition += 15;

        // Línea divisoria
        doc.setDrawColor(200);
        doc.line(15, yPosition, 195, yPosition);
        yPosition += 10;

        // ========== 1. DATOS PERSONALES ==========
        doc.setFontSize(14);
        doc.setTextColor(33, 150, 243);
        doc.text('1. DATOS PERSONALES', 20, yPosition);
        yPosition += 8;

        const personalData = [
            ['Nombre completo:', patientData.full_name || 'No registrado'],
            ['RUT:', patientData.id_number || 'No registrado'],
            ['Fecha nacimiento:', patientData.birth_date || 'No registrado'],
            ['Edad:', patientData.age ? `${patientData.age} años` : 'N/A'],
            ['Tipo sangre:', patientData.blood_type || 'No registrado'],
            ['Previsión:', patientData.insurance || 'No registrado']
        ];

        doc.autoTable({
            startY: yPosition,
            head: false,
            body: personalData,
            columnStyles: {
                0: { fontStyle: 'bold', cellWidth: 50, textColor: [40, 40, 40] },
                1: { cellWidth: 'auto', textColor: [80, 80, 80] }
            },
            margin: { left: 20 },
            tableWidth: 170,
            styles: { fontSize: 10, cellPadding: 3 }
        });
        yPosition = doc.autoTable.previous.finalY + 10;

        // ========== 2. HISTORIAL MENSTRUAL ==========
        doc.setFontSize(14);
        doc.text('2. HISTORIAL MENSTRUAL', 20, yPosition);
        yPosition += 8;

        const menstrualData = [
            ['Edad menarquia:', patientData.menarche_age ? `${patientData.menarche_age} años` : 'N/A'],
            ['Duración ciclo:', patientData.cycle_length ? `${patientData.cycle_length} días` : 'N/A'],
            ['Duración período:', patientData.period_duration ? `${patientData.period_duration} días` : 'N/A'],
            ['Última menstruación:', patientData.last_period || 'N/A'],
            ['Nivel dolor:', patientData.pain_level ? `${patientData.pain_level}/10` : 'N/A'],
            ['Dolor premenstrual:', patientData.pain_premenstrual ? 'Sí' : 'No'],
            ['Dolor menstrual:', patientData.pain_menstrual ? 'Sí' : 'No'],
            ['Dolor ovulación:', patientData.pain_ovulation ? 'Sí' : 'No'],
            ['Dolor crónico:', patientData.pain_chronic ? 'Sí' : 'No']
        ];

        doc.autoTable({
            startY: yPosition,
            head: false,
            body: menstrualData,
            columnStyles: {
                0: { fontStyle: 'bold', cellWidth: 55 },
                1: { cellWidth: 'auto' }
            },
            margin: { left: 20 },
            tableWidth: 170
        });
        yPosition = doc.autoTable.previous.finalY + 10;

        // ========== 3. HISTORIAL MÉDICO ==========
        doc.setFontSize(14);
        doc.text('3. HISTORIAL MÉDICO', 20, yPosition);
        yPosition += 8;

        const medicalHistory = [
            ['Cirugías ginecológicas:', patientData.gynecological_surgery ? 'Sí' : 'No'],
            ['Enf. inflamatoria pélvica:', patientData.pelvic_inflammatory ? 'Sí' : 'No'],
            ['Quistes ováricos:', patientData.ovarian_cysts ? 'Sí' : 'No'],
            ['Endometriosis familiar:', patientData.family_endometriosis ? 'Sí' : 'No'],
            ['Enf. autoinmunes familiar:', patientData.family_autoimmune ? 'Sí' : 'No'],
            ['Cáncer familiar:', patientData.family_cancer ? 'Sí' : 'No'],
            ['Enf. autoinmune:', patientData.comorbidity_autoimmune ? 'Sí' : 'No'],
            ['Trastorno tiroideo:', patientData.comorbidity_thyroid ? 'Sí' : 'No'],
            ['Síndrome intestino irritable:', patientData.comorbidity_ibs ? 'Sí' : 'No']
        ];

        doc.autoTable({
            startY: yPosition,
            head: false,
            body: medicalHistory,
            columnStyles: {
                0: { fontStyle: 'bold', cellWidth: 70 },
                1: { cellWidth: 'auto' }
            },
            margin: { left: 20 },
            tableWidth: 170
        });
        yPosition = doc.autoTable.previous.finalY + 10;

        // ========== 4. SÍNTOMAS ==========
        doc.setFontSize(14);
        doc.text('4. SÍNTOMAS', 20, yPosition);
        yPosition += 8;

        const symptomsData = [
            ['Dolor durante relaciones:', patientData.symptoms?.pain_during_sex ? 'Sí' : 'No'],
            ['Síntomas intestinales:', patientData.symptoms?.bowel_symptoms ? 'Sí' : 'No'],
            ['Síntomas urinarios:', patientData.symptoms?.urinary_symptoms ? 'Sí' : 'No'],
            ['Fatiga crónica:', patientData.symptoms?.fatigue ? 'Sí' : 'No'],
            ['Problemas fertilidad:', patientData.symptoms?.infertility ? 'Sí' : 'No'],
            ['Otros síntomas:', patientData.symptoms?.other_symptoms || 'Ninguno']
        ];

        doc.autoTable({
            startY: yPosition,
            head: false,
            body: symptomsData,
            columnStyles: {
                0: { fontStyle: 'bold', cellWidth: 60 },
                1: { cellWidth: 'auto' }
            },
            margin: { left: 20 },
            tableWidth: 170
        });
        yPosition = doc.autoTable.previous.finalY + 10;

        console.log("Biomarcadores directos:", patientData.biomarkers);
        // ========== 5. BIOMARCADORES ==========
        doc.setFontSize(14);
        doc.text('5. BIOMARCADORES', 20, yPosition);
        yPosition += 8;

        // Acceder a los biomarcadores correctamente
        const biomarkers = patientData.biomarkers || {};
        
        // Función para formatear valores de biomarcadores
        const formatBiomarker = (value, unit) => {
            if (value === null || value === undefined || value === '') return 'No medido';
            return `${value} ${unit}`;
        };

        // Versión resiliente que busca en múltiples ubicaciones
        const getBiomarkerValue = (field) => {
            return patientData.biomarkers?.[field] || 
                patientData.formData?.biomarkers?.[field] || 
                patientData[field] || 
                null;
        };

        const biomarkersData = [
            ['CA-125:', formatBiomarker(getBiomarkerValue('ca125'), 'U/mL')],
            ['IL-6:', formatBiomarker(getBiomarkerValue('il6'), 'pg/mL')],
            ['TNF-α:', formatBiomarker(getBiomarkerValue('tnf_alpha'), 'pg/mL')],
            ['VEGF:', formatBiomarker(getBiomarkerValue('vegf'), 'pg/mL')],
            ['AMH:', formatBiomarker(getBiomarkerValue('amh'), 'ng/mL')],
            ['PCR:', formatBiomarker(getBiomarkerValue('crp'), 'mg/L')]
        ];

        doc.autoTable({
            startY: yPosition,
            head: false,
            body: biomarkersData,
            columnStyles: { 
                0: { fontStyle: 'bold', cellWidth: 40 },
                1: { cellWidth: 'auto' } 
            },
            margin: { left: 20 },
            tableWidth: 170
        });
        yPosition = doc.autoTable.previous.finalY + 10;

        // ========== 6. EXAMEN FÍSICO ==========
        doc.setFontSize(14);
        doc.text('6. EXAMEN FÍSICO', 20, yPosition);
        yPosition += 8;

        const examData = [
            ['Estatura:', patientData.height ? `${patientData.height} cm` : 'No registrado'],
            ['Peso:', patientData.weight ? `${patientData.weight} kg` : 'No registrado'],
            ['IMC:', patientData.bmi || 'No calculado'],
            ['Examen pélvico:', patientData.pelvic_exam || 'No realizado'],
            ['Examen vaginal:', patientData.vaginal_exam || 'No realizado'],
            ['Notas clínicas:', patientData.clinical_notes || 'Sin notas']
        ];

        doc.autoTable({
            startY: yPosition,
            head: false,
            body: examData,
            columnStyles: {
                0: { fontStyle: 'bold', cellWidth: 45 },
                1: { cellWidth: 'auto' }
            },
            margin: { left: 20 },
            tableWidth: 170
        });
        yPosition = doc.autoTable.previous.finalY + 10;

        // ========== 7. IMAGENOLOGÍA ==========
        if (patientData.imaging || patientData.imaging_details) {
            doc.setFontSize(14);
            doc.text('7. IMAGENOLOGÍA', 20, yPosition);
            yPosition += 8;

            const imagingData = [
                ['Resultados:', patientData.imaging || 'No realizado'],
                ['Detalles:', patientData.imaging_details || 'Sin detalles']
            ];

            doc.autoTable({
                startY: yPosition,
                head: false,
                body: imagingData,
                columnStyles: {
                    0: { fontStyle: 'bold', cellWidth: 40 },
                    1: { cellWidth: 'auto' }
                },
                margin: { left: 20 },
                tableWidth: 170
            });
            yPosition = doc.autoTable.previous.finalY + 10;
        }

        // ========== 8. MEDICAMENTOS ==========
        if (patientData.medications) {
            doc.setFontSize(14);
            doc.text('8. MEDICAMENTOS', 20, yPosition);
            yPosition += 8;

            // Dividir el texto en líneas para que no se salga del PDF
            const splitText = doc.splitTextToSize(patientData.medications, 170);

            doc.autoTable({
                startY: yPosition,
                head: false,
                body: [[splitText]],
                margin: { left: 20 },
                tableWidth: 170,
                styles: { cellPadding: 4 }
            });
            yPosition = doc.autoTable.previous.finalY + 10;
        }

        // ========== PIE DE PÁGINA ==========
        doc.setFontSize(10);
        doc.setTextColor(100);
        doc.text('Documento generado automáticamente por el sistema SITME - Endometriosis Toolkit', 105, 285, { align: 'center' });
        doc.text('www.sitme.cl - © ' + new Date().getFullYear(), 105, 290, { align: 'center' });

        // Guardar el PDF
        const fileName = `Ficha_Endometriosis_${(patientData.full_name || 'Paciente').replace(/[^a-z0-9]/gi, '_')}_${new Date().toISOString().slice(0, 10)}.pdf`;
        doc.save(fileName);

    } catch (error) {
        console.error('Error al generar PDF:', error);
        showError(`Error al generar PDF: ${error.message}`);
    }
}

// Funciones para llenar las secciones del modal
function fillPersonalInfo(data) {
    const container = document.getElementById('personalInfo');
    container.innerHTML = '';

    const personalData = [
        { label: 'RUT', value: data.id_number, icon: 'fa-id-card' },
        { label: 'Fecha de Nacimiento', value: data.birth_date, icon: 'fa-birthday-cake' },
        { label: 'Edad', value: data.age ? `${data.age} años` : null, icon: 'fa-user' },
        { label: 'Tipo de Sangre', value: data.blood_type, icon: 'fa-tint' },
        { label: 'Previsión', value: data.insurance, icon: 'fa-hospital' },
        { label: 'ID Clínico', value: data.clinic_id, icon: 'fa-clipboard-list' }
    ];

    personalData.forEach(item => {
        const itemElement = createModalItem(item.label, item.value, item.icon);
        container.appendChild(itemElement);
    });
}

function fillMenstrualInfo(data) {
    const container = document.getElementById('menstrualInfo');
    container.innerHTML = '';

    const menstrualData = [
        { label: 'Edad de Menarquia', value: data.menarche_age ? `${data.menarche_age} años` : null, icon: 'fa-calendar' },
        { label: 'Duración del Ciclo', value: data.cycle_length ? `${data.cycle_length} días` : null, icon: 'fa-calendar-week' },
        { label: 'Duración del Período', value: data.period_duration ? `${data.period_duration} días` : null, icon: 'fa-calendar-day' },
        { label: 'Última Menstruación', value: data.last_period, icon: 'fa-calendar-check' },
        { label: 'Dolor Menstrual', value: data.pain_level ? `${data.pain_level}/10` : null, icon: 'fa-pain' },
        { label: 'Dolor Crónico', value: data.pain_chronic, icon: 'fa-head-side-mask', isBoolean: true }
    ];

    menstrualData.forEach(item => {
        const itemElement = createModalItem(item.label, item.value, item.icon, item.isBoolean);
        container.appendChild(itemElement);
    });
}

function fillMedicalHistory(data) {
    const container = document.getElementById('medicalHistory');
    container.innerHTML = '';

    const medicalData = [
        { label: 'Cirugías Ginecológicas', value: data.gynecological_surgery, icon: 'fa-procedures', isBoolean: true },
        { label: 'Enf. Inflamatoria Pélvica', value: data.pelvic_inflammatory, icon: 'fa-virus', isBoolean: true },
        { label: 'Quistes Ováricos', value: data.ovarian_cysts, icon: 'fa-egg', isBoolean: true },
        { label: 'Endometriosis Familiar', value: data.family_endometriosis, icon: 'fa-dna', isBoolean: true },
        { label: 'Enf. Autoinmunes Familiar', value: data.family_autoimmune, icon: 'fa-allergies', isBoolean: true },
        { label: 'Cáncer Familiar', value: data.family_cancer, icon: 'fa-ribbon', isBoolean: true },
        { label: 'Enf. Autoinmune', value: data.comorbidity_autoimmune, icon: 'fa-allergy', isBoolean: true },
        { label: 'Trastorno Tiroideo', value: data.comorbidity_thyroid, icon: 'fa-butterfly', isBoolean: true },
        { label: 'Síndrome Intestino Irritable', value: data.comorbidity_ibs, icon: 'fa-stomach', isBoolean: true }
    ];

    medicalData.forEach(item => {
        const itemElement = createModalItem(item.label, item.value, item.icon, item.isBoolean);
        container.appendChild(itemElement);
    });
}

function fillMedicationsInfo(data) {
    const container = document.getElementById('medicationsInfo');

    if (data.medications) {
        container.innerHTML = `
            <div class="modal-value">
                ${data.medications.replace(/\n/g, '<br>')}
            </div>
        `;
    } else {
        container.innerHTML = '<div class="modal-value empty">No se registraron medicamentos</div>';
    }
}

function fillExamResults(data) {
    const container = document.getElementById('examResults');
    container.innerHTML = '';

    const examData = [
        { label: 'Estatura', value: data.height ? `${data.height} cm` : null, icon: 'fa-ruler-vertical' },
        { label: 'Peso', value: data.weight ? `${data.weight} kg` : null, icon: 'fa-weight' },
        { label: 'IMC', value: data.bmi, icon: 'fa-calculator' },
        { label: 'Examen Pélvico', value: data.pelvic_exam, icon: 'fa-procedures' },
        { label: 'Examen Vaginal', value: data.vaginal_exam, icon: 'fa-female' },
        { label: 'Notas Clínicas', value: data.clinical_notes, icon: 'fa-notes-medical' }
    ];

    examData.forEach(item => {
        const itemElement = createModalItem(item.label, item.value, item.icon);
        container.appendChild(itemElement);
    });
}

function fillBiomarkersInfo(data) {
    const container = document.getElementById('biomarkersInfo');
    container.innerHTML = '';

    const biomarkersData = [
        { label: 'CA-125', value: data.ca125 ? `${data.ca125} U/mL` : null, icon: 'fa-flask' },
        { label: 'IL-6', value: data.il6 ? `${data.il6} pg/mL` : null, icon: 'fa-flask' },
        { label: 'TNF-α', value: data.tnf_alpha ? `${data.tnf_alpha} pg/mL` : null, icon: 'fa-flask' },
        { label: 'VEGF', value: data.vegf ? `${data.vegf} pg/mL` : null, icon: 'fa-flask' },
        { label: 'AMH', value: data.amh ? `${data.amh} ng/mL` : null, icon: 'fa-flask' },
        { label: 'PCR', value: data.crp ? `${data.crp} mg/L` : null, icon: 'fa-flask' }
    ];

    biomarkersData.forEach(item => {
        const itemElement = createModalItem(item.label, item.value, item.icon);
        container.appendChild(itemElement);
    });
}

function fillImagingInfo(data) {
    const container = document.getElementById('imagingInfo');

    let content = '';
    if (data.imaging) {
        content += `<div class="modal-value"><strong>Resultado:</strong> ${data.imaging}</div>`;
    }

    if (data.imaging_details) {
        content += `<div class="modal-value" style="margin-top: 10px;">${data.imaging_details.replace(/\n/g, '<br>')}</div>`;
    }

    if (content) {
        container.innerHTML = content;
    } else {
        container.innerHTML = '<div class="modal-value empty">No se registraron resultados de imágenes</div>';
    }
}

// Función auxiliar para crear items del modal
function createModalItem(label, value, icon, isBoolean = false) {
    const item = document.createElement('div');
    item.className = 'modal-item';

    if (isBoolean) {
        const booleanValue = value ? 'Sí' : 'No';
        const booleanClass = value ? 'boolean-true' : 'boolean-false';
        item.innerHTML = `
            <div class="modal-label">
                <i class="fas ${icon}"></i>
                <span>${label}</span>
            </div>
            <div class="boolean-value ${booleanClass}">
                <i class="fas ${value ? 'fa-check' : 'fa-times'}"></i>
                ${booleanValue}
            </div>
        `;
    } else {
        item.innerHTML = `
            <div class="modal-label">
                <i class="fas ${icon}"></i>
                <span>${label}</span>
            </div>
            <div class="modal-value">
                ${value || '<span class="empty">No registrado</span>'}
            </div>
        `;
    }

    return item;
}

// Configurar pestañas del modal
function setupModalTabs() {
    const tabButtons = document.querySelectorAll('.modal-tab-btn');
    const tabContents = document.querySelectorAll('.modal-tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');

            // Remover clase active de todos los botones y contenidos
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Agregar clase active al botón y contenido seleccionado
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Configurar eventos del modal
function setupModalEvents() {
    const modal = document.getElementById('patientModal');
    const closeModal = document.querySelector('.close-modal');
    const closeBtn = document.querySelector('.modal-btn.close-btn');

    // Cerrar modal al hacer clic en la X
    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Cerrar modal al hacer clic en el botón Cerrar
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Cerrar modal al hacer clic fuera del contenido
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Manejo del consentimiento informado
document.addEventListener('DOMContentLoaded', function () {
    const consentModal = document.getElementById('consentModal');
    const consentCheckbox = document.getElementById('consentCheckbox');
    const consentAcceptBtn = document.getElementById('consentAcceptBtn');

    // Versión actual del sistema (cambiar esto cuando haya actualizaciones importantes)
    const CURRENT_SYSTEM_VERSION = '2.1';

    // Habilitar botón cuando se marca el checkbox
    consentCheckbox.addEventListener('change', function () {
        consentAcceptBtn.disabled = !this.checked;
    });

    // Cerrar modal al aceptar
    consentAcceptBtn.addEventListener('click', function () {
        consentModal.style.display = 'none';
        // Guardar en localStorage que el consentimiento fue aceptado
        localStorage.setItem('consentAccepted', 'true');
        localStorage.setItem('consentDate', new Date().toISOString());
        localStorage.setItem('acceptedVersion', CURRENT_SYSTEM_VERSION);
    });

    // Verificar si debe mostrarse el modal
    function shouldShowConsentModal() {
        // 1. Si nunca ha aceptado, mostrar
        if (localStorage.getItem('consentAccepted') !== 'true') {
            return true;
        }

        // 2. Verificar expiración (6 meses)
        const consentDate = localStorage.getItem('consentDate');
        if (consentDate) {
            const sixMonthsAgo = new Date();
            sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

            if (new Date(consentDate) < sixMonthsAgo) {
                return true;
            }
        }

        // 3. Verificar si la versión aceptada es diferente a la actual
        const acceptedVersion = localStorage.getItem('acceptedVersion');
        if (acceptedVersion !== CURRENT_SYSTEM_VERSION) {
            return true;
        }

        return false;
    }

    // Mostrar modal si es necesario
    if (shouldShowConsentModal()) {
        consentModal.style.display = 'block';
    } else {
        consentModal.style.display = 'none';
    }
});

// Subida de archivos
document.getElementById('fileUpload').addEventListener('change', function(e) {
  const files = e.target.files;
  const previewContainer = document.getElementById('filePreviews');
  previewContainer.innerHTML = '';
  
  if (files.length > 5) {
    showError('Máximo 5 archivos permitidos');
    return;
  }

  Array.from(files).forEach(file => {
    if (file.size > 2 * 1024 * 1024) {
      showError(`El archivo ${file.name} excede 2MB`);
      return;
    }

    const preview = document.createElement('div');
    preview.className = 'file-preview';
    
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        preview.innerHTML = `
          <img src="${event.target.result}" alt="${file.name}">
          <button class="remove-file"><i class="fas fa-times"></i></button>
        `;
        previewContainer.appendChild(preview);
      };
      reader.readAsDataURL(file);
    } else if (file.type === 'application/pdf') {
      preview.innerHTML = `
        <div class="pdf-preview">
          <i class="fas fa-file-pdf"></i>
          <span>${file.name}</span>
          <button class="remove-file"><i class="fas fa-times"></i></button>
        </div>
      `;
      previewContainer.appendChild(preview);
    }
  });
});

// Eliminar archivos
document.addEventListener('click', function(e) {
  if (e.target.closest('.remove-file')) {
    e.target.closest('.file-preview').remove();
  }
});