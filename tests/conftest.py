import pytest
from app import app, db, Medico
from werkzeug.security import generate_password_hash
import os

import subprocess
import time
import signal
import subprocess
import time
import pytest
import signal

BASE_URL = "https://sitme-api.onrender.com"

def test_login_flow(page):
    page.goto(f"{BASE_URL}")
    assert "SITME" in page.title()  # ajusta a lo que esperes

@pytest.fixture(scope='module')
def test_app():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('TEST_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/test_db')
    app.config['SECRET_KEY'] = 'test-secret-key'

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()   # Cerrar la sesión activa
        db.drop_all()

@pytest.fixture
def client(test_app):
    """Cliente de prueba"""
    with test_app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers(client):
    # Limpia antes para evitar duplicados
    db.session.query(Medico).delete()
    db.session.commit()

    test_medico = Medico(
        email="test@example.com",
        password_hash=generate_password_hash("testpassword"),
        nombre="Test Doctor",
        colegiado="TEST123",
        activo=True
    )
    db.session.add(test_medico)
    db.session.commit()
    
    response = client.post('/api/login', json={
        'email': 'test@example.com',
        'password': 'testpassword'
    })
    token = response.json['token']
    
    return {'Authorization': f'Bearer {token}'}

@pytest.fixture
def sample_patient_data():
    """Datos de paciente de ejemplo para pruebas"""
    return {
        "personal": {
            "full_name": "Test Patient",
            "id_number": "12345678-9",
            "birth_date": "1990-01-01",
            "age": 30,
            "blood_type": "A+",
            "insurance": "fonasa"
        },
        "history": {
            "gynecological_surgery": True,
            "pelvic_inflammatory": False,
            "ovarian_cysts": True,
            "family_endometriosis": True,
            "family_autoimmune": False,
            "family_cancer": False,
            "comorbidity_autoimmune": False,
            "comorbidity_thyroid": True,
            "comorbidity_ibs": False,
            "medications": "Ibuprofeno 400mg cada 8 horas"
        },
        "menstrual": {
            "menarche_age": 12,
            "cycle_length": 28,
            "period_duration": 5,
            "last_period": "2023-06-01",
            "pain_level": 7,
            "pain_premenstrual": True,
            "pain_menstrual": True,
            "pain_ovulation": False,
            "pain_chronic": False
        },
        "symptoms": {
            "pain_during_sex": True,
            "bowel_symptoms": True,
            "urinary_symptoms": False,
            "fatigue": True,
            "infertility": False,
            "other_symptoms": "Ninguno"
        },
        "biomarkers": {
            "ca125": 45.2,
            "il6": 5.1,
            "tnf_alpha": 12.3,
            "vegf": 350.0,
            "amh": 2.5,
            "crp": 8.7,
            "imaging": "endometrioma",
            "imaging_details": "Endometrioma ovárico derecho de 3cm"
        },
        "examination": {
            "height": 165,
            "weight": 60,
            "bmi": 22.0,
            "pelvic_exam": "tenderness",
            "vaginal_exam": "normal",
            "clinical_notes": "Dolor a la movilización uterina"
        }
    } 


