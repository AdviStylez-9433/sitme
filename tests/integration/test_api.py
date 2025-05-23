import pytest
import json

def test_predict_endpoint(client, auth_headers, sample_patient_data):
    """Test endpoint de predicción"""
    # Preparar datos para predicción
    predict_data = {
        "age": sample_patient_data["personal"]["age"],
        "menarche_age": sample_patient_data["menstrual"]["menarche_age"],
        "cycle_length": sample_patient_data["menstrual"]["cycle_length"],
        "period_duration": sample_patient_data["menstrual"]["period_duration"],
        "pain_level": sample_patient_data["menstrual"]["pain_level"],
        "pain_during_sex": sample_patient_data["symptoms"]["pain_during_sex"],
        "family_history": sample_patient_data["history"]["family_endometriosis"],
        "bowel_symptoms": sample_patient_data["symptoms"]["bowel_symptoms"],
        "urinary_symptoms": sample_patient_data["symptoms"]["urinary_symptoms"],
        "fatigue": sample_patient_data["symptoms"]["fatigue"],
        "infertility": sample_patient_data["symptoms"]["infertility"],
        "ca125": sample_patient_data["biomarkers"]["ca125"],
        "crp": sample_patient_data["biomarkers"]["crp"]
    }
    
    response = client.post('/predict', json=predict_data, headers=auth_headers)
    data = json.loads(response.data)
    
    assert response.status_code == 200
    assert "probability" in data
    assert "risk_level" in data
    assert "recommendations" in data
    assert isinstance(data["probability"], float)
    assert data["probability"] >= 0 and data["probability"] <= 1

def test_save_simulation(client, auth_headers, sample_patient_data):
    """Test guardar simulación"""
    response = client.post('/save_simulation', json={
        "form_data": sample_patient_data,
        "prediction": {
            "probability": 0.75,
            "risk_level": "high",
            "model_version": "v4.1",
            "recommendations": [
                "Consulta con especialista",
                "Ecografía transvaginal"
            ]
        },
        "clinic_id": "TEST123"
    }, headers=auth_headers)
    
    data = json.loads(response.data)
    assert response.status_code == 200
    assert "simulation_id" in data
    assert data["success"] is True

def test_get_history(client, auth_headers):
    """Test obtener historial"""
    response = client.get('/get_history', headers=auth_headers)
    data = json.loads(response.data)
    
    assert response.status_code == 200
    assert "records" in data
    assert isinstance(data["records"], list)