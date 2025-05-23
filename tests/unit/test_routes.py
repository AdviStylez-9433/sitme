import pytest
from unittest.mock import patch, MagicMock
from app import app, init_db, Medico

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()  # Inicializar DB para pruebas
        yield client

def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"SITME" in response.data

def test_predict_route_missing_data(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert b"Datos no proporcionados" in response.data

@patch('app.Medico.query')
def test_login_success(mock_query, client):
    # Configurar mock
    mock_medico = MagicMock()
    mock_medico.id = 1
    mock_medico.email = "test@example.com"
    mock_medico.check_password.return_value = True
    mock_medico.activo = True
    mock_medico.generate_auth_token.return_value = "fake_token"
    mock_query.filter_by.return_value.first.return_value = mock_medico

    # Hacer petición
    response = client.post('/api/login', json={
        'email': 'test@example.com',
        'password': 'correct_password'
    })
    
    assert response.status_code == 200
    assert b"token" in response.data