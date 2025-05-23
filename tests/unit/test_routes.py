import pytest
from app import app

def test_home_route(client):
    """Test ruta home"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"SITME" in response.data

def test_protected_route_without_auth(client):
    """Test ruta protegida sin autenticación"""
    response = client.get('/api/protected-route')
    assert response.status_code == 401

def test_login_invalid_credentials(client):
    """Test login con credenciales inválidas"""
    response = client.post('/api/login', json={
        'email': 'nonexistent@test.com',
        'password': 'wrongpassword'
    })
    assert response.status_code == 401
    assert b"Credenciales" in response.data
    
def test_login_success(client, auth_headers):
    """Test login exitoso"""
    response = client.get('/api/protected-route', headers=auth_headers)
    assert response.status_code == 200