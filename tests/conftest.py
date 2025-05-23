import pytest
from app import app, db, init_db, Medico
import os
from werkzeug.security import generate_password_hash

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/test_db')
    app.config['SECRET_KEY'] = 'test-secret-key'
    
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

@pytest.fixture
def auth_headers(client):
    # Helper para obtener headers de autenticación para pruebas
    test_medico = Medico(
        email="test@example.com",
        password_hash=generate_password_hash("testpassword"),
        nombre="Test Doctor"
    )
    db.session.add(test_medico)
    db.session.commit()
    
    response = client.post('/api/login', json={
        'email': 'test@example.com',
        'password': 'testpassword'
    })
    token = response.json['token']
    
    return {'Authorization': f'Bearer {token}'}