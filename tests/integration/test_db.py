import pytest
from app import app, db, Medico, get_db_connection

@pytest.fixture
def test_db():
    app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:postgres@localhost:5432/test_db"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    with app.app_context():
        db.create_all()
        yield db
        db.drop_all()

def test_medico_creation(test_db):
    medico = Medico(
        email="test@example.com",
        password_hash="hashed_password",
        nombre="Dr. Test"
    )
    test_db.session.add(medico)
    test_db.session.commit()
    
    assert Medico.query.count() == 1
    assert Medico.query.first().email == "test@example.com"

def test_save_simulation(client, test_db):
    test_data = {
        "form_data": {
            "personal": {
                "full_name": "Test Patient",
                "id_number": "12345678-9",
                "age": 30
            },
            # ... otros datos necesarios
        },
        "prediction": {
            "probability": 0.75,
            "risk_level": "high"
        }
    }
    
    response = client.post('/save_simulation', json=test_data)
    assert response.status_code == 200
    assert b"simulation_id" in response.data