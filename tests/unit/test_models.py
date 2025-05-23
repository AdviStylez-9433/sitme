import pytest
from app import app, Medico, db
from werkzeug.security import check_password_hash

def test_medico_creation(test_app):
    """Test creación de médico"""
    with test_app.app_context():
        medico = Medico(
            email="doctor@test.com",
            password_hash="hashed_pass",
            nombre="Dr. Test",
            colegiado="TEST123"
        )
        db.session.add(medico)
        db.session.commit()
        
        assert Medico.query.count() == 1
        assert medico.email == "doctor@test.com"
        assert medico.activo is True

def test_password_hashing(test_app):
    """Test hashing de contraseña"""
    with test_app.app_context():
        medico = Medico(email="test@test.com")
        medico.set_password("securepassword123")
        
        assert check_password_hash(medico.password_hash, "securepassword123")
        assert not check_password_hash(medico.password_hash, "wrongpassword")

def test_auth_token(test_app):
    """Test generación de token JWT"""
    with test_app.app_context():
        medico = Medico(email="token@test.com")
        token = medico.generate_auth_token()
        
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT tiene 3 partes