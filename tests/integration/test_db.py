import pytest
from app import app, Medico, db

def test_db_connection(test_app):
    """Test conexión a la base de datos"""
    with test_app.app_context():
        # Verificar que las tablas existen
        assert 'medicos' in db.engine.table_names()
        assert 'patient_simulations' in db.engine.table_names()

def test_medico_crud_operations(test_app):
    """Test operaciones CRUD para Médico"""
    with test_app.app_context():
        # Create
        medico = Medico(
            email="crud@test.com",
            password_hash="hashed",
            nombre="Dr. CRUD"
        )
        db.session.add(medico)
        db.session.commit()
        
        # Read
        found = Medico.query.filter_by(email="crud@test.com").first()
        assert found is not None
        assert found.nombre == "Dr. CRUD"
        
        # Update
        found.nombre = "Dr. Updated"
        db.session.commit()
        updated = Medico.query.get(found.id)
        assert updated.nombre == "Dr. Updated"
        
        # Delete
        db.session.delete(updated)
        db.session.commit()
        assert Medico.query.get(found.id) is None

def test_patient_simulation_creation(test_app, sample_patient_data):
    """Test creación de simulación de paciente"""
    with test_app.app_context():
        # Crear médico primero
        medico = Medico(email="doctor@sim.com", nombre="Dr. Sim")
        db.session.add(medico)
        db.session.commit()
        
        # Insertar simulación
        query = """
            INSERT INTO patient_simulations (
                clinic_id, full_name, id_number, age,
                gynecological_surgery, family_endometriosis,
                pain_level, pain_during_sex, probability, risk_level
            ) VALUES (
                'TEST123', %(name)s, '12345678-9', 30,
                TRUE, TRUE, 7, TRUE, 0.75, 'high'
            )
        """
        params = {'name': 'Test Patient'}
        
        result = db.engine.execute(query, params)
        assert result.rowcount == 1