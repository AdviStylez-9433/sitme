import pytest
from app import db, Medico
from werkzeug.security import check_password_hash, generate_password_hash

def test_create_medico():
    medico = Medico(
        nombre="Juan Pérez",
        email="juan@example.com",
        password_hash=generate_password_hash("securepass")
    )
    db.session.add(medico)
    db.session.commit()

    fetched = Medico.query.filter_by(email="juan@example.com").first()
    assert fetched is not None
    assert fetched.nombre == "Juan Pérez"
    assert check_password_hash(fetched.password_hash, "securepass")
