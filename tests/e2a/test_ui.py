import pytest
from playwright.sync_api import sync_playwright, expect
import time

@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Cambiar a True para CI
        yield browser
        browser.close()

@pytest.fixture
def page(browser):
    page = browser.new_page()
    yield page
    page.close()

def test_login_flow(page):
    """Test completo de flujo de login"""
    page.goto("http://localhost:5000")
    
    # Verificar que el formulario de login está visible
    expect(page.locator("#loginFormContainer")).to_be_visible()
    
    # Rellenar credenciales incorrectas
    page.fill("#loginEmail", "wrong@example.com")
    page.fill("#loginPassword", "wrongpassword")
    page.click("#loginBtn")
    
    # Verificar mensaje de error
    expect(page.locator("#loginError")).to_contain_text("Credenciales inválidas")
    
    # Rellenar credenciales correctas (mock)
    page.fill("#loginEmail", "test@example.com")
    page.fill("#loginPassword", "testpassword")
    page.click("#loginBtn")
    
    # Verificar redirección
    expect(page.locator("#mainAppContent")).to_be_visible()
    expect(page.locator("#medicoNombre")).to_contain_text("Test Doctor")

def test_patient_form_submission(page):
    """Test completo de formulario de paciente"""
    # Login primero
    page.goto("http://localhost:5000")
    page.fill("#loginEmail", "test@example.com")
    page.fill("#loginPassword", "testpassword")
    page.click("#loginBtn")
    
    # Navegar al formulario
    expect(page.locator("#mainAppContent")).to_be_visible()
    
    # Pestaña 1: Datos personales
    page.fill("#full_name", "Ana Pérez")
    page.fill("#rut", "12345678-9")
    page.fill("#birth_date", "1990-01-01")
    page.click("input[value='fonasa']")
    
    # Ir a siguiente pestaña
    page.click("button[data-next='tab2']")
    
    # Pestaña 2: Antecedentes
    page.check("#family_endometriosis")
    page.check("#comorbidity_thyroid")
    
    # Continuar llenando el formulario...
    page.click("button[data-next='tab3']")
    
    # Pestaña 3: Menstrual
    page.fill("#menarche_age", "12")
    page.fill("#cycle_length", "28")
    page.fill("#period_duration", "5")
    page.fill("#pain_level", "7")
    
    # ... completar otras pestañas
    
    # Enviar formulario
    page.click("button[type='submit']")
    
    # Verificar resultados
    expect(page.locator("#resultContainer")).to_be_visible()
    expect(page.locator("#probabilityCircle")).to_contain_text("%")
    expect(page.locator("#riskFactorsList")).not_to_be_empty()