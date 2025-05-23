import pytest
from playwright.sync_api import sync_playwright

@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # headless=True para CI
        yield browser
        browser.close()

def test_login_flow(browser):
    page = browser.new_page()
    page.goto("http://localhost:5000")
    
    # Rellenar formulario de login
    page.fill("#loginEmail", "test@example.com")
    page.fill("#loginPassword", "password123")
    page.click("#loginBtn")
    
    # Verificar redirección
    assert page.is_visible("#mainAppContent")
    assert "Bienvenido" in page.inner_text("body")

def test_patient_form_submission(browser):
    page = browser.new_page()
    page.goto("http://localhost:5000")
    
    # Simular login (puedes hacerlo mediante API si prefieres)
    # ...
    
    # Rellenar formulario
    page.fill("#full_name", "Test Patient")
    page.fill("#rut", "12345678-9")
    # ... completar otros campos
    
    page.click("button[type='submit']")
    
    # Verificar resultados
    assert page.is_visible("#resultContainer")
    assert "Probabilidad" in page.inner_text("#probabilityCircle")