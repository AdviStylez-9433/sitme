import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def generate_endometriosis_dataset(n_samples=30000):
    """Genera dataset sintético mejorado de endometriosis"""
    np.random.seed(42)
    
    # 1. Datos demográficos
    age = np.clip(skewnorm.rvs(4, loc=25, scale=8, size=n_samples), 15, 45).astype(int)
    bmi = np.clip(np.random.normal(24, 4, n_samples), 16, 45).round(1)
    
    # 2. Historia menstrual
    menarche_age = np.clip(np.random.normal(12.5, 1.5, n_samples), 8, 16).astype(int)
    cycle_length = np.clip(np.random.normal(28, 3, n_samples), 21, 35).astype(int)
    period_duration = np.clip(np.random.normal(5, 1.5, n_samples), 2, 10).astype(int)
    
    # 3. Síntomas de dolor (bimodal)
    pain_level = np.concatenate([
        np.random.normal(3, 1.2, int(n_samples*0.4)),
        np.random.normal(7, 1.3, int(n_samples*0.6))
    ]).clip(1, 10).round().astype(int)
    
    # 4. Biomarcadores
    ca125 = np.concatenate([
        np.random.lognormal(2.5, 0.4, int(n_samples*0.6)),
        np.random.lognormal(3.5, 0.6, int(n_samples*0.4))
    ]).round(1)
    crp = np.random.exponential(3, n_samples).round(2)
    
    # 5. Síntomas (distribución beta para prevalencia realista)
    symptoms = {
        'dysmenorrhea': (beta.rvs(3, 2, size=n_samples) > 0.5),
        'dyspareunia': (beta.rvs(2, 3, size=n_samples) > 0.4),
        'chronic_pelvic_pain': (beta.rvs(2, 4, size=n_samples) > 0.3),
        'infertility': (beta.rvs(1, 3, size=n_samples) > 0.25),
        'family_history': (beta.rvs(1, 4, size=n_samples) > 0.2)
    }
    
    # 6. Diagnóstico (reglas clínicas + ruido)
    base_risk = (
        0.3 * (pain_level > 6) +
        0.2 * (ca125 > 35) +
        0.15 * (symptoms['dysmenorrhea']) +
        0.1 * (symptoms['family_history']) +
        0.05 * (menarche_age < 12)
    )
    endometriosis = (base_risk + np.random.normal(0, 0.1, n_samples)) > 0.45
    
    # 7. Crear DataFrame
    data = {
        'age': age,
        'bmi': bmi,
        'menarche_age': menarche_age,
        'cycle_length': cycle_length,
        'period_duration': period_duration,
        'pain_level': pain_level,
        'ca125': ca125,
        'crp': crp,
        'endometriosis': endometriosis.astype(int)
    }
    data.update({k: v.astype(int) for k, v in symptoms.items()})
    
    return pd.DataFrame(data)

def train_and_save_model():
    """Entrena y guarda el modelo"""
    # 1. Generar o cargar dataset
    dataset_path = "data/endometriosis_dataset.csv"
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        df = generate_endometriosis_dataset(15000)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)
    
    # 2. Preprocesamiento
    features = [
        'age', 'bmi', 'menarche_age', 'cycle_length', 
        'period_duration', 'pain_level', 'ca125', 'crp',
        'dysmenorrhea', 'dyspareunia', 'chronic_pelvic_pain',
        'infertility', 'family_history'
    ]
    X = df[features]
    y = df['endometriosis']
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Configurar modelo
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Entrenar con calibración
    calibrated_model = CalibratedClassifierCV(
        model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # 6. Evaluar
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print("\n🔍 Evaluación del Modelo:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # 7. Guardar modelo
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/endometriosis_model_v4.pkl"
    joblib.dump(calibrated_model, model_path)
    
    print(f"\n✅ Modelo entrenado y guardado en {model_path}")
    print(f"📊 Distribución de clases: {y.mean():.2%} positivos")

if __name__ == "__main__":
    train_and_save_model()