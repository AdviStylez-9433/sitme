import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from time import time

import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

def generate_endometriosis_dataset(n_samples=10000):
    """Genera dataset sintético mejorado de endometriosis con mayor eficiencia"""
    np.random.seed(42)
    
    # Optimización: Generación vectorizada de datos
    # 1. Datos demográficos (usar clip directamente en la generación)
    age = np.clip(skewnorm.rvs(4, loc=25, scale=8, size=n_samples), 15, 45).astype(int)
    bmi = np.round(np.clip(np.random.normal(24, 4, n_samples), 16, 45), 1)
    
    # 2. Historia menstrual (evitar llamadas redundantes a clip)
    menarche_age = np.clip(np.random.normal(12.5, 1.5, n_samples), 8, 16).astype(int)
    cycle_length = np.clip(np.random.normal(28, 3, n_samples), 21, 35).astype(int)
    period_duration = np.clip(np.random.normal(5, 1.5, n_samples), 2, 10).astype(int)
    
    # 3. Síntomas de dolor (optimizar la concatenación)
    n_pain_low = int(n_samples*0.4)
    pain_level = np.empty(n_samples)
    pain_level[:n_pain_low] = np.random.normal(3, 1.2, n_pain_low)
    pain_level[n_pain_low:] = np.random.normal(7, 1.3, n_samples - n_pain_low)
    pain_level = np.clip(pain_level, 1, 10).round().astype(int)
    
    # 4. Biomarcadores (optimizar generación)
    n_ca125_low = int(n_samples*0.6)
    ca125 = np.empty(n_samples)
    ca125[:n_ca125_low] = np.random.lognormal(2.5, 0.4, n_ca125_low)
    ca125[n_ca125_low:] = np.random.lognormal(3.5, 0.6, n_samples - n_ca125_low)
    ca125 = np.round(ca125, 1)
    crp = np.round(np.random.exponential(3, n_samples), 2)
    
    # 5. Síntomas (usar una sola llamada a beta.rvs para todos los síntomas)
    symptom_params = {
        'dysmenorrhea': (3, 2, 0.5),
        'dyspareunia': (2, 3, 0.4),
        'chronic_pelvic_pain': (2, 4, 0.3),
        'infertility': (1, 3, 0.25),
        'family_history': (1, 4, 0.2)
    }
    
    symptoms = {}
    for name, (a, b, threshold) in symptom_params.items():
        symptoms[name] = (beta.rvs(a, b, size=n_samples)) > threshold
    
    # 6. Diagnóstico (vectorizar completamente)
    base_risk = (
        0.3 * (pain_level > 6).astype(float) +
        0.2 * (ca125 > 35).astype(float) +
        0.15 * symptoms['dysmenorrhea'].astype(float) +
        0.1 * symptoms['family_history'].astype(float) +
        0.05 * (menarche_age < 12).astype(float)
    )
    endometriosis = (base_risk + np.random.normal(0, 0.1, n_samples)) > 0.45
    
    # 7. Crear DataFrame (optimizar usando dict comprehension)
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
    """Entrena y guarda el modelo con optimizaciones de eficiencia"""
    start_time = time()
    
    # 1. Generar o cargar dataset
    dataset_path = "data/endometriosis_dataset.csv"
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        print("Generando dataset sintético...")
        df = generate_endometriosis_dataset(15000)
        df.to_csv(dataset_path, index=False)
    else:
        print("Cargando dataset existente...")
        df = pd.read_csv(dataset_path)
    
    # 2. Preprocesamiento (selección optimizada de características)
    features = [
        'age', 'bmi', 'menarche_age', 'cycle_length', 
        'period_duration', 'pain_level', 'ca125', 'crp',
        'dysmenorrhea', 'dyspareunia', 'chronic_pelvic_pain',
        'infertility', 'family_history'
    ]
    X = df[features]
    y = df['endometriosis']
    
    # 3. Dividir datos (optimizar stratify para datasets grandes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Configurar modelo con parámetros optimizados para eficiencia
    model = RandomForestClassifier(
        n_estimators=200,  # Reducido para velocidad pero manteniendo rendimiento
        max_depth=10,      # Profundidad ligeramente reducida
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,        # Usar todos los cores
        max_samples=0.8,   # Submuestreo para mayor velocidad
        bootstrap=True
    )
    
    # 5. Entrenar con calibración optimizada
    print("\nEntrenando modelo...")
    calibrated_model = CalibratedClassifierCV(
        model, 
        method='isotonic', 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Menos folds para velocidad
    )
    
    calibrated_model.fit(X_train, y_train)
    
        # 6. Evaluar (con métricas adicionales)
    print("\n🔍 Evaluación del Modelo:")
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # === Añadir explicabilidad con SHAP ===
    print("\nGenerando explicaciones SHAP...")
    try:
        # Usar el primer estimador calibrado (hay uno por fold de calibración)
        base_estimator = calibrated_model.calibrated_classifiers_[0].estimator
        
        # Crear el explainer SHAP
        explainer = shap.TreeExplainer(base_estimator)
        
        # Calcular valores SHAP para una muestra de los datos de prueba (por eficiencia)
        sample_idx = np.random.choice(X_test.index, size=min(100, len(X_test)), replace=False)
        X_test_sample = X_test.loc[sample_idx]
        shap_values = explainer.shap_values(X_test_sample)
        
        # Guardar gráfico SHAP summary
        plt.figure()
        shap.summary_plot(shap_values[1], X_test_sample, show=False)
        plt.savefig("static/plots/shap_summary.png", bbox_inches='tight')
        plt.close()
        
        print("✅ Explicaciones SHAP generadas correctamente")
    except Exception as e:
        print(f"⚠️ Error generando explicaciones SHAP: {str(e)}")
    
    # === Añadir explicabilidad con LIME ===
    print("Generando explicaciones LIME...")
    try:
        lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=features,
            class_names=['No Endometriosis', 'Endometriosis'],
            verbose=False,
            mode='classification'
        )
        
        # Guardar el explainer de LIME
        joblib.dump(lime_explainer, f"{model_dir}/lime_explainer.pkl")
        print("✅ Explicaciones LIME generadas correctamente")
    except Exception as e:
        print(f"⚠️ Error generando explicaciones LIME: {str(e)}")
    
    # 7. Guardar modelo optimizado
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/endometriosis_model_optimized.pkl"
    
    # Guardar todos los componentes necesarios
    joblib.dump({
        'model': calibrated_model,
        'features': features,
        'class_names': ['No Endometriosis', 'Endometriosis']
    }, model_path, compress=3)
    
    print(f"\n✅ Modelo entrenado y guardado en {model_path}")
    print(f"📊 Distribución de clases: {y.mean():.2%} positivos")
    print(f"⏱ Tiempo total de ejecución: {time() - start_time:.2f} segundos")
    
    # 6. Evaluar (con métricas adicionales)
    print("\n🔍 Evaluación del Modelo:")
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # 7. Guardar modelo optimizado
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/endometriosis_model_optimized.pkl"
    
    # Usar compresión para reducir tamaño del archivo
    joblib.dump(calibrated_model, model_path, compress=3)
    
    print(f"\n✅ Modelo entrenado y guardado en {model_path}")
    print(f"📊 Distribución de clases: {y.mean():.2%} positivos")
    print(f"⏱ Tiempo total de ejecución: {time() - start_time:.2f} segundos")

if __name__ == "__main__":
    train_and_save_model()