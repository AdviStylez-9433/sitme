import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, f1_score, precision_recall_curve, average_precision_score
import joblib
import os
from time import time
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

def improved_generate_endometriosis_dataset(n_samples=20000):
    """Genera dataset sintético con distribuciones más realistas"""
    np.random.seed(42)
    
    # 1. Datos demográficos con distribuciones más precisas
    age = np.clip(skewnorm.rvs(5, loc=28, scale=7, size=n_samples), 15, 50).astype(int)
    bmi = np.round(np.clip(np.random.normal(26, 5, n_samples), 16, 45), 1)
    
    # 2. Historia menstrual basada en estudios clínicos
    menarche_age = np.clip(np.random.normal(12.8, 1.3, n_samples), 8, 17).astype(int)
    cycle_length = np.clip(np.random.normal(28, 2.5, n_samples), 21, 35).astype(int)
    period_duration = np.clip(np.random.normal(5.2, 1.3, n_samples), 2, 9).astype(int)
    
    # 3. Síntomas de dolor con distribución bimodal mejorada
    pain_level = np.where(np.random.rand(n_samples) < 0.35, 
                         np.clip(np.random.normal(7, 1.5, n_samples), 4, 10),
                         np.clip(np.random.normal(4, 1.5, n_samples), 1, 6)).round().astype(int)
    
    # 4. Biomarcadores con valores clínicamente relevantes
    ca125 = np.where(np.random.rand(n_samples) < 0.3,
                    np.clip(np.random.lognormal(3.8, 0.6, n_samples), 5, 300),
                    np.clip(np.random.lognormal(2.7, 0.4, n_samples), 5, 35))
    ca125 = np.round(ca125, 1)
    
    crp = np.round(np.clip(np.random.exponential(2.5, n_samples) + 
                        np.random.normal(0, 0.5, n_samples), 
                        0.3, 10), 2)
    
    # 5. Síntomas con prevalencias basadas en literatura médica
    symptoms = {
        'dysmenorrhea': (beta.rvs(3, 1.5, size=n_samples) > 0.6).astype(int),
        'dyspareunia': (beta.rvs(2, 3, size=n_samples) > 0.35).astype(int),
        'chronic_pelvic_pain': (beta.rvs(2, 4, size=n_samples) > 0.25).astype(int),
        'infertility': (beta.rvs(1, 5, size=n_samples) > 0.15).astype(int),
        'family_history': (beta.rvs(1, 6, size=n_samples) > 0.1).astype(int)
    }
    
    # 6. Modelo de riesgo más sofisticado
    risk_factors = (
        0.25 * (pain_level > 6) +
        0.25 * (ca125 > 35) +
        0.15 * symptoms['dysmenorrhea'] +
        0.12 * symptoms['dyspareunia'] +
        0.1 * symptoms['family_history'] +
        0.08 * (menarche_age < 12) +
        0.05 * (bmi > 30)
    )
    
    endometriosis = (risk_factors + np.random.normal(0, 0.07, n_samples)) > 0.42
    
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
    data.update(symptoms)
    
    return pd.DataFrame(data)

def improved_train_and_save_model():
    """Entrena y guarda el modelo con mejoras significativas"""
    start_time = time()
    
    # 1. Generar o cargar dataset mejorado
    dataset_path = "data/endometriosis_dataset.csv"
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        print("Generando dataset sintético mejorado...")
        df = improved_generate_endometriosis_dataset(25000)
        df.to_csv(dataset_path, index=False)
    else:
        print("Cargando dataset existente...")
        df = pd.read_csv(dataset_path)
    
    # 2. Preprocesamiento avanzado
    features = [
        'age', 'bmi', 'menarche_age', 'cycle_length', 
        'period_duration', 'pain_level', 'ca125', 'crp',
        'dysmenorrhea', 'dyspareunia', 'chronic_pelvic_pain',
        'infertility', 'family_history'
    ]
    X = df[features]
    y = df['endometriosis']
    
    # 3. División estratificada mejorada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Pipeline de modelado mejorado
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    )
    
    # Pipeline con selección de características y escalado
    model_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selector', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='1.25*median')),
        ('model', base_model)
    ])
    
    # 5. Entrenamiento con validación cruzada mejorada
    print("\nEntrenando modelo mejorado...")
    calibrated_model = CalibratedClassifierCV(
        model_pipeline,
        method='isotonic',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # Validación cruzada para evaluación preliminar
    cv_scores = cross_val_score(calibrated_model, X_train, y_train, 
                              cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"\nValidación cruzada AUC-ROC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Entrenamiento final
    calibrated_model.fit(X_train, y_train)
    
    # 6. Evaluación exhaustiva
    print("\n🔍 Evaluación Detallada del Modelo Mejorado:")
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, digits=3))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.3f}")
    
    # 7. Guardar modelo mejorado
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/endometriosis_model_optimized.pkl"
    
    joblib.dump(calibrated_model, model_path, compress=('zlib', 3))
    
    print(f"\n✅ Modelo mejorado entrenado y guardado en {model_path}")
    print(f"📊 Distribución de clases - Positivos: {y.mean():.2%}, Negativos: {1-y.mean():.2%}")
    print(f"⏱ Tiempo total de ejecución: {time() - start_time:.2f} segundos")

if __name__ == "__main__":
    improved_train_and_save_model()