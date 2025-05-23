import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from xgboost import XGBClassifier  # Cambiado de RandomForest
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
    """Genera dataset sintético con distribuciones basadas en evidencia clínica"""
    np.random.seed(42)
    
    # 1. Datos demográficos (distribuciones basadas en estudios poblacionales)
    age = np.clip(skewnorm.rvs(5, loc=28, scale=7, size=n_samples), 15, 50).astype(int)
    bmi = np.round(np.clip(np.random.normal(26, 5, n_samples), 16, 45), 1)
    
    # 2. Historia menstrual (basado en meta-análisis de endometriosis)
    menarche_age = np.clip(np.random.normal(12.2, 1.5, n_samples), 8, 17).astype(int)  # Menarquia más temprana en endometriosis
    cycle_length = np.clip(np.random.normal(27, 3, n_samples), 21, 35).astype(int)     # Ciclos más cortos
    period_duration = np.clip(np.random.normal(5.5, 1.5, n_samples), 2, 9).astype(int) # Sangrado más prolongado
    
    # 3. Dolor menstrual (distribución bimodal - población general vs endometriosis)
    pain_level = np.where(np.random.rand(n_samples) < 0.4,  # 40% probabilidad de ser caso con dolor severo
                        np.clip(np.random.normal(7.5, 1.2, n_samples), 5, 10).round().astype(int),
                        np.clip(np.random.normal(3.8, 1.5, n_samples), 1, 6).round().astype(int))
    
    # 4. Biomarcadores (valores basados en estudios de laboratorio)
    ca125 = np.where(np.random.rand(n_samples) < 0.35,  # 35% de casos con CA-125 elevado
                    np.clip(np.random.lognormal(3.9, 0.5, n_samples), 35, 300),
                    np.clip(np.random.lognormal(2.7, 0.4, n_samples), 5, 35))
    ca125 = np.round(ca125, 1)
    
    crp = np.round(np.clip(np.random.exponential(2.8, n_samples), 0.3, 15), 2)  # Inflamación más marcada
    
    # 5. Síntomas y antecedentes con prevalencias basadas en estudios
    symptoms = {
        # Síntomas principales (prevalencia en endometriosis vs población general)
        'dysmenorrhea': (beta.rvs(4, 2, size=n_samples) > 0.7).astype(int),  # 70% vs 30% general
        'dyspareunia': (beta.rvs(3, 3, size=n_samples) > 0.5).astype(int),    # 50% vs 15% general
        'chronic_pelvic_pain': (beta.rvs(3, 4, size=n_samples) > 0.4).astype(int), # 40% vs 5% general
        'pain_during_sex': (beta.rvs(3, 3, size=n_samples) > 0.5).astype(int), # 50% vs 20% general
        'bowel_symptoms': (beta.rvs(3, 3, size=n_samples) > 0.45).astype(int), # 45% vs 10% general
        'urinary_symptoms': (beta.rvs(2, 4, size=n_samples) > 0.3).astype(int), # 30% vs 5% general
        'fatigue': (beta.rvs(3, 2, size=n_samples) > 0.6).astype(int),         # 60% vs 25% general
        
        # Infertilidad (dependiendo de edad y severidad)
        'infertility': np.where(age < 35,
                              (beta.rvs(2, 5, size=n_samples) > 0.25).astype(int), # 25% <35 años
                              (beta.rvs(3, 4, size=n_samples) > 0.4).astype(int)), # 40% ≥35 años
        
        # Antecedentes ginecológicos
        'gynecological_surgery': (beta.rvs(2, 5, size=n_samples) > 0.25).astype(int), # 25% vs 10% general
        'pelvic_inflammatory': (beta.rvs(1, 8, size=n_samples) > 0.1).astype(int),    # 10% vs 2% general
        'ovarian_cysts': (beta.rvs(2, 5, size=n_samples) > 0.25).astype(int),         # 25% vs 10% general
        
        # Antecedentes familiares
        'family_endometriosis': (beta.rvs(1, 9, size=n_samples) > 0.1).astype(int),   # 10% vs 2% general
        'family_autoimmune': (beta.rvs(1, 7, size=n_samples) > 0.12).astype(int),     # 12% vs 5% general
        'family_cancer': (beta.rvs(1, 10, size=n_samples) > 0.08).astype(int),        # 8% vs 3% general
        
        # Comorbilidades
        'comorbidity_autoimmune': (beta.rvs(1, 8, size=n_samples) > 0.1).astype(int), # 10% vs 3% general
        'comorbidity_thyroid': (beta.rvs(1, 6, size=n_samples) > 0.15).astype(int),   # 15% vs 5% general
        'comorbidity_ibs': (beta.rvs(1, 5, size=n_samples) > 0.15).astype(int),       # 15% vs 5% general
        
        # Patrones de dolor
        'pain_premenstrual': (beta.rvs(3, 2, size=n_samples) > 0.6).astype(int),      # 60% vs 30% general
        'pain_menstrual': (beta.rvs(4, 1.5, size=n_samples) > 0.75).astype(int),      # 75% vs 40% general
        'pain_ovulation': (beta.rvs(2, 4, size=n_samples) > 0.3).astype(int),         # 30% vs 10% general
        'pain_chronic': (beta.rvs(3, 3, size=n_samples) > 0.45).astype(int)           # 45% vs 8% general
    }
    
    # 6. Modelo de riesgo ponderado según importancia clínica (basado en estudios)
    risk_factors = (
        # Factores principales (OR > 3 en estudios)
        0.18 * (pain_level >= 7) +                     # Dolor severo (OR 4.2)
        0.15 * (ca125 > 35) +                          # CA-125 elevado (OR 3.8)
        0.12 * symptoms['dysmenorrhea'] +              # Dismenorrea (OR 3.5)
        0.10 * symptoms['dyspareunia'] +               # Dispareunia (OR 3.2)
        0.09 * symptoms['family_endometriosis'] +      # Historia familiar (OR 3.1)
        
        # Factores secundarios (OR 2-3)
        0.07 * symptoms['bowel_symptoms'] +            # Síntomas intestinales (OR 2.8)
        0.06 * symptoms['chronic_pelvic_pain'] +       # Dolor pélvico crónico (OR 2.7)
        0.05 * symptoms['gynecological_surgery'] +     # Cirugías previas (OR 2.5)
        0.05 * (menarche_age < 12) +                   # Menarquia temprana (OR 2.4)
        0.04 * symptoms['pain_during_sex'] +           # Dolor en relaciones (OR 2.3)
        
        # Factores adicionales (OR 1.5-2)
        0.03 * symptoms['urinary_symptoms'] +          # Síntomas urinarios (OR 1.9)
        0.03 * symptoms['pelvic_inflammatory'] +       # EPI (OR 1.8)
        0.03 * symptoms['ovarian_cysts'] +             # Quistes ováricos (OR 1.7)
        0.02 * symptoms['infertility'] +               # Infertilidad (OR 1.6)
        0.02 * (bmi > 30) +                           # Obesidad (OR 1.5)
        
        # Otros factores contribuyentes
        0.02 * symptoms['pain_menstrual'] +
        0.01 * symptoms['pain_premenstrual'] +
        0.01 * symptoms['pain_ovulation'] +
        0.01 * symptoms['pain_chronic'] +
        0.01 * symptoms['comorbidity_thyroid'] +
        0.01 * symptoms['comorbidity_ibs']
    )
    
    # Umbral ajustado para prevalencia general de ~10-15%
    endometriosis = (risk_factors + np.random.normal(0, 0.05, n_samples)) > 0.55
    
    # 7. Crear DataFrame con todas las variables
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
    
    # 4. Pipeline de modelado mejorado con XGBoost
    base_model = XGBClassifier(
        n_estimators=200,  # Aumentado para XGBoost
        max_depth=5,       # Profundidad reducida para XGBoost
        learning_rate=0.05,  # Tasa de aprendizaje típica para XGBoost
        subsample=0.8,     # Submuestreo de filas
        colsample_bytree=0.8,  # Submuestreo de columnas
        gamma=0.1,         # Regularización mínima de pérdida
        reg_alpha=0.1,     # Regularización L1
        reg_lambda=1.0,    # Regularización L2
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),  # Balanceo de clases
        eval_metric='auc',  # Métrica de evaluación
    )
    
    # Pipeline con selección de características y escalado
    model_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selector', SelectFromModel(
            XGBClassifier(n_estimators=100, random_state=42),
            threshold='1.25*median')),
        ('model', base_model)
    ])
    
    # 5. Entrenamiento con validación cruzada mejorada
    print("\nEntrenando modelo mejorado con XGBoost...")
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
    print("\n🔍 Evaluación Detallada del Modelo Mejorado (XGBoost):")
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
    
    print(f"\n✅ Modelo XGBoost entrenado y guardado en {model_path}")
    print(f"📊 Distribución de clases - Positivos: {y.mean():.2%}, Negativos: {1-y.mean():.2%}")
    print(f"⏱ Tiempo total de ejecución: {time() - start_time:.2f} segundos")

if __name__ == "__main__":
    improved_train_and_save_model()