import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
from datetime import datetime

def generate_endometriosis_dataset(n_samples=20000):
    """Genera dataset sintético mejorado con relaciones clínicas más realistas"""
    np.random.seed(42)
    
    # 1. Datos demográficos - distribuciones diferenciadas por condición
    age_non_endo = np.clip(np.random.normal(28, 6, int(n_samples*0.7)), 15, 45).astype(int)
    age_endo = np.clip(np.random.normal(32, 5, int(n_samples*0.3)), 20, 45).astype(int)
    age = np.concatenate([age_non_endo, age_endo])
    np.random.shuffle(age)
    
    # BMI: mayor prevalencia en mujeres con BMI bajo o normal
    bmi_non_endo = np.clip(np.random.normal(26, 4, int(n_samples*0.7)), 18, 45).round(1)
    bmi_endo = np.clip(np.random.normal(23, 3, int(n_samples*0.3)), 17, 38).round(1)
    bmi = np.concatenate([bmi_non_endo, bmi_endo])
    np.random.shuffle(bmi)
    
    # 2. Historia menstrual - menarquia temprana más común en endometriosis
    menarche_non_endo = np.clip(np.random.normal(12.8, 1.3, int(n_samples*0.7)), 8, 16).astype(int)
    menarche_endo = np.clip(np.random.normal(11.5, 1.5, int(n_samples*0.3)), 8, 15).astype(int)
    menarche_age = np.concatenate([menarche_non_endo, menarche_endo])
    np.random.shuffle(menarche_age)
    
    # 3. Síntomas de dolor - más severo en endometriosis
    pain_non_endo = np.random.normal(4, 1.5, int(n_samples*0.7)).clip(1, 8).round().astype(int)
    pain_endo = np.random.normal(7, 1.2, int(n_samples*0.3)).clip(4, 10).round().astype(int)
    pain_level = np.concatenate([pain_non_endo, pain_endo])
    np.random.shuffle(pain_level)
    
    # 4. Biomarcadores - CA125 elevado en endometriosis
    ca125_non_endo = np.random.lognormal(2.5, 0.3, int(n_samples*0.7)).round(1)
    ca125_endo = np.random.lognormal(3.8, 0.5, int(n_samples*0.3)).round(1)
    ca125 = np.concatenate([ca125_non_endo, ca125_endo])
    np.random.shuffle(ca125)
    
    # CRP - inflamación más común en endometriosis
    crp_non_endo = np.random.exponential(2, int(n_samples*0.7)).round(2)
    crp_endo = np.random.exponential(5, int(n_samples*0.3)).round(2)
    crp = np.concatenate([crp_non_endo, crp_endo])
    np.random.shuffle(crp)
    
    # 5. Síntomas - relaciones más realistas con la condición
    base_risk = (
        0.4 * (pain_level > 6) +
        0.3 * (ca125 > 35) +
        0.2 * (menarche_age < 12) +
        0.1 * (bmi < 22)
    )
    
    # Síntomas correlacionados con el riesgo base
    dysmenorrhea = (beta.rvs(3, 2, size=n_samples) + 0.3*(base_risk > 0.5)) > 0.6
    dyspareunia = (beta.rvs(2, 3, size=n_samples) + 0.4*(base_risk > 0.6)) > 0.55
    chronic_pelvic_pain = (beta.rvs(2, 4, size=n_samples) + 0.5*(base_risk > 0.7)) > 0.5
    infertility = (beta.rvs(1, 3, size=n_samples) + 0.6*(base_risk > 0.8)) > 0.4
    family_history = (beta.rvs(1, 4, size=n_samples) + 0.2*(base_risk > 0.4)) > 0.25
    
    # 6. Diagnóstico con relaciones más complejas
    endometriosis = (
        (0.25 * (pain_level > 7) +
         0.20 * (ca125 > 40) +
         0.15 * (dysmenorrhea) +
         0.10 * (family_history) +
         0.10 * (menarche_age < 12) +
         0.10 * (chronic_pelvic_pain) +
         0.05 * (bmi < 20) +
         0.05 * (crp > 5) +
         np.random.normal(0, 0.05, n_samples)) > 0.55
    ).astype(int)
    
    # 7. Crear DataFrame con características adicionales
    data = {
        'age': age,
        'bmi': bmi,
        'menarche_age': menarche_age,
        'cycle_length': np.clip(np.random.normal(28, 3, n_samples), 21, 35).astype(int),
        'period_duration': np.clip(np.random.normal(5, 1.5, n_samples), 2, 10).astype(int),
        'pain_level': pain_level,
        'ca125': ca125,
        'crp': crp,
        'dysmenorrhea': dysmenorrhea.astype(int),
        'dyspareunia': dyspareunia.astype(int),
        'chronic_pelvic_pain': chronic_pelvic_pain.astype(int),
        'infertility': infertility.astype(int),
        'family_history': family_history.astype(int),
        'endometriosis': endometriosis
    }
    
    # Características adicionales basadas en relaciones conocidas
    data['pain_during_ovulation'] = ((beta.rvs(2, 3, size=n_samples) + 0.3*endometriosis) > 0.5).astype(int)
    data['heavy_bleeding'] = ((beta.rvs(3, 2, size=n_samples) + 0.2*endometriosis) > 0.6).astype(int)
    data['bowel_symptoms'] = ((beta.rvs(1, 4, size=n_samples) + 0.4*endometriosis) > 0.4).astype(int)
    data['fatigue'] = ((beta.rvs(2, 3, size=n_samples) + 0.3*endometriosis) > 0.5).astype(int)
    
    return pd.DataFrame(data)

def optimize_model(X_train, y_train):
    """Optimiza los hiperparámetros del modelo usando RandomizedSearchCV"""
    base_model = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    param_dist = {
        'n_estimators': [300, 400, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("⏳ Optimizando hiperparámetros...")
    search.fit(X_train, y_train)
    
    print(f"🎯 Mejores parámetros: {search.best_params_}")
    print(f"🏆 Mejor AUC: {search.best_score_:.3f}")
    
    return search.best_estimator_

def train_and_save_model():
    """Entrena y guarda el modelo mejorado"""
    # 1. Generar o cargar dataset
    dataset_path = "data/endometriosis_dataset_v2.csv"
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        print("🧪 Generando dataset sintético mejorado...")
        df = generate_endometriosis_dataset()
        df.to_csv(dataset_path, index=False)
    else:
        print("📂 Cargando dataset existente...")
        df = pd.read_csv(dataset_path)
    
    # 2. Preprocesamiento mejorado
    features = [
        'age', 'bmi', 'menarche_age', 'cycle_length', 
        'period_duration', 'pain_level', 'ca125', 'crp',
        'dysmenorrhea', 'dyspareunia', 'chronic_pelvic_pain',
        'infertility', 'family_history', 'pain_during_ovulation',
        'heavy_bleeding', 'bowel_symptoms', 'fatigue'
    ]
    
    # Crear características de interacción
    df['pain_ca125_interaction'] = df['pain_level'] * df['ca125'] / 100
    df['early_menarche_bmi'] = (df['menarche_age'] < 12) * (df['bmi'] < 22)
    df['crp_bmi_ratio'] = df['crp'] / df['bmi']
    features += ['pain_ca125_interaction', 'early_menarche_bmi', 'crp_bmi_ratio']
    
    X = df[features]
    y = df['endometriosis']
    
    # 3. Dividir datos con validación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    
    # 4. Balanceo de clases con SMOTE
    print("⚖️ Aplicando SMOTE para balancear clases...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 5. Optimización de hiperparámetros
    best_model = optimize_model(X_train_res, y_train_res)
    
    # 6. Evaluar en conjunto de validación
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"📊 Validation AUC: {val_auc:.3f}")
    
    # 7. Calibración condicional
    if val_auc < 0.85:
        print("🔧 Calibrando probabilidades del modelo...")
        calibrated_model = CalibratedClassifierCV(
            best_model, method='isotonic', cv=5, ensemble=True)
        calibrated_model.fit(X_train_res, y_train_res)
        final_model = calibrated_model
    else:
        final_model = best_model
    
    # 8. Evaluación final
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    print("\n🔍 Evaluación Final del Modelo:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # 9. Análisis de características importantes
    importances = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Características más importantes:")
    print(importances.head(10))
    
    # 10. Guardar modelo y metadatos
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Versión basada en fecha para control de versiones
    version = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"{model_dir}/endometriosis_model_v{version}.pkl"
    
    joblib.dump(final_model, model_path)
    
    # Guardar información del entrenamiento
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(df),
        'test_auc': roc_auc_score(y_test, y_proba),
        'validation_auc': val_auc,
        'features': features,
        'top_features': importances.head(10).to_dict('records'),
        'model_params': final_model.get_params(),
        'class_distribution': {
            'positive': y.mean(),
            'negative': 1 - y.mean()
        }
    }
    
    metadata_path = f"{model_dir}/model_metadata_v{version}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Modelo mejorado entrenado y guardado en {model_path}")
    print(f"📄 Metadatos guardados en {metadata_path}")
    print(f"📊 Distribución de clases: Positivos {y.mean():.2%}, Negativos {1-y.mean():.2%}")

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento del modelo de endometriosis...")
    train_and_save_model()