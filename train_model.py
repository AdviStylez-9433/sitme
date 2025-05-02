import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
import joblib
import os
from time import time

def generate_endometriosis_dataset(n_samples=10000):
    """Genera dataset sintético mejorado con mayor proporción de casos positivos"""
    np.random.seed(42)
    
    # Aumentar la proporción base de casos positivos del 28% al 35%
    n_positive = int(n_samples * 0.35)
    n_negative = n_samples - n_positive
    
    # 1. Datos demográficos - diferenciar por condición
    age_negative = np.clip(skewnorm.rvs(4, loc=24, scale=7, size=n_negative), 15, 45).astype(int)
    age_positive = np.clip(skewnorm.rvs(5, loc=30, scale=6, size=n_positive), 18, 45).astype(int)
    age = np.concatenate([age_negative, age_positive])
    
    bmi_negative = np.round(np.clip(np.random.normal(25, 4, n_negative), 16, 45), 1)
    bmi_positive = np.round(np.clip(np.random.normal(23, 3, n_positive), 16, 40), 1)
    bmi = np.concatenate([bmi_negative, bmi_positive])
    
    # 2. Historia menstrual - menarquia más temprana en casos positivos
    menarche_negative = np.clip(np.random.normal(12.8, 1.3, n_negative), 8, 16).astype(int)
    menarche_positive = np.clip(np.random.normal(11.8, 1.5, n_positive), 8, 15).astype(int)
    menarche_age = np.concatenate([menarche_negative, menarche_positive])
    
    # 3. Síntomas de dolor - más severo en positivos
    pain_negative = np.clip(np.random.normal(4, 1.5, n_negative), 1, 8).astype(int)
    pain_positive = np.clip(np.random.normal(7, 1.2, n_positive), 4, 10).astype(int)
    pain_level = np.concatenate([pain_negative, pain_positive])
    
    # 4. Biomarcadores - niveles más altos en positivos
    ca125_negative = np.random.lognormal(2.5, 0.3, n_negative)
    ca125_positive = np.random.lognormal(3.8, 0.5, n_positive)
    ca125 = np.round(np.concatenate([ca125_negative, ca125_positive]), 1)
    
    crp_negative = np.random.exponential(2, n_negative)
    crp_positive = np.random.exponential(5, n_positive)
    crp = np.round(np.concatenate([crp_negative, crp_positive]), 2)
    
    # 5. Síntomas - mayor prevalencia en positivos
    symptom_params = {
        'dysmenorrhea': (3, 2, 0.5, 0.2),
        'dyspareunia': (2, 3, 0.4, 0.3),
        'chronic_pelvic_pain': (2, 4, 0.3, 0.4),
        'infertility': (1, 3, 0.25, 0.35),
        'family_history': (1, 4, 0.2, 0.25)
    }
    
    symptoms = {}
    for name, (a, b, threshold_neg, threshold_pos) in symptom_params.items():
        symptoms_neg = beta.rvs(a, b, size=n_negative) > threshold_neg
        symptoms_pos = beta.rvs(a, b, size=n_positive) > (threshold_pos + threshold_neg)
        symptoms[name] = np.concatenate([symptoms_neg, symptoms_pos])
    
    # 6. Diagnóstico - asegurar correlación con características
    base_risk = (
        0.35 * (pain_level > 6).astype(float) +
        0.25 * (ca125 > 35).astype(float) +
        0.15 * symptoms['dysmenorrhea'].astype(float) +
        0.10 * symptoms['family_history'].astype(float) +
        0.08 * (menarche_age < 12).astype(float) +
        0.07 * (bmi < 22).astype(float)
    )
    endometriosis = (base_risk + np.random.normal(0, 0.05, n_samples)) > 0.5
    
    # 7. Crear DataFrame
    data = {
        'age': age,
        'bmi': bmi,
        'menarche_age': menarche_age,
        'cycle_length': np.clip(np.random.normal(28, 3, n_samples), 21, 35).astype(int),
        'period_duration': np.clip(np.random.normal(5, 1.5, n_samples), 2, 10).astype(int),
        'pain_level': pain_level,
        'ca125': ca125,
        'crp': crp,
        'endometriosis': endometriosis.astype(int)
    }
    data.update({k: v.astype(int) for k, v in symptoms.items()})
    
    # Mezclar los datos
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def train_optimized_model():
    """Entrena un modelo optimizado para la clase positiva"""
    start_time = time()
    
    # 1. Generar o cargar dataset
    dataset_path = "data/endometriosis_improved.csv"
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        print("Generando dataset sintético mejorado...")
        df = generate_endometriosis_dataset(20000)
        df.to_csv(dataset_path, index=False)
    else:
        print("Cargando dataset existente...")
        df = pd.read_csv(dataset_path)
    
    # 2. Feature engineering adicional
    features = [
        'age', 'bmi', 'menarche_age', 'cycle_length', 
        'period_duration', 'pain_level', 'ca125', 'crp',
        'dysmenorrhea', 'dyspareunia', 'chronic_pelvic_pain',
        'infertility', 'family_history'
    ]
    
    # Crear características de interacción importantes
    df['pain_ca125'] = df['pain_level'] * df['ca125'] / 100
    df['early_menarche'] = (df['menarche_age'] < 12).astype(int)
    df['bmi_pain'] = df['bmi'] * df['pain_level'] / 10
    features += ['pain_ca125', 'early_menarche', 'bmi_pain']
    
    X = df[features]
    y = df['endometriosis']
    
    # 3. Dividir datos conservando distribución
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Pipeline con SMOTE y RandomForest optimizado
    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
    
    model = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    # Hiperparámetros para optimización
    param_dist = {
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [10, 15, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2'],
        'smote__sampling_strategy': [0.5, 0.6, 0.7]
    }
    
    pipeline = imbpipeline([
        ('smote', smote),
        ('model', model)
    ])
    
    # 5. Búsqueda aleatoria enfocada en recall de la clase positiva
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        scoring='recall',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_iter=15,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\nOptimizando modelo para clase positiva...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"\nMejores parámetros: {search.best_params_}")
    
    # 6. Evaluación detallada
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n🔍 Evaluación del Modelo Optimizado:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Matriz de Confusión:")
    print(f"Verdaderos Negativos: {cm[0,0]} | Falsos Positivos: {cm[0,1]}")
    print(f"Falsos Negativos: {cm[1,0]} | Verdaderos Positivos: {cm[1,1]}")
    
    # 7. Calibración final
    print("\nCalibrando probabilidades...")
    calibrated_model = CalibratedClassifierCV(
        best_model.named_steps['model'],
        method='isotonic',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )
    
    # Reentrenar con los mejores parámetros en datos balanceados
    X_res, y_res = smote.set_params(**{'sampling_strategy': search.best_params_['smote__sampling_strategy']}).fit_resample(X_train, y_train)
    calibrated_model.fit(X_res, y_res)
    
    # Evaluación final
    y_pred_cal = calibrated_model.predict(X_test)
    y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    print("\n🔍 Evaluación del Modelo Calibrado:")
    print(classification_report(y_test, y_pred_cal))
    print(f"AUC-ROC Calibrado: {roc_auc_score(y_test, y_proba_cal):.3f}")
    
    # 8. Guardar modelo
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/endometriosis_model_optimized.pkl"
    joblib.dump(calibrated_model, model_path, compress=3)
    
    print(f"\n✅ Modelo optimizado guardado en {model_path}")
    print(f"📊 Distribución de clases: {y.mean():.2%} positivos")
    print(f"⏱ Tiempo total: {(time() - start_time)/60:.2f} minutos")

if __name__ == "__main__":
    train_optimized_model()