import pandas as pd
import numpy as np
from scipy.stats import skewnorm, beta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, f1_score, average_precision_score
import joblib
import os
from time import time
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

def improved_generate_endometriosis_dataset(n_samples=20000):
    """Genera dataset sintético con distribuciones basadas en evidencia clínica"""
    np.random.seed(42)
    
    # 1. Datos demográficos
    age = np.clip(skewnorm.rvs(5, loc=28, scale=7, size=n_samples), 15, 50).astype(int)
    bmi = np.round(np.clip(np.random.normal(26, 5, n_samples), 16, 45), 1)
    
    # 2. Historia menstrual
    menarche_age = np.clip(np.random.normal(12.2, 1.5, n_samples), 8, 17).astype(int)
    cycle_length = np.clip(np.random.normal(27, 3, n_samples), 21, 35).astype(int)
    period_duration = np.clip(np.random.normal(5.5, 1.5, n_samples), 2, 9).astype(int)
    
    # 3. Dolor menstrual
    pain_level = np.where(np.random.rand(n_samples) < 0.4,
                        np.clip(np.random.normal(7.5, 1.2, n_samples), 5, 10).round().astype(int),
                        np.clip(np.random.normal(3.8, 1.5, n_samples), 1, 6).round().astype(int))
    
    # 4. Biomarcadores
    ca125 = np.where(np.random.rand(n_samples) < 0.35,
                    np.clip(np.random.lognormal(3.9, 0.5, n_samples), 35, 300),
                    np.clip(np.random.lognormal(2.7, 0.4, n_samples), 5, 35))
    ca125 = np.round(ca125, 1)
    
    crp = np.round(np.clip(np.random.exponential(2.8, n_samples), 0.3, 15), 2)
    
    # 5. Síntomas y antecedentes
    symptoms = {
        'dysmenorrhea': (beta.rvs(4, 2, size=n_samples) > 0.7).astype(int),
        'dyspareunia': (beta.rvs(3, 3, size=n_samples) > 0.5).astype(int),
        'chronic_pelvic_pain': (beta.rvs(3, 4, size=n_samples) > 0.4).astype(int),
        'pain_during_sex': (beta.rvs(3, 3, size=n_samples) > 0.5).astype(int),
        'bowel_symptoms': (beta.rvs(3, 3, size=n_samples) > 0.45).astype(int),
        'urinary_symptoms': (beta.rvs(2, 4, size=n_samples) > 0.3).astype(int),
        'fatigue': (beta.rvs(3, 2, size=n_samples) > 0.6).astype(int),
        'infertility': np.where(age < 35,
                              (beta.rvs(2, 5, size=n_samples) > 0.25).astype(int),
                              (beta.rvs(3, 4, size=n_samples) > 0.4).astype(int)),
        'gynecological_surgery': (beta.rvs(2, 5, size=n_samples) > 0.25).astype(int),
        'pelvic_inflammatory': (beta.rvs(1, 8, size=n_samples) > 0.1).astype(int),
        'ovarian_cysts': (beta.rvs(2, 5, size=n_samples) > 0.25).astype(int),
        'family_history': (beta.rvs(1, 9, size=n_samples) > 0.1).astype(int),  # Cambiado a family_history
        'family_autoimmune': (beta.rvs(1, 7, size=n_samples) > 0.12).astype(int),
        'family_cancer': (beta.rvs(1, 10, size=n_samples) > 0.08).astype(int),
        'comorbidity_autoimmune': (beta.rvs(1, 8, size=n_samples) > 0.1).astype(int),
        'comorbidity_thyroid': (beta.rvs(1, 6, size=n_samples) > 0.15).astype(int),
        'comorbidity_ibs': (beta.rvs(1, 5, size=n_samples) > 0.15).astype(int),
        'pain_premenstrual': (beta.rvs(3, 2, size=n_samples) > 0.6).astype(int),
        'pain_menstrual': (beta.rvs(4, 1.5, size=n_samples) > 0.75).astype(int),
        'pain_ovulation': (beta.rvs(2, 4, size=n_samples) > 0.3).astype(int),
        'pain_chronic': (beta.rvs(3, 3, size=n_samples) > 0.45).astype(int)
    }
    
    # 6. Crear DataFrame
    data = {
        'age': age,
        'bmi': bmi,
        'menarche_age': menarche_age,
        'cycle_length': cycle_length,
        'period_duration': period_duration,
        'pain_level': pain_level,
        'ca125': ca125,
        'crp': crp,
        'endometriosis': ((np.random.rand(n_samples) < 0.15).astype(int))  # Prevalencia base del 15%
    }
    data.update(symptoms)
    
    return pd.DataFrame(data)

def improved_train_and_save_model():
    """Entrena y guarda el modelo con mejoras significativas"""
    start_time = time()
    
    # 1. Generar o cargar dataset
    dataset_path = "data/endometriosis_dataset.csv"
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        print("Generando dataset sintético mejorado...")
        df = improved_generate_endometriosis_dataset(25000)
        df.to_csv(dataset_path, index=False)
    else:
        print("Cargando dataset existente...")
        df = pd.read_csv(dataset_path)
    
    # 2. Preprocesamiento - Usar columnas que realmente existen
    features = [
        'age', 'bmi', 'menarche_age', 'cycle_length', 
        'period_duration', 'pain_level', 'ca125', 'crp',
        'dysmenorrhea', 'dyspareunia', 'chronic_pelvic_pain',
        'infertility', 'family_history'  # Cambiado a family_history
    ]
    
    # Verificar que todas las features existen en el DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {missing_features}")
    
    X = df[features]
    y = df['endometriosis']
    
    # Resto del código permanece igual...
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline de modelado
    base_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),
        eval_metric='auc'
    )
    
    model_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selector', SelectFromModel(
            XGBClassifier(n_estimators=100, random_state=42),
            threshold='1.25*median')),
        ('model', base_model)
    ])
    
    # Entrenamiento con validación cruzada
    print("\nEntrenando modelo mejorado con XGBoost...")
    calibrated_model = CalibratedClassifierCV(
        model_pipeline,
        method='isotonic',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    cv_scores = cross_val_score(calibrated_model, X_train, y_train, 
                              cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"\nValidación cruzada AUC-ROC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    calibrated_model.fit(X_train, y_train)
    
    # Evaluación
    print("\n🔍 Evaluación Detallada del Modelo Mejorado (XGBoost):")
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, digits=3))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.3f}")
    
    # Guardar modelo
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/endometriosis_model_optimized.pkl"
    
    joblib.dump(calibrated_model, model_path, compress=('zlib', 3))
    
    print(f"\n✅ Modelo XGBoost entrenado y guardado en {model_path}")
    print(f"📊 Distribución de clases - Positivos: {y.mean():.2%}, Negativos: {1-y.mean():.2%}")
    print(f"⏱ Tiempo total de ejecución: {time() - start_time:.2f} segundos")

if __name__ == "__main__":
    improved_train_and_save_model()