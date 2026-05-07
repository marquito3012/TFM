import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Configuración de rutas
DATA_DIR = "/compartida/data"
OUTPUT_DIR = "/compartida/outputs"
REPORT_DIR = "/compartida/outputs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def preprocess_data(df, encoders=None):
    df = df.copy()
    # Identificar columnas categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                # Manejar etiquetas no vistas
                df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                # Si falta una columna en el encoder, la borramos o ignoramos
                df[col] = 0 
    return df, encoders

def run_tstr():
    print("Cargando datos reales...")
    real = pd.read_csv(f"{DATA_DIR}/diabetic_data_clean.csv")
    
    # El objetivo es 'readmitted'. Vamos a simplificarlo a binario para la evaluación de utilidad si es necesario, 
    # pero el dataset original tiene 'NO', '>30', '<30'.
    # Lo convertiremos a binario: 1 si reingresa (cualquier tipo), 0 si NO.
    real['target'] = (real['readmitted'] != 'NO').astype(int)
    real = real.drop(columns=['readmitted'])
    
    # Preprocesamiento base (para obtener encoders consistentes)
    real_encoded, encoders = preprocess_data(real)
    
    # Split Real-Train / Real-Test
    X = real_encoded.drop(columns=['target'])
    y = real_encoded['target']
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    # 1. Baseline: TRTR (Train Real, Test Real)
    print("Entrenando Baseline TRTR...")
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_real, y_train_real)
    y_pred = model.predict(X_test_real)
    y_prob = model.predict_proba(X_test_real)[:, 1]
    
    results.append({
        "Model": "Baseline (TRTR)",
        "F1-Score": f1_score(y_test_real, y_pred),
        "AUC-ROC": roc_auc_score(y_test_real, y_prob)
    })
    
    # 2. TSTR para cada modelo sintético
    synthetic_files = {
        "CTGAN": "synthetic_ctgan.csv",
        "TVAE": "synthetic_tvae.csv",
        "TabDDPM": "synthetic_tabddpm.csv"
    }
    
    for name, filename in synthetic_files.items():
        print(f"Evaluando TSTR para {name}...")
        syn = pd.read_csv(f"{OUTPUT_DIR}/{filename}")
        syn['target'] = (syn['readmitted'] != 'NO').astype(int)
        syn = syn.drop(columns=['readmitted'])
        
        # Preprocesar sintético con los mismos encoders
        syn_encoded, _ = preprocess_data(syn, encoders)
        
        # ASEGURAR ORDEN DE COLUMNAS (Importante para XGBoost)
        feature_order = X_test_real.columns.tolist()
        X_train_syn = syn_encoded[feature_order]
        y_train_syn = syn_encoded['target']
        
        # Entrenar en sintético, testear en REAL-TEST
        model_syn = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model_syn.fit(X_train_syn, y_train_syn)
        
        y_pred_syn = model_syn.predict(X_test_real)
        y_prob_syn = model_syn.predict_proba(X_test_real)[:, 1]
        
        results.append({
            "Model": f"{name} (TSTR)",
            "F1-Score": f1_score(y_test_real, y_pred_syn),
            "AUC-ROC": roc_auc_score(y_test_real, y_prob_syn)
        })

    results_df = pd.DataFrame(results)
    print("\nResultados de Utilidad (TSTR):")
    print(results_df)
    results_df.to_csv(f"{REPORT_DIR}/tstr_results.csv", index=False)
    
    # Calcular Brecha (Gap)
    baseline_f1 = results[0]["F1-Score"]
    results_df['F1_Gap_%'] = ((baseline_f1 - results_df['F1-Score']) / baseline_f1) * 100
    print("\nBrecha de rendimiento respecto al Baseline:")
    print(results_df[["Model", "F1_Gap_%"]])

if __name__ == "__main__":
    run_tstr()
