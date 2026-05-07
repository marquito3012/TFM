import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import os

# Configuración de rutas
DATA_DIR = "/compartida/data"
OUTPUT_DIR = "/compartida/outputs"
REPORT_DIR = "/compartida/outputs/reports"

def preprocess_for_mia(df, encoders=None, scaler=None):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    if encoders is None:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        
    return df, encoders, scaler

def run_mia_attack(real_train_p, real_test_p, synthetic_p):
    # Fit NN on Synthetic data
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn.fit(synthetic_p)
    
    # Distances from Real-Train to Synthetic
    dist_train, _ = nn.kneighbors(real_train_p)
    # Distances from Real-Test to Synthetic
    dist_test, _ = nn.kneighbors(real_test_p)
    
    # MIA attempt: Use distance as a predictor of membership
    # Small distance -> Member (1), Large distance -> Non-member (0)
    # Note: We invert distance because smaller distance = higher likelihood of being a member
    y_true = np.concatenate([np.ones(len(dist_train)), np.zeros(len(dist_test))])
    y_scores = np.concatenate([-dist_train.flatten(), -dist_test.flatten()])
    
    auc = roc_auc_score(y_true, y_scores)
    
    # Optimal threshold for accuracy
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (y_scores >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    return auc, acc

def main():
    print("Cargando datos...")
    real = pd.read_csv(f"{DATA_DIR}/diabetic_data_clean.csv")
    
    # Simular división que el usuario hizo para el entrenamiento (suponemos 80/20)
    real_train, real_test = train_test_split(real, test_size=0.2, random_state=42)
    
    print("Preprocesando...")
    real_train_p, encoders, scaler = preprocess_for_mia(real_train)
    real_test_p, _, _ = preprocess_for_mia(real_test, encoders, scaler)
    
    models = ["CTGAN", "TVAE", "TabDDPM"]
    files = ["synthetic_ctgan.csv", "synthetic_tvae.csv", "synthetic_tabddpm.csv"]
    
    results = []
    
    for name, file in zip(models, files):
        print(f"Ejecutando ataque MIA sobre {name}...")
        syn = pd.read_csv(f"{OUTPUT_DIR}/{file}")
        syn = syn[real.columns] # Alinear columnas
        
        syn_p, _, _ = preprocess_for_mia(syn, encoders, scaler)
        
        auc, acc = run_mia_attack(real_train_p, real_test_p, syn_p)
        
        print(f"  - AUC del Atacante: {auc:.4f}")
        print(f"  - Accuracy del Atacante: {acc:.4f}")
        
        results.append({
            "Model": name,
            "MIA_AUC": auc,
            "MIA_Accuracy": acc
        })

    # Guardar resultados
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{REPORT_DIR}/mia_results.csv", index=False)
    
    print("\nResumen MIA (Valores cercanos a 0.5 indican alta privacidad):")
    print(res_df)

if __name__ == "__main__":
    main()
