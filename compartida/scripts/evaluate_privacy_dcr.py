import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de rutas
DATA_DIR = "/compartida/data"
OUTPUT_DIR = "/compartida/outputs"
REPORT_DIR = "/compartida/outputs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def preprocess_for_distance(df, encoders=None, scaler=None):
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

def calculate_dcr(real_processed, synthetic_processed):
    # Fit NearestNeighbors on Real data
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn.fit(real_processed)
    
    # Find distance to closest record in Real for each Synthetic record
    distances, _ = nn.kneighbors(synthetic_processed)
    return distances.flatten()

def main():
    print("Cargando y preprocesando datos...")
    real = pd.read_csv(f"{DATA_DIR}/diabetic_data_clean.csv")
    
    # Usaremos una muestra si el dataset es demasiado grande para agilizar, 
    # pero 100k con NN es factible en un entorno con recursos.
    # Para el TFM usaremos el dataset completo para rigor científico.
    
    real_p, encoders, scaler = preprocess_for_distance(real)
    
    models = ["CTGAN", "TVAE", "TabDDPM"]
    files = ["synthetic_ctgan.csv", "synthetic_tvae.csv", "synthetic_tabddpm.csv"]
    
    all_distances = {}
    
    for name, file in zip(models, files):
        print(f"Calculando DCR para {name}...")
        syn = pd.read_csv(f"{OUTPUT_DIR}/{file}")
        # Alinear columnas con real
        syn = syn[real.columns]
        
        syn_p, _, _ = preprocess_for_distance(syn, encoders, scaler)
        
        dists = calculate_dcr(real_p, syn_p)
        all_distances[name] = dists
        
        # Estadísticas básicas
        print(f"  - Distancia mínima: {np.min(dists):.6f}")
        print(f"  - Distancia media: {np.mean(dists):.6f}")
        print(f"  - % Registros con distancia < 0.01: {np.mean(dists < 0.01)*100:.4f}%")

    # Guardar resultados
    print("\nGenerando visualizaciones...")
    plt.figure(figsize=(12, 6))
    for name, dists in all_distances.items():
        sns.kdeplot(dists, label=name, fill=True, alpha=0.3)
    
    plt.title("Distribución de Distancia al Registro más Cercano (DCR)")
    plt.xlabel("Distancia Euclidiana al registro real más próximo")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{REPORT_DIR}/dcr_distribution.png")
    plt.close()
    
    # Guardar CSV de resumen
    summary = []
    for name, dists in all_distances.items():
        summary.append({
            "Model": name,
            "Min_DCR": np.min(dists),
            "Mean_DCR": np.mean(dists),
            "Median_DCR": np.median(dists),
            "Risk_Threshold_0.01_%": np.mean(dists < 0.01)*100
        })
    pd.DataFrame(summary).to_csv(f"{REPORT_DIR}/dcr_summary.csv", index=False)
    print(f"\nResumen guardado en {REPORT_DIR}/dcr_summary.csv")

if __name__ == "__main__":
    main()
