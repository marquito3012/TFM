import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import os

# Configuración de rutas
DATA_DIR = "/compartida/data"
OUTPUT_DIR = "/compartida/outputs"
REPORT_DIR = "/compartida/outputs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    real = pd.read_csv(f"{DATA_DIR}/diabetic_data_clean.csv")
    ctgan = pd.read_csv(f"{OUTPUT_DIR}/synthetic_ctgan.csv")
    tvae = pd.read_csv(f"{OUTPUT_DIR}/synthetic_tvae.csv")
    tabddpm = pd.read_csv(f"{OUTPUT_DIR}/synthetic_tabddpm.csv")
    return real, {"CTGAN": ctgan, "TVAE": tvae, "TabDDPM": tabddpm}

def calculate_wasserstein(real, synthetic_dict, num_cols):
    results = []
    for name, df in synthetic_dict.items():
        for col in num_cols:
            # Ensure both are numeric and drop NaNs
            r_vals = real[col].dropna()
            s_vals = df[col].dropna()
            dist = wasserstein_distance(r_vals, s_vals)
            results.append({"Model": name, "Column": col, "Wasserstein": dist})
    return pd.DataFrame(results)

def correlation_difference(real, synthetic_dict, num_cols):
    real_corr = real[num_cols].corr()
    diffs = {}
    for name, df in synthetic_dict.items():
        syn_corr = df[num_cols].corr()
        diff = np.abs(real_corr - syn_corr)
        diffs[name] = diff.mean().mean()
        
        # Plotting heatmap of differences
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, annot=True, cmap="YlOrRd", fmt=".2f")
        plt.title(f"Correlation Difference Heatmap: {name}")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/corr_diff_{name.lower()}.png")
        plt.close()
    return diffs

def main():
    print("Cargando datos...")
    real, synthetics = load_data()
    
    num_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Variables numéricas detectadas: {num_cols}")
    
    print("Calculando Distancia de Wasserstein...")
    wd_df = calculate_wasserstein(real, synthetics, num_cols)
    wd_summary = wd_df.groupby("Model")["Wasserstein"].mean().reset_index()
    print("\nResumen de Distancia de Wasserstein (menor es mejor):")
    print(wd_summary)
    wd_df.to_csv(f"{REPORT_DIR}/wasserstein_distances.csv", index=False)
    
    print("\nCalculando Diferencias de Correlación...")
    corr_diffs = correlation_difference(real, synthetics, num_cols)
    print("\nDiferencia Media de Correlación (MAE entre matrices):")
    for model, diff in corr_diffs.items():
        print(f"{model}: {diff:.4f}")

    # Guardar resumen final
    with open(f"{REPORT_DIR}/fidelity_summary.txt", "w") as f:
        f.write("RESUMEN DE FIDELIDAD ESTADISTICA\n")
        f.write("================================\n\n")
        f.write("Distancia de Wasserstein Media:\n")
        f.write(wd_summary.to_string())
        f.write("\n\nDiferencia Media de Correlacion:\n")
        for model, diff in corr_diffs.items():
            f.write(f"{model}: {diff:.4f}\n")

if __name__ == "__main__":
    main()
