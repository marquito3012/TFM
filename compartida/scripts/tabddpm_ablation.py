"""
tabddpm_ablation.py
===================
Estudio de ablación del modelo TabDDPM para el TFM.

Variantes entrenadas:
  v1_baseline → schedule=linear, T=1000 (ya completado)
  v2_cosine   → schedule=cosine, T=1000
  v3_lin500   → schedule=linear, T=500
  v4_cos500   → schedule=cosine, T=500

Uso:
    docker compose exec tfm python scripts/tabddpm_ablation.py
"""

import os
import sys
import time
import subprocess
import argparse

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, OUTPUT_DIR, RANDOM_SEED
from data_loader import load_clean_data
from tvae_ablation import evaluate_fidelity, evaluate_utility, preprocess_tstr

VARIANTS = [
    {
        "id": "v1_baseline",
        "label": "V1 — Baseline (Linear, T=1000)",
        "schedule": "linear",
        "T": 1000
    },
    {
        "id": "v2_cosine",
        "label": "V2 — Cosine, T=1000",
        "schedule": "cosine",
        "T": 1000
    },
    {
        "id": "v3_lin500",
        "label": "V3 — Linear, T=500",
        "schedule": "linear",
        "T": 500
    },
    {
        "id": "v4_cos500",
        "label": "V4 — Cosine, T=500",
        "schedule": "cosine",
        "T": 500
    }
]

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

def build_report(results: list, baseline_f1: float) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["f1_gap_%"] = ((df["f1_score"] - baseline_f1) / baseline_f1 * 100).round(2)
    df["f1_gap_%"] = df["f1_gap_%"].map(lambda x: f"{x:+.2f}%")
    return df

def save_markdown(results_df: pd.DataFrame, baseline_f1: float) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "tabddpm_ablation_report.md")

    lines = [
        "# Estudio de Ablación — TabDDPM",
        "",
        "**Objetivo:** Evaluar el impacto de la parametrización del proceso de difusión (`schedule` y pasos `T`) en la calidad y utilidad predictiva de los datos sintéticos.",
        "",
        "## Variantes evaluadas",
        "",
        "| ID | Descripción | `schedule` | `T` (pasos) |",
        "|---|---|---|---|",
        "| V1 | Baseline (configuración original) | lineal | 1000 |",
        "| V2 | Schedule Coseno | coseno | 1000 |",
        "| V3 | Reducción de coste (Lineal) | lineal | 500 |",
        "| V4 | Reducción de coste (Coseno) | coseno | 500 |",
        "",
        "## Resultados",
        "",
    ]

    cols_display = ["label", "wasserstein_mean", "corr_diff_mae", "f1_score", "auc_roc", "f1_gap_%"]
    header = "| Variante | Wasserstein ↓ | Corr. Diff ↓ | F1-Score | AUC-ROC | Gap F1 vs. Baseline |"
    sep    = "|---|---|---|---|---|---|"
    lines += [header, sep]
    for _, row in results_df.iterrows():
        vals = [
            str(row.get("label", "")),
            f"{row.get('wasserstein_mean', float('nan')):.4f}",
            f"{row.get('corr_diff_mae', float('nan')):.4f}",
            f"{row.get('f1_score', float('nan')):.4f}",
            f"{row.get('auc_roc', float('nan')):.4f}",
            str(row.get("f1_gap_%", "—")),
        ]
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Conclusiones Esperadas",
        "- **Impacto del Schedule Coseno:** Suele ralentizar la destrucción de información en pasos iniciales, lo que permite a la red de Denoising preservar detalles finos (mejorando potencialmente la fidelidad).",
        "- **Impacto de T=500:** Reduce a la mitad el coste de inferencia (generación), pero una caída grande en TSTR significaría que se han degradado las fronteras de decisión.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Informe Markdown guardado en: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_eval", action="store_true", help="Solo evaluar, no entrenar.")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("\n" + "="*65)
    print("  TFM — Estudio de Ablación TabDDPM")
    print("="*65)

    df_real, num_cols, cat_cols = load_clean_data(verbose=False)

    df_tstr = df_real.copy()
    df_tstr["target"] = (df_tstr["readmitted"] != "NO").astype(int)
    df_tstr = df_tstr.drop(columns=["readmitted"])
    df_tstr_enc, encoders = preprocess_tstr(df_tstr)

    X_all = df_tstr_enc.drop(columns=["target"])
    y_all = df_tstr_enc["target"]
    _, X_test_real, _, y_test_real = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_SEED)
    feature_order = X_test_real.columns.tolist()

    from xgboost import XGBClassifier
    print("\n[...] Calculando baseline TRTR (Real → Real)...")
    X_train_real = X_all.loc[~X_all.index.isin(X_test_real.index)]
    y_train_real = y_all.loc[X_train_real.index]
    baseline_model = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    baseline_model.fit(X_train_real, y_train_real)
    baseline_f1  = float(f1_score(y_test_real, baseline_model.predict(X_test_real)))
    baseline_auc = float(roc_auc_score(y_test_real, baseline_model.predict_proba(X_test_real)[:, 1]))
    
    results = [{
        "id": "baseline_trtr",
        "label": "Baseline TRTR",
        "wasserstein_mean": 0.0,
        "corr_diff_mae": 0.0,
        "f1_score": baseline_f1,
        "auc_roc": baseline_auc,
    }]

    for variant in VARIANTS:
        vid = variant["id"]
        out_csv = os.path.join(OUTPUT_DIR, f"synthetic_tabddpm_{vid}.csv")
        out_model = os.path.join(MODELS_DIR, f"tabddpm_model_{vid}.pt")

        if vid == "v1_baseline":
            original_csv = os.path.join(OUTPUT_DIR, "synthetic_tabddpm.csv")
            if not os.path.exists(out_csv) and os.path.exists(original_csv):
                import shutil
                shutil.copy(original_csv, out_csv)
            if not os.path.exists(out_csv):
                print(f"[!] Falta {out_csv}. Debes entrenar el baseline primero.")
                continue
        else:
            if not args.only_eval and not os.path.exists(out_csv):
                print(f"\n[+] Entrenando {variant['label']}...")
                cmd = [
                    "python", "scripts/train_tabddpm.py",
                    "--schedule", variant["schedule"],
                    "--T", str(variant["T"]),
                    "--output", out_csv,
                    "--model_output", out_model
                ]
                subprocess.run(cmd, check=True)
            elif not os.path.exists(out_csv):
                print(f"[SKIP] {out_csv} no encontrado.")
                continue

        print(f"\n[EVAL] Evaluando {variant['label']}...")
        synth = pd.read_csv(out_csv)
        print("  [Fidelidad]")
        fidelity = evaluate_fidelity(df_real, synth, variant["label"])
        print("  [Utilidad]")
        utility = evaluate_utility(df_real, synth, X_test_real, y_test_real, feature_order, encoders, variant["label"])

        results.append({
            "id": vid,
            "label": variant["label"],
            **fidelity,
            **utility,
        })

    results_df = build_report(results, baseline_f1)
    
    print("\n" + "="*65)
    print("  RESULTADOS ABLACIÓN TABDDPM")
    print("="*65)
    print(results_df.to_string(index=False))

    csv_path = os.path.join(REPORT_DIR, "tabddpm_ablation_results.csv")
    results_df.to_csv(csv_path, index=False)
    save_markdown(results_df, baseline_f1)

if __name__ == "__main__":
    main()
