"""
tvae_ablation.py
================
Estudio de ablación del modelo TVAE para el TFM.

Hipótesis investigadas (según feedback del tutor):
  1. ¿El loss_factor (peso reconstrucción vs KL) explica la caída del F1?
  2. ¿El embedding_dim insuficiente limita la capacidad del espacio latente?
  3. ¿Es el problema estructural del VAE ante distribuciones desbalanceadas?

Variantes entrenadas:
  v1_baseline → loss_factor=2,  embedding_dim=128  (configuración original)
  v2_lf5      → loss_factor=5,  embedding_dim=128  (más peso a reconstrucción)
  v3_lf10     → loss_factor=10, embedding_dim=128  (reconstrucción dominante)
  v4_ed256    → loss_factor=5,  embedding_dim=256  (mayor capacidad latente)

Evaluación por variante:
  - Wasserstein Media (fidelidad estadística)
  - Diferencia de correlación MAE (fidelidad estructural)
  - F1-Score TSTR con XGBoost (utilidad predictiva)
  - AUC-ROC TSTR (calibración probabilística)

Uso (desde el contenedor Docker):
    docker compose exec tfm python scripts/tvae_ablation.py
    docker compose exec tfm python scripts/tvae_ablation.py --skip_v1  # si v1 ya existe
    docker compose exec tfm python scripts/tvae_ablation.py --only_eval # solo evalúa, no reentrena

Salidas:
    /compartida/models/tvae_v*.pkl             — modelos entrenados
    /compartida/outputs/synthetic_tvae_v*.csv  — datos sintéticos por variante
    /compartida/outputs/reports/tvae_ablation_results.csv  — tabla comparativa
    /compartida/outputs/reports/tvae_ablation_report.md    — informe en markdown
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CLEAN_DATA_PATH,
    MODELS_DIR,
    NUMERICAL_COLS,
    N_SYNTHETIC_SAMPLES,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from data_loader import load_clean_data, get_metadata_sdv

# ---------------------------------------------------------------------------
# VARIANTES DE ABLACIÓN
# ---------------------------------------------------------------------------

VARIANTS = [
    {
        "id":            "v1_baseline",
        "label":         "V1 — Baseline (lf=2, ed=128)",
        "loss_factor":   2,
        "embedding_dim": 128,
        "compress_dims":   (128, 128),
        "decompress_dims": (128, 128),
        "epochs":        300,
        "batch_size":    500,
        "l2scale":       1e-5,
    },
    {
        "id":            "v2_lf5",
        "label":         "V2 — loss_factor=5, ed=128",
        "loss_factor":   5,
        "embedding_dim": 128,
        "compress_dims":   (128, 128),
        "decompress_dims": (128, 128),
        "epochs":        300,
        "batch_size":    500,
        "l2scale":       1e-5,
    },
    {
        "id":            "v3_lf10",
        "label":         "V3 — loss_factor=10, ed=128",
        "loss_factor":   10,
        "embedding_dim": 128,
        "compress_dims":   (128, 128),
        "decompress_dims": (128, 128),
        "epochs":        300,
        "batch_size":    500,
        "l2scale":       1e-5,
    },
    {
        "id":            "v4_ed256",
        "label":         "V4 — loss_factor=5, ed=256",
        "loss_factor":   5,
        "embedding_dim": 256,
        "compress_dims":   (256, 256),
        "decompress_dims": (256, 256),
        "epochs":        300,
        "batch_size":    500,
        "l2scale":       1e-5,
    },
]

REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# ---------------------------------------------------------------------------
# ARGUMENTOS CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ablación TVAE — TFM Marco Fernández")
    p.add_argument(
        "--skip_v1", action="store_true",
        help="Omite entrenar V1 (baseline) si ya existe el CSV y el modelo."
    )
    p.add_argument(
        "--only_eval", action="store_true",
        help="Solo ejecuta la evaluación (asume que todos los CSVs existen)."
    )
    p.add_argument(
        "--no_cuda", action="store_true",
        help="Fuerza CPU aunque haya GPU disponible."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# ENTRENAMIENTO DE UNA VARIANTE
# ---------------------------------------------------------------------------

def train_variant(variant: dict, df: pd.DataFrame, cuda: bool) -> str:
    """
    Entrena una variante de TVAE y guarda modelo + CSV sintético.
    Devuelve la ruta al CSV generado.
    """
    vid  = variant["id"]
    out_csv   = os.path.join(OUTPUT_DIR, f"synthetic_tvae_{vid}.csv")
    model_path = os.path.join(MODELS_DIR, f"tvae_{vid}.pkl")

    print(f"\n{'='*65}")
    print(f"  Entrenando: {variant['label']}")
    print(f"  loss_factor={variant['loss_factor']}  "
          f"embedding_dim={variant['embedding_dim']}  "
          f"épocas={variant['epochs']}")
    print(f"{'='*65}")

    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer

    # Metadatos SDV
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    sdv_meta = get_metadata_sdv(df)
    for col, props in sdv_meta["columns"].items():
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype=props["sdtype"])

    model = TVAESynthesizer(
        metadata=metadata,
        epochs=variant["epochs"],
        batch_size=variant["batch_size"],
        compress_dims=variant["compress_dims"],
        decompress_dims=variant["decompress_dims"],
        embedding_dim=variant["embedding_dim"],
        l2scale=variant["l2scale"],
        loss_factor=variant["loss_factor"],
        cuda=cuda,
        verbose=True,
    )

    t0 = time.time()
    model.fit(df)
    elapsed = time.time() - t0
    print(f"[OK] Entrenamiento completado en {elapsed / 60:.1f} min.")

    # Diagnóstico del espacio latente
    _diagnose_latent(model, df)

    # Guardar modelo
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    print(f"[OK] Modelo guardado en: {model_path}")

    # Generar y guardar CSV sintético
    print(f"[...] Generando {N_SYNTHETIC_SAMPLES:,} filas sintéticas...")
    t1 = time.time()
    synth = model.sample(num_rows=N_SYNTHETIC_SAMPLES)
    print(f"[OK] Generación en {time.time() - t1:.1f}s  →  {synth.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    synth.to_csv(out_csv, index=False)
    print(f"[OK] CSV guardado en: {out_csv}")

    return out_csv


def _diagnose_latent(model, df: pd.DataFrame) -> None:
    """Valida que μ≈0, σ≈1 en el espacio latente del encoder."""
    print("\n  [Diagnóstico latente]")
    try:
        encoder  = model._model.encoder
        device   = next(encoder.parameters()).device
        transformer = model._data_processor
        transformed = transformer.transform(df.sample(min(2000, len(df)), random_state=42))
        data_tensor = torch.FloatTensor(transformed.values).to(device)
        with torch.no_grad():
            mu, log_var = encoder(data_tensor)
        mu_np  = mu.cpu().numpy()
        std_np = torch.exp(0.5 * log_var).cpu().numpy()
        print(f"    μ  — media: {mu_np.mean():.4f}  std: {mu_np.std():.4f}  "
              f"(ideal ≈ 0.0)")
        print(f"    σ  — media: {std_np.mean():.4f}  std: {std_np.std():.4f}  "
              f"(ideal ≈ 1.0)")
        kl_collapse = std_np.mean() < 0.3
        if kl_collapse:
            print("    ⚠️  POSIBLE KL COLLAPSE: σ muy baja → el encoder ignora el input.")
        else:
            print("    ✅  Espacio latente con regularización correcta.")
    except Exception as e:
        print(f"    [INFO] Diagnóstico no disponible: {e}")


# ---------------------------------------------------------------------------
# EVALUACIÓN DE FIDELIDAD
# ---------------------------------------------------------------------------

def evaluate_fidelity(real: pd.DataFrame, synth: pd.DataFrame, label: str) -> dict:
    """Calcula Wasserstein media y diferencia de correlación."""
    num_cols = real.select_dtypes(include=[np.number]).columns.tolist()

    # Wasserstein
    wd_vals = []
    for col in num_cols:
        r = real[col].dropna()
        s = synth[col].dropna() if col in synth.columns else pd.Series(dtype=float)
        if len(s) > 0:
            wd_vals.append(wasserstein_distance(r, s))
    wd_mean = float(np.mean(wd_vals)) if wd_vals else float("nan")

    # Diferencia de correlación
    common_num = [c for c in num_cols if c in synth.columns]
    real_corr  = real[common_num].corr()
    synth_corr = synth[common_num].corr()
    corr_diff  = float(np.abs(real_corr - synth_corr).mean().mean())

    print(f"  Wasserstein media : {wd_mean:.4f}")
    print(f"  Diff. correlación : {corr_diff:.4f}")

    return {"wasserstein_mean": wd_mean, "corr_diff_mae": corr_diff}


# ---------------------------------------------------------------------------
# EVALUACIÓN DE UTILIDAD (TSTR)
# ---------------------------------------------------------------------------

def preprocess_tstr(df: pd.DataFrame, encoders: dict | None = None):
    """Label-encoding común para TSTR. Devuelve (df_enc, encoders)."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
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
                df[col] = df[col].astype(str).map(
                    lambda x, lc=le: x if x in lc.classes_ else lc.classes_[0]
                )
                df[col] = le.transform(df[col])
            elif col in df.columns:
                df[col] = 0
    return df, encoders


def evaluate_utility(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    X_test_real,
    y_test_real,
    feature_order: list,
    encoders: dict,
    label: str,
) -> dict:
    """Entrena XGBoost con sintéticos, evalúa en test real (TSTR)."""
    from xgboost import XGBClassifier

    # Preparar target binario en sintéticos
    synth = synth.copy()
    synth["target"] = (synth["readmitted"] != "NO").astype(int)
    synth = synth.drop(columns=["readmitted"])

    synth_enc, _ = preprocess_tstr(synth, encoders)

    # Alinear columnas
    missing_cols = [c for c in feature_order if c not in synth_enc.columns]
    for c in missing_cols:
        synth_enc[c] = 0
    X_train_syn = synth_enc[feature_order]
    y_train_syn = synth_enc["target"]

    model = XGBClassifier(
        n_estimators=100, random_state=RANDOM_SEED,
        use_label_encoder=False, eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train_syn, y_train_syn)

    y_pred = model.predict(X_test_real)
    y_prob = model.predict_proba(X_test_real)[:, 1]

    f1  = float(f1_score(y_test_real, y_pred))
    auc = float(roc_auc_score(y_test_real, y_prob))

    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    return {"f1_score": f1, "auc_roc": auc}


# ---------------------------------------------------------------------------
# GENERACIÓN DEL INFORME
# ---------------------------------------------------------------------------

def build_report(results: list, baseline_f1: float) -> pd.DataFrame:
    """Construye DataFrame comparativo con gap de F1."""
    df = pd.DataFrame(results)
    df["f1_gap_%"] = ((df["f1_score"] - baseline_f1) / baseline_f1 * 100).round(2)
    df["f1_gap_%"] = df["f1_gap_%"].map(lambda x: f"{x:+.2f}%")
    return df


def save_markdown(results_df: pd.DataFrame, baseline_f1: float) -> None:
    """Exporta informe en Markdown para incluir en la memoria."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "tvae_ablation_report.md")

    lines = [
        "# Estudio de Ablación — TVAE",
        "",
        "**Objetivo:** Determinar si la caída del F1-Score del TVAE (−41% respecto al baseline real) "
        "es atribuible a hiperparámetros sub-óptimos (`loss_factor`, `embedding_dim`) "
        "o si responde a una limitación estructural del modelo VAE frente a distribuciones de clases desbalanceadas.",
        "",
        "## Variantes evaluadas",
        "",
        "| ID | Descripción | `loss_factor` | `embedding_dim` |",
        "|---|---|---|---|",
        "| V1 | Baseline (configuración original) | 2 | 128 |",
        "| V2 | Mayor peso a reconstrucción | 5 | 128 |",
        "| V3 | Reconstrucción dominante | 10 | 128 |",
        "| V4 | Mayor capacidad latente + reconstrucción | 5 | 256 |",
        "",
        "## Resultados",
        "",
    ]

    # Tabla de resultados
    cols_display = ["label", "wasserstein_mean", "corr_diff_mae", "f1_score", "auc_roc", "f1_gap_%"]
    available = [c for c in cols_display if c in results_df.columns]
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
        "## Análisis e interpretación",
        "",
        "> **[COMPLETAR tras revisar los resultados]**",
        "",
        "### Hipótesis 1: `loss_factor` sub-óptimo",
        "Comparar V1 (lf=2) con V2 (lf=5) y V3 (lf=10). "
        "Si el F1 mejora significativamente con lf mayor, el ELBO estaba dominado por la "
        "regularización KL, forzando distribuciones latentes demasiado suaves que pierden "
        "discriminabilidad entre clases.",
        "",
        "### Hipótesis 2: `embedding_dim` insuficiente",
        "Comparar V2 (lf=5, ed=128) con V4 (lf=5, ed=256). "
        "Si V4 mejora sobre V2, el espacio latente de 128 dimensiones no tiene capacidad "
        "suficiente para representar la complejidad de 39 columnas mixtas.",
        "",
        "### Hipótesis 3: Limitación estructural",
        "Si ninguna variante supera el umbral del −5% de gap F1, la conclusión es que "
        "la arquitectura VAE tiene una limitación estructural ante este tipo de distribución: "
        "al generar desde una gaussiana isotrópica sin mecanismo condicional, colapsa la "
        "clase minoritaria `<30` (11.1%) hacia la moda de la distribución latente, "
        "perdiendo su señal discriminativa. Este fenómeno es conocido en la literatura "
        "(El Emam et al., 2020; Stadler et al., 2022) y es la razón por la que "
        "arquitecturas como CTGAN y TabDDPM incorporan mecanismos condicionales.",
        "",
        f"*Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Informe Markdown guardado en: {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  TFM — Estudio de Ablación TVAE")
    print("  Investigación del F1-Score −41% vs. Baseline Real")
    print("="*65)

    # GPU
    cuda = torch.cuda.is_available() and not args.no_cuda
    if cuda:
        print(f"\n[GPU] Aceleración habilitada: {torch.cuda.get_device_name(0)}")
    else:
        print("\n[CPU] Entrenando en CPU.")

    # Cargar dataset real
    print("\n[...] Cargando dataset real...")
    df_real, num_cols, cat_cols = load_clean_data(verbose=True)

    # --- Preparar test set real (fijo para todos los TSTR) ---
    df_tstr = df_real.copy()
    df_tstr["target"] = (df_tstr["readmitted"] != "NO").astype(int)
    df_tstr = df_tstr.drop(columns=["readmitted"])
    df_tstr_enc, encoders = preprocess_tstr(df_tstr)

    X_all = df_tstr_enc.drop(columns=["target"])
    y_all = df_tstr_enc["target"]
    _, X_test_real, _, y_test_real = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED
    )
    feature_order = X_test_real.columns.tolist()

    # --- Baseline TRTR (Real → Real) ---
    from xgboost import XGBClassifier
    print("\n[...] Calculando baseline TRTR (Real → Real)...")
    X_train_real = X_all.loc[~X_all.index.isin(X_test_real.index)]
    y_train_real = y_all.loc[X_train_real.index]
    baseline_model = XGBClassifier(
        n_estimators=100, random_state=RANDOM_SEED,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    )
    baseline_model.fit(X_train_real, y_train_real)
    y_pred_base = baseline_model.predict(X_test_real)
    y_prob_base = baseline_model.predict_proba(X_test_real)[:, 1]
    baseline_f1  = float(f1_score(y_test_real, y_pred_base))
    baseline_auc = float(roc_auc_score(y_test_real, y_prob_base))
    print(f"  Baseline F1-Score : {baseline_f1:.4f}")
    print(f"  Baseline AUC-ROC  : {baseline_auc:.4f}")

    results = [{
        "id":               "baseline_trtr",
        "label":            "Baseline TRTR (Real → Real)",
        "loss_factor":      "—",
        "embedding_dim":    "—",
        "wasserstein_mean": 0.0,
        "corr_diff_mae":    0.0,
        "f1_score":         baseline_f1,
        "auc_roc":          baseline_auc,
    }]

    # ---------------------------------------------------------------------------
    # BUCLE PRINCIPAL: ENTRENAR + EVALUAR CADA VARIANTE
    # ---------------------------------------------------------------------------

    for variant in VARIANTS:
        vid     = variant["id"]
        out_csv = os.path.join(OUTPUT_DIR, f"synthetic_tvae_{vid}.csv")

        # ---- FASE 1: ENTRENAMIENTO (o reutilizar CSV existente) ----
        if args.only_eval:
            if not os.path.exists(out_csv):
                print(f"\n[SKIP] {vid}: CSV no encontrado en {out_csv}, omitiendo.")
                continue
            print(f"\n[LOAD] {vid}: usando CSV existente → {out_csv}")

        elif vid == "v1_baseline" and args.skip_v1:
            # Reutilizar el CSV original si existe
            original_csv = os.path.join(OUTPUT_DIR, "synthetic_tvae.csv")
            if os.path.exists(original_csv) and not os.path.exists(out_csv):
                import shutil
                shutil.copy(original_csv, out_csv)
                print(f"\n[COPY] v1_baseline: copiado desde {original_csv}")
            elif os.path.exists(out_csv):
                print(f"\n[SKIP] v1_baseline: CSV ya existe en {out_csv}")
            else:
                print(f"\n[WARN] v1_baseline: no se encontró CSV original. Entrenando...")
                train_variant(variant, df_real, cuda)
        else:
            train_variant(variant, df_real, cuda)

        # ---- FASE 2: EVALUACIÓN ----
        if not os.path.exists(out_csv):
            print(f"\n[ERROR] CSV no encontrado tras entrenamiento: {out_csv}")
            continue

        print(f"\n[EVAL] Evaluando {variant['label']}...")
        synth = pd.read_csv(out_csv)

        print("  [Fidelidad]")
        fidelity = evaluate_fidelity(df_real, synth, variant["label"])

        print("  [Utilidad TSTR]")
        utility = evaluate_utility(
            df_real, synth,
            X_test_real, y_test_real,
            feature_order, encoders,
            variant["label"],
        )

        results.append({
            "id":               vid,
            "label":            variant["label"],
            "loss_factor":      variant["loss_factor"],
            "embedding_dim":    variant["embedding_dim"],
            **fidelity,
            **utility,
        })

    # ---------------------------------------------------------------------------
    # INFORME FINAL
    # ---------------------------------------------------------------------------

    results_df = build_report(results, baseline_f1)

    print("\n" + "="*65)
    print("  RESULTADOS FINALES DEL ESTUDIO DE ABLACIÓN")
    print("="*65)
    print(results_df.to_string(index=False))

    # Guardar CSV de resultados
    csv_path = os.path.join(REPORT_DIR, "tvae_ablation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Tabla de resultados guardada en: {csv_path}")

    # Guardar informe Markdown
    save_markdown(results_df, baseline_f1)

    print("\n" + "="*65)
    print("  ABLACIÓN COMPLETADA")
    print("="*65)


if __name__ == "__main__":
    main()
