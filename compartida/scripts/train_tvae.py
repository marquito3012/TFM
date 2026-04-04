"""
train_tvae.py
=============
Entrenamiento del modelo TVAE (Tabular Variational Autoencoder) sobre el
dataset diabético limpio.

Referencia:
    Xu et al. (2019) "Modeling Tabular data using Conditional GAN"  
    (TVAE se presenta en el mismo paper que CTGAN como baseline.)
    NeurIPS 2019. https://arxiv.org/abs/1907.00503

Uso desde el contenedor Docker:
    python scripts/train_tvae.py
    python scripts/train_tvae.py --epochs 500 --embedding_dim 256
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TVAE_CONFIG,
    N_SYNTHETIC_SAMPLES,
    RANDOM_SEED,
    SYNTH_TVAE_PATH,
    MODELS_DIR,
)
from data_loader import load_clean_data, get_metadata_sdv


# ---------------------------------------------------------------------------
# ARGUMENTOS CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena TVAE sobre el dataset diabético limpio."
    )
    parser.add_argument("--epochs",        type=int, default=TVAE_CONFIG["epochs"])
    parser.add_argument("--batch_size",    type=int, default=TVAE_CONFIG["batch_size"])
    parser.add_argument("--embedding_dim", type=int, default=TVAE_CONFIG["embedding_dim"])
    parser.add_argument("--n_samples",     type=int, default=N_SYNTHETIC_SAMPLES)
    parser.add_argument("--output",        type=str, default=SYNTH_TVAE_PATH)
    parser.add_argument("--no_cuda",       action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------

def train(args) -> None:
    print("=" * 60)
    print("  FASE 3 — TVAE (Tabular Variational Autoencoder)")
    print("=" * 60)

    # 1. Cargar datos
    df, num_cols, cat_cols = load_clean_data(verbose=True)

    # 2. GPU
    cuda_available = torch.cuda.is_available() and not args.no_cuda
    if cuda_available:
        print(f"[GPU] Aceleración habilitada: {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] Entrenando en CPU.")

    # 3. Metadatos SDV
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    sdv_meta = get_metadata_sdv(df)
    for col, props in sdv_meta["columns"].items():
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype=props["sdtype"])

    # 4. Instanciar TVAE
    from sdv.single_table import TVAESynthesizer

    model = TVAESynthesizer(
        metadata=metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        compress_dims=TVAE_CONFIG["compress_dims"],
        decompress_dims=TVAE_CONFIG["decompress_dims"],
        embedding_dim=args.embedding_dim,
        l2scale=TVAE_CONFIG["l2scale"],
        loss_factor=TVAE_CONFIG["loss_factor"],
        cuda=cuda_available,
        verbose=True,
    )

    # 5. Entrenar
    print(f"\n[Entrenamiento] Iniciando — {args.epochs} épocas, batch={args.batch_size}")
    t0 = time.time()
    model.fit(df)
    elapsed = time.time() - t0
    print(f"[Entrenamiento] Completado en {elapsed / 60:.1f} minutos.")

    # 6. Guardar modelo
    model_path = os.path.join(MODELS_DIR, "tvae_model.pkl")
    model.save(model_path)
    print(f"[Guardado] Modelo en: {model_path}")

    # 7. Generar datos sintéticos
    print(f"\n[Generación] Generando {args.n_samples:,} filas sintéticas…")
    t1 = time.time()
    synthetic_df = model.sample(num_rows=args.n_samples)
    print(f"[Generación] Completada en {time.time() - t1:.1f}s")

    # 8. Guardar CSV sintético
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    synthetic_df.to_csv(args.output, index=False)
    print(f"[Guardado] Datos sintéticos en: {args.output}")
    print(f"           Shape: {synthetic_df.shape}")

    # 9. Diagnóstico rápido del espacio latente
    _diagnose_latent_space(model, df)


def _diagnose_latent_space(model, df):
    """
    Valida que el espacio latente aprendido tenga propiedades
    gaussianas plausibles (media ≈ 0, std ≈ 1).
    """
    print("\n--- Diagnóstico del Espacio Latente (TVAE) ---")
    try:
        import numpy as np
        # Accedemos al encoder interno de SDV/TVAE
        encoder = model._model.encoder
        device  = next(encoder.parameters()).device

        from sdv.single_table import TVAESynthesizer
        # Transformamos una muestra del dataset real al espacio latente
        transformer = model._data_processor
        transformed = transformer.transform(df.sample(2000, random_state=42))
        data_tensor = torch.FloatTensor(transformed.values).to(device)

        with torch.no_grad():
            mu, log_var = encoder(data_tensor)

        mu_np  = mu.cpu().numpy()
        std_np = torch.exp(0.5 * log_var).cpu().numpy()

        print(f"  μ  — media: {mu_np.mean():.4f}  |  std global: {mu_np.std():.4f}")
        print(f"  σ  — media: {std_np.mean():.4f}  |  std global: {std_np.std():.4f}")
        print("  (Ideal: μ≈0, σ≈1 — indican regularización latente correcta)")
    except Exception as e:
        print(f"  [INFO] Diagnóstico latente no disponible: {e}")


# ---------------------------------------------------------------------------
# CARGAR MODELO Y GENERAR
# ---------------------------------------------------------------------------

def load_and_sample(n_samples: int = N_SYNTHETIC_SAMPLES, output: str = SYNTH_TVAE_PATH):
    from sdv.single_table import TVAESynthesizer

    model_path = os.path.join(MODELS_DIR, "tvae_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Entrena primero.")

    model = TVAESynthesizer.load(model_path)
    print(f"[Cargado] Modelo TVAE desde: {model_path}")

    synthetic_df = model.sample(num_rows=n_samples)
    synthetic_df.to_csv(output, index=False)
    print(f"[Guardado] {n_samples:,} filas en: {output}")
    return synthetic_df


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
