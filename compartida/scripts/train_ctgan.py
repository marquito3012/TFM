"""
train_ctgan.py
==============
Entrenamiento del modelo CTGAN (Conditional Tabular GAN) sobre el dataset
diabético limpio.

Referencia:
    Xu et al. (2019) "Modeling Tabular data using Conditional GAN"
    NeurIPS 2019. https://arxiv.org/abs/1907.00503

Uso desde el contenedor Docker:
    python scripts/train_ctgan.py
    python scripts/train_ctgan.py --epochs 500 --batch_size 1000
"""

import argparse
import os
import sys
import time

import torch

# Permite ejecutar el script desde /compartida o desde /compartida/scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CTGAN_CONFIG,
    N_SYNTHETIC_SAMPLES,
    RANDOM_SEED,
    SYNTH_CTGAN_PATH,
    MODELS_DIR,
)
from data_loader import load_clean_data, get_metadata_sdv


# ---------------------------------------------------------------------------
# ARGUMENTOS CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena CTGAN sobre el dataset diabético limpio."
    )
    parser.add_argument("--epochs",      type=int,   default=CTGAN_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=CTGAN_CONFIG["batch_size"])
    parser.add_argument("--n_samples",   type=int,   default=N_SYNTHETIC_SAMPLES,
                        help="Número de filas sintéticas a generar.")
    parser.add_argument("--output",      type=str,   default=SYNTH_CTGAN_PATH,
                        help="Ruta de salida del CSV sintético.")
    parser.add_argument("--no_cuda",     action="store_true",
                        help="Fuerza ejecución en CPU aunque haya GPU disponible.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------

def train(args) -> None:
    # 1. Cargar datos
    print("=" * 60)
    print("  FASE 3 — CTGAN (Conditional Tabular GAN)")
    print("=" * 60)

    df, num_cols, cat_cols = load_clean_data(verbose=True)

    # 2. Detectar GPU
    cuda_available = torch.cuda.is_available() and not args.no_cuda
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"[GPU] Aceleración habilitada: {device_name}")
    else:
        print("[CPU] No se detectó GPU o se desactivó con --no_cuda. Entrenando en CPU.")

    # 3. Construir metadatos SDV
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Sobreescribimos los tipos para asegurar coherencia con nuestra config
    sdv_meta = get_metadata_sdv(df)
    for col, props in sdv_meta["columns"].items():
        if col in df.columns:
            metadata.update_column(
                column_name=col,
                sdtype=props["sdtype"]
            )

    # 4. Instanciar CTGAN
    from sdv.single_table import CTGANSynthesizer

    model = CTGANSynthesizer(
        metadata=metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        generator_dim=CTGAN_CONFIG["generator_dim"],
        discriminator_dim=CTGAN_CONFIG["discriminator_dim"],
        generator_lr=CTGAN_CONFIG["generator_lr"],
        discriminator_lr=CTGAN_CONFIG["discriminator_lr"],
        discriminator_steps=CTGAN_CONFIG["discriminator_steps"],
        log_frequency=CTGAN_CONFIG["log_frequency"],
        verbose=CTGAN_CONFIG["verbose"],
        cuda=cuda_available,
    )

    # 5. Entrenar
    print(f"\n[Entrenamiento] Iniciando — {args.epochs} épocas, batch={args.batch_size}")
    t0 = time.time()
    model.fit(df)
    elapsed = time.time() - t0
    print(f"[Entrenamiento] Completado en {elapsed / 60:.1f} minutos.")

    # 6. Guardar modelo
    model_path = os.path.join(MODELS_DIR, "ctgan_model.pkl")
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


# ---------------------------------------------------------------------------
# CARGAR MODELO YA ENTRENADO Y GENERAR MUESTRAS ADICIONALES
# ---------------------------------------------------------------------------

def load_and_sample(n_samples: int = N_SYNTHETIC_SAMPLES, output: str = SYNTH_CTGAN_PATH):
    """
    Carga un modelo CTGAN guardado previamente y genera n_samples filas.
    Útil para re-generar sin volver a entrenar.
    """
    from sdv.single_table import CTGANSynthesizer

    model_path = os.path.join(MODELS_DIR, "ctgan_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Entrena primero.")

    model = CTGANSynthesizer.load(model_path)
    print(f"[Cargado] Modelo CTGAN desde: {model_path}")

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
