"""
config.py
=========
Configuración centralizada para la Fase 3 del TFM.
Define rutas, hiperparámetros y metadatos del dataset.
"""

import os

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /compartida

DATA_DIR       = os.path.join(BASE_DIR, "data")
SCRIPTS_DIR    = os.path.join(BASE_DIR, "scripts")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")          # datos sintéticos generados
MODELS_DIR     = os.path.join(BASE_DIR, "models")           # checkpoints guardados
REPORTS_DIR    = os.path.join(BASE_DIR, "docs")

RAW_DATA_PATH   = os.path.join(DATA_DIR, "diabetic_data.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "diabetic_data_clean.csv")

# Ficheros de salida por modelo
SYNTH_CTGAN_PATH  = os.path.join(OUTPUT_DIR, "synthetic_ctgan.csv")
SYNTH_TVAE_PATH   = os.path.join(OUTPUT_DIR, "synthetic_tvae.csv")
SYNTH_TABDDPM_PATH = os.path.join(OUTPUT_DIR, "synthetic_tabddpm.csv")

# ---------------------------------------------------------------------------
# METADATOS DEL DATASET
# ---------------------------------------------------------------------------

# Variable objetivo
TARGET_COL = "readmitted"

# Columnas numéricas continuas/discretas
NUMERICAL_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "prior_visits",          # variable creada en Fase 2
]

# Columnas categóricas (incluye variables binarias derivadas y target)
CATEGORICAL_COLS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "any_med_change",        # variable creada en Fase 2
    "readmitted",
]

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — CTGAN
# ---------------------------------------------------------------------------
CTGAN_CONFIG = {
    "epochs":           300,
    "batch_size":       500,
    "generator_dim":    (256, 256),
    "discriminator_dim":(256, 256),
    "generator_lr":     2e-4,
    "discriminator_lr": 2e-4,
    "discriminator_steps": 1,
    "log_frequency":    True,
    "verbose":          True,
    "cuda":             True,         # se sobreescribe a False si no hay GPU
}

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — TVAE
# ---------------------------------------------------------------------------
TVAE_CONFIG = {
    "epochs":           300,
    "batch_size":       500,
    "compress_dims":    (128, 128),
    "decompress_dims":  (128, 128),
    "embedding_dim":    128,
    "l2scale":          1e-5,
    "loss_factor":      2,
    "cuda":             True,
}

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — TabDDPM  (implementación propia desde cero)
# ---------------------------------------------------------------------------
TABDDPM_CONFIG = {
    # Difusión
    "T":                1000,         # pasos de ruido
    "beta_start":       1e-4,
    "beta_end":         0.02,
    "schedule":         "linear",     # "linear" | "cosine"

    # Red neuronal (MLP de denoising)
    "hidden_dims":      [512, 512, 512],
    "dropout":          0.0,

    # Entrenamiento
    "epochs":           1000,
    "batch_size":       4096,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "num_timesteps_eval": 10,        # pasos de evaluación durante entrenamiento

    # Generación
    "n_samples":        99340,       # misma cantidad que el dataset real
}

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL DE GENERACIÓN
# ---------------------------------------------------------------------------
N_SYNTHETIC_SAMPLES = 99_340   # nº de filas sintéticas a generar (igual que el real)
RANDOM_SEED         = 42
