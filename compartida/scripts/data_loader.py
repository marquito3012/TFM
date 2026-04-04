"""
data_loader.py
==============
Carga, valida e informa sobre el dataset limpio.
Devuelve el DataFrame listo para ser consumido por cada modelo generativo.

Uso:
    from scripts.data_loader import load_clean_data, get_metadata_sdv
    df, num_cols, cat_cols = load_clean_data()
"""

import os
import pandas as pd
import numpy as np

from config import (
    CLEAN_DATA_PATH,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    OUTPUT_DIR,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    for d in [OUTPUT_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que las columnas numéricas sean float32
    y las categóricas, object (string).
    """
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def _validate(df: pd.DataFrame) -> None:
    """Lanza avisos si hay valores nulos o columnas inesperadas."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        print("[AVISO] Columnas con NaN detectadas:")
        print(cols_with_nulls.to_string())
    else:
        print("[OK] No se detectaron NaN en el dataset.")

    expected = set(NUMERICAL_COLS + CATEGORICAL_COLS)
    actual = set(df.columns)
    missing = expected - actual
    extra   = actual - expected
    if missing:
        print(f"[AVISO] Columnas esperadas no encontradas: {missing}")
    if extra:
        print(f"[INFO]  Columnas adicionales en el CSV:    {extra}")


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

def load_clean_data(
    path: str = CLEAN_DATA_PATH,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Carga el dataset limpio y devuelve (df, numerical_cols, categorical_cols).

    Returns
    -------
    df : pd.DataFrame
        Dataset completo listo para los modelos generativos.
    num_cols : list[str]
        Columnas numéricas presentes en el DataFrame.
    cat_cols : list[str]
        Columnas categóricas presentes en el DataFrame.
    """
    _ensure_dirs()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset limpio no encontrado en: {path}\n"
            "Ejecuta primero el notebook 02_limpieza_ingenieria.ipynb"
        )

    df = pd.read_csv(path, low_memory=False)

    if verbose:
        print(f"[OK] Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"     Ruta: {path}")

    df = _cast_types(df)

    if verbose:
        _validate(df)
        print("\n--- Distribución de la variable objetivo ---")
        print(df[TARGET_COL].value_counts(normalize=True).mul(100).round(2).to_string())

    # Filtramos a las columnas que existan en el DataFrame
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    return df, num_cols, cat_cols


# ---------------------------------------------------------------------------
# METADATOS PARA SDV  (usado por CTGAN y TVAE)
# ---------------------------------------------------------------------------

def get_metadata_sdv(df: pd.DataFrame) -> dict:
    """
    Construye el diccionario de metadatos compatible con SDV / CTGAN / TVAE.

    Returns
    -------
    dict con estructura:
        {
          "columns": {
              "col_name": {"sdtype": "numerical" | "categorical"},
              ...
          },
          "primary_key": None
        }
    """
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    columns_meta = {}
    for col in num_cols:
        columns_meta[col] = {"sdtype": "numerical"}
    for col in cat_cols:
        columns_meta[col] = {"sdtype": "categorical"}

    metadata = {
        "columns": columns_meta,
        "primary_key": None,
    }
    return metadata


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA RÁPIDO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, num_cols, cat_cols = load_clean_data(verbose=True)
    print(f"\nColumnas numéricas  ({len(num_cols)}): {num_cols}")
    print(f"Columnas categóricas({len(cat_cols)}): {cat_cols}")
