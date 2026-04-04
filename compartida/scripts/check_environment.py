"""
check_environment.py
====================
Verifica el entorno de ejecución antes de comenzar el entrenamiento:
  - Versiones de librerías clave
  - Disponibilidad de GPU (CUDA / ROCm)
  - Existencia de los datasets
  - Integridad básica del CSV limpio

Uso:
    python scripts/check_environment.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_python():
    print(f"Python            : {sys.version.split()[0]}")
    assert sys.version_info >= (3, 10), "Se requiere Python >= 3.10"


def check_libraries():
    libs = {
        "torch":       "PyTorch",
        "pandas":      "pandas",
        "numpy":       "numpy",
        "sklearn":     "scikit-learn",
        "sdv":         "SDV",
        "ctgan":       "CTGAN",
        "rdt":         "RDT",
        "matplotlib":  "matplotlib",
        "seaborn":     "seaborn",
    }
    all_ok = True
    for module, label in libs.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✅  {label:<18}: {ver}")
        except ImportError:
            print(f"  ❌  {label:<18}: NO INSTALADA")
            all_ok = False
    return all_ok


def check_gpu():
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅  GPU detectada : {name}  ({mem_gb:.1f} GB VRAM)")
    else:
        print("  ⚠️   GPU no disponible — se usará CPU (entrenamiento más lento)")
    return cuda


def check_datasets():
    from config import CLEAN_DATA_PATH, RAW_DATA_PATH

    for path, label in [(RAW_DATA_PATH, "Raw CSV"), (CLEAN_DATA_PATH, "Clean CSV")]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  ✅  {label:<12}: {path}  ({size_mb:.1f} MB)")
        else:
            print(f"  ❌  {label:<12}: NO ENCONTRADO → {path}")


def check_csv_integrity():
    import pandas as pd
    from config import CLEAN_DATA_PATH

    if not os.path.exists(CLEAN_DATA_PATH):
        print("  ⚠️   Dataset limpio no disponible para verificar integridad.")
        return

    df = pd.read_csv(CLEAN_DATA_PATH, nrows=5)
    print(f"  ✅  Columnas detectadas ({len(df.columns)}): {list(df.columns[:6])} …")

    import pandas as _pd
    df_full = _pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    print(f"  ✅  Total de filas: {len(df_full):,}")

    null_total = df_full.isnull().sum().sum()
    if null_total > 0:
        print(f"  ⚠️   Se detectaron {null_total} valores nulos en el dataset.")
    else:
        print("  ✅  Sin valores nulos detectados.")


def check_output_dirs():
    from config import OUTPUT_DIR, MODELS_DIR
    for d, label in [(OUTPUT_DIR, "outputs/"), (MODELS_DIR, "models/")]:
        os.makedirs(d, exist_ok=True)
        print(f"  ✅  Directorio listo: {d}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFICACIÓN DEL ENTORNO — TFM Fase 3")
    print("=" * 60)

    print("\n🐍 Python:")
    check_python()

    print("\n📦 Librerías:")
    libs_ok = check_libraries()

    print("\n🖥️  GPU:")
    check_gpu()

    print("\n📂 Datasets:")
    check_datasets()

    print("\n🔍 Integridad del CSV limpio:")
    check_csv_integrity()

    print("\n📁 Directorios de salida:")
    check_output_dirs()

    print("\n" + "=" * 60)
    if libs_ok:
        print("✅  Entorno listo para iniciar el entrenamiento.")
        print("   Ejecuta: python scripts/run_all_models.py --quick")
    else:
        print("❌  Hay librerías faltantes. Ejecuta: pip install -r requirements.txt")
    print("=" * 60)
