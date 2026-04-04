"""
run_all_models.py
=================
Script orquestador que entrena los tres modelos generativos secuencialmente
(CTGAN → TVAE → TabDDPM) y genera los datos sintéticos correspondientes.

Uso:
    python scripts/run_all_models.py
    python scripts/run_all_models.py --models ctgan tvae
    python scripts/run_all_models.py --models tabddpm --epochs 2000
    python scripts/run_all_models.py --quick   # 50 épocas para pruebas rápidas
"""

import argparse
import subprocess
import sys
import os
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON      = sys.executable   # mismo intérprete que está ejecutando este script


# ---------------------------------------------------------------------------
# ARGUMENTOS CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Orquestador de entrenamiento: CTGAN, TVAE, TabDDPM."
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["ctgan", "tvae", "tabddpm"],
        default=["ctgan", "tvae", "tabddpm"],
        help="Modelos a entrenar (por defecto: todos).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Épocas para TODOS los modelos (sobreescribe config.py).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: 50 épocas. Útil para validar que todo funciona.",
    )
    p.add_argument(
        "--no_cuda",
        action="store_true",
        help="Forzar CPU en todos los modelos.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

def run_model(script_name: str, extra_args: list) -> bool:
    """
    Ejecuta un script de entrenamiento como subproceso.
    Devuelve True si terminó con éxito, False si hubo error.
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [PYTHON, script_path] + extra_args

    print("\n" + "═" * 62)
    print(f"  Ejecutando: {' '.join(cmd)}")
    print("═" * 62 + "\n")

    t0  = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0

    if ret.returncode == 0:
        print(f"\n✅  {script_name} completado en {elapsed / 60:.1f} min.")
        return True
    else:
        print(f"\n❌  {script_name} falló (código {ret.returncode}).")
        return False


# ---------------------------------------------------------------------------
# PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    epochs = 50 if args.quick else args.epochs
    extra_global = []
    if args.no_cuda:
        extra_global.append("--no_cuda")

    MODEL_SCRIPTS = {
        "ctgan":   "train_ctgan.py",
        "tvae":    "train_tvae.py",
        "tabddpm": "train_tabddpm.py",
    }

    results = {}
    total_start = time.time()

    for model_name in args.models:
        script = MODEL_SCRIPTS[model_name]
        extra  = list(extra_global)
        if epochs is not None:
            extra += ["--epochs", str(epochs)]

        ok = run_model(script, extra)
        results[model_name] = "✅ OK" if ok else "❌ ERROR"

    total_elapsed = time.time() - total_start

    # Resumen final
    print("\n" + "═" * 62)
    print("  RESUMEN DE EJECUCIÓN")
    print("═" * 62)
    for model, status in results.items():
        print(f"  {model.upper():<10}  {status}")
    print(f"\n  Tiempo total: {total_elapsed / 60:.1f} minutos")
    print("═" * 62)

    # Verificar que los CSVs sintéticos existan
    from config import OUTPUT_DIR, SYNTH_CTGAN_PATH, SYNTH_TVAE_PATH, SYNTH_TABDDPM_PATH
    output_map = {
        "ctgan":   SYNTH_CTGAN_PATH,
        "tvae":    SYNTH_TVAE_PATH,
        "tabddpm": SYNTH_TABDDPM_PATH,
    }
    print("\n  Archivos de salida:")
    for model in args.models:
        path  = output_map[model]
        exists = "✅" if os.path.exists(path) else "❌ NO ENCONTRADO"
        print(f"  {model.upper():<10}  {exists}  {path}")


if __name__ == "__main__":
    main()
