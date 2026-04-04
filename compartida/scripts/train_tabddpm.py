"""
train_tabddpm.py
================
Implementación propia de TabDDPM (Tabular Denoising Diffusion Probabilistic Model)
sobre el dataset diabético limpio, construida sobre PyTorch puro.

Referencia:
    Kotelnikov et al. (2023) "TabDDPM: Modelling Tabular Data with Diffusion Models"
    ICML 2023. https://arxiv.org/abs/2209.15421

Arquitectura:
    ┌──────────────────────────────────────────┐
    │  Forward Process (q):                    │
    │    x_0 → x_1 → … → x_T  (añadir ruido)  │
    │                                          │
    │  Reverse Process (p_θ):                  │
    │    x_T → … → x_1 → x_0  (denoising MLP) │
    └──────────────────────────────────────────┘

Uso:
    python scripts/train_tabddpm.py
    python scripts/train_tabddpm.py --epochs 2000 --batch_size 2048
"""

import argparse
import os
import sys
import math
import time
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TABDDPM_CONFIG,
    N_SYNTHETIC_SAMPLES,
    RANDOM_SEED,
    SYNTH_TABDDPM_PATH,
    MODELS_DIR,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
)
from data_loader import load_clean_data


# ---------------------------------------------------------------------------
# ARGUMENTOS CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Entrena TabDDPM sobre el dataset diabético.")
    p.add_argument("--epochs",      type=int,   default=TABDDPM_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=TABDDPM_CONFIG["batch_size"])
    p.add_argument("--lr",          type=float, default=TABDDPM_CONFIG["lr"])
    p.add_argument("--T",           type=int,   default=TABDDPM_CONFIG["T"])
    p.add_argument("--n_samples",   type=int,   default=N_SYNTHETIC_SAMPLES)
    p.add_argument("--output",      type=str,   default=SYNTH_TABDDPM_PATH)
    p.add_argument("--no_cuda",     action="store_true")
    p.add_argument("--schedule",    type=str,   default=TABDDPM_CONFIG["schedule"],
                   choices=["linear", "cosine"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# PREPROCESAMIENTO PARA TabDDPM
# ---------------------------------------------------------------------------

class TabularPreprocessor:
    """
    Convierte el DataFrame mixto (numérico + categórico) en un tensor
    continuo para el proceso de difusión.

    - Numéricos: StandardScaler → float
    - Categóricos: OneHotEncoder → float (el modelo trabaja en el espacio continuo)
    
    El proceso inverso reconstruye el DataFrame original.
    """

    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self._scalers = {}
        self._ohe_categories = {}  # col → lista de categorías únicas
        self._ohe_offsets = {}
        self._input_dim = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        from sklearn.preprocessing import StandardScaler

        # Numéricos
        for col in self.num_cols:
            sc = StandardScaler()
            sc.fit(df[[col]].fillna(0.0))
            self._scalers[col] = sc

        # Categóricos
        offset = len(self.num_cols)
        for col in self.cat_cols:
            cats = sorted(df[col].astype(str).unique().tolist())
            self._ohe_categories[col] = cats
            self._ohe_offsets[col] = offset
            offset += len(cats)

        self._input_dim = offset
        print(f"[Preprocesador] Dimensión del vector latente: {self._input_dim}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        X = np.zeros((n, self._input_dim), dtype=np.float32)

        # Numéricos escalados
        for i, col in enumerate(self.num_cols):
            vals = df[col].fillna(0.0).values.reshape(-1, 1)
            X[:, i] = self._scalers[col].transform(vals).ravel()

        # Categóricos → OHE
        for col, cats in self._ohe_categories.items():
            off = self._ohe_offsets[col]
            for j, cat in enumerate(cats):
                mask = df[col].astype(str) == cat
                X[mask, off + j] = 1.0

        return X

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        result = {}

        # Numéricos
        for i, col in enumerate(self.num_cols):
            vals = self._scalers[col].inverse_transform(X[:, i].reshape(-1, 1)).ravel()
            result[col] = vals.round().astype(int)   # la mayoría son enteros

        # Categóricos: argmax sobre el bloque OHE
        for col, cats in self._ohe_categories.items():
            off = self._ohe_offsets[col]
            block = X[:, off: off + len(cats)]
            idx = np.argmax(block, axis=1)
            result[col] = [cats[i] for i in idx]

        return pd.DataFrame(result)

    @property
    def input_dim(self) -> int:
        return self._input_dim


# ---------------------------------------------------------------------------
# SCHEDULE DE RUIDO (β)
# ---------------------------------------------------------------------------

def make_beta_schedule(schedule: str, T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Devuelve el vector β_t para t=1…T."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)

    elif schedule == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999)

    else:
        raise ValueError(f"Schedule desconocido: {schedule}")


class DiffusionSchedule:
    """Pre-calcula todos los coeficientes derivados del schedule β."""

    def __init__(self, betas: torch.Tensor, device: torch.device):
        self.T = len(betas)
        betas = betas.to(device)

        self.betas        = betas
        self.alphas       = 1.0 - betas
        self.alphas_bar   = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab      = torch.sqrt(self.alphas_bar)
        self.sqrt_one_mab = torch.sqrt(1.0 - self.alphas_bar)

        # Para la reversa
        alphas_bar_prev       = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        self.posterior_var    = betas * (1.0 - alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.posterior_mean_c1 = betas * torch.sqrt(alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.posterior_mean_c2 = (1.0 - alphas_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Proceso forward: añade ruido a x_0 para obtener x_t."""
        sqrt_ab   = self.sqrt_ab[t].view(-1, 1)
        sqrt_1mab = self.sqrt_one_mab[t].view(-1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise


# ---------------------------------------------------------------------------
# RED NEURONAL DE DENOISING (ε-predictor MLP con embeddings de timestep)
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Embedding sinusoidal del timestep (igual que el Transformer original)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.embed_dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DenoisingMLP(nn.Module):
    """
    MLP que predice el ruido ε dado (x_t, t).
    
    Arquitectura:
        x_t → Linear → [ResBlock × N] → Linear → ε_hat
    Cada ResBlock incorpora el embedding del timestep vía suma aditiva.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 time_embed_dim: int = 128, dropout: float = 0.0):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # Capa de proyección de entrada
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Bloques residuales
        self.blocks = nn.ModuleList()
        self.time_projs = nn.ModuleList()
        for i, h in enumerate(hidden_dims):
            in_h  = hidden_dims[i]
            out_h = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[-1]
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(in_h),
                nn.Linear(in_h, out_h),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(out_h, out_h),
            ))
            self.time_projs.append(nn.Linear(time_embed_dim, out_h))

        # Capa de salida
        self.out = nn.Linear(hidden_dims[-1], input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_proj(x)

        for block, t_proj in zip(self.blocks, self.time_projs):
            h = h + block(h) + t_proj(t_emb)

        return self.out(h)


# ---------------------------------------------------------------------------
# LOOP DE ENTRENAMIENTO
# ---------------------------------------------------------------------------

def train_loop(
    model: DenoisingMLP,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    eval_every: int = 50,
) -> List[float]:

    losses = []
    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for (x0,) in loader:
            x0 = x0.to(device)
            B  = x0.size(0)

            # Muestrear t uniformemente en [0, T-1]
            t     = torch.randint(0, schedule.T, (B,), device=device)
            noise = torch.randn_like(x0)
            xt    = schedule.q_sample(x0, t, noise)

            # Predicción del ruido
            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping para estabilidad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if epoch % eval_every == 0 or epoch == 1:
            print(f"  Época {epoch:>5}/{epochs}  |  Loss MSE: {avg_loss:.6f}")

    return losses


# ---------------------------------------------------------------------------
# MUESTREO (REVERSE DIFFUSION)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(
    model: DenoisingMLP,
    schedule: DiffusionSchedule,
    n_samples: int,
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Genera n_samples filas mediante el proceso de difusión reversa (DDPM sampling).
    Parte de ruido puro x_T ~ N(0, I) y denoisea hasta x_0.
    """
    model.eval()
    x = torch.randn(n_samples, input_dim, device=device)

    for t_idx in reversed(range(schedule.T)):
        t_batch = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        # Predicción del ruido
        noise_pred = model(x, t_batch)

        # Coeficientes del paso
        beta_t      = schedule.betas[t_idx]
        alpha_t     = schedule.alphas[t_idx]
        alpha_bar_t = schedule.alphas_bar[t_idx]

        # Media del paso reverso
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(-3, 3)  # suavizamos los outliers

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )

        # Ruido adicional (solo si t > 0)
        if t_idx > 0:
            var   = schedule.posterior_var[t_idx].clamp(min=1e-20)
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean

    return x


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

def train(args) -> None:
    print("=" * 60)
    print("  FASE 3 — TabDDPM (Tabular Diffusion Model)")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Dispositivo
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    if device.type == "cuda":
        print(f"[GPU] Aceleración habilitada: {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] Entrenando en CPU. Considera usar GPU para > 500 épocas.")

    # 2. Cargar datos
    df, num_cols, cat_cols = load_clean_data(verbose=True)

    # 3. Preprocesar
    print("\n[Preprocesamiento] Ajustando transformadores…")
    preprocessor = TabularPreprocessor(num_cols=num_cols, cat_cols=cat_cols)
    preprocessor.fit(df)

    X = preprocessor.transform(df)
    X_tensor = torch.FloatTensor(X)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Preprocesamiento] X.shape: {X.shape}  |  Batches/época: {len(loader)}")

    # 4. Schedule de ruido
    betas    = make_beta_schedule(args.schedule, args.T,
                                  TABDDPM_CONFIG["beta_start"],
                                  TABDDPM_CONFIG["beta_end"])
    schedule = DiffusionSchedule(betas, device)
    print(f"\n[Difusión] Schedule '{args.schedule}' con T={args.T} pasos")
    print(f"           β_0={betas[0]:.5f}  β_T={betas[-1]:.5f}")

    # 5. Modelo
    denoiser = DenoisingMLP(
        input_dim=preprocessor.input_dim,
        hidden_dims=TABDDPM_CONFIG["hidden_dims"],
        dropout=TABDDPM_CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"\n[Modelo] Parámetros entrenables: {n_params:,}")

    # 6. Optimizador
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=args.lr,
        weight_decay=TABDDPM_CONFIG["weight_decay"],
    )
    # Scheduler cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # 7. Entrenamiento
    print(f"\n[Entrenamiento] Iniciando — {args.epochs} épocas, batch={args.batch_size}")
    t0 = time.time()
    losses = train_loop(
        model=denoiser,
        schedule=schedule,
        loader=loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        eval_every=TABDDPM_CONFIG["num_timesteps_eval"],
    )
    elapsed = time.time() - t0
    print(f"[Entrenamiento] Completado en {elapsed / 60:.1f} minutos.")

    # 8. Guardar modelo y preprocesador
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "tabddpm_model.pt")
    torch.save({
        "model_state_dict": denoiser.state_dict(),
        "preprocessor":     preprocessor,
        "schedule_betas":   betas.cpu(),
        "config":           TABDDPM_CONFIG,
        "args":             vars(args),
        "losses":           losses,
    }, model_path)
    print(f"[Guardado] Modelo en: {model_path}")

    # 9. Guardar curva de pérdida
    _save_loss_curve(losses)

    # 10. Generar datos sintéticos
    print(f"\n[Generación] Generando {args.n_samples:,} filas sintéticas…")
    t1 = time.time()
    X_synth = sample(denoiser, schedule, args.n_samples, preprocessor.input_dim, device)
    X_synth_np = X_synth.cpu().numpy()
    print(f"[Generación] Completada en {time.time() - t1:.1f}s")

    # 11. Transformación inversa → DataFrame
    synth_df = preprocessor.inverse_transform(X_synth_np)

    # 12. Guardar CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    synth_df.to_csv(args.output, index=False)
    print(f"[Guardado] Datos sintéticos en: {args.output}")
    print(f"           Shape: {synth_df.shape}")


def _save_loss_curve(losses: List[float]) -> None:
    """Guarda la curva de pérdida como CSV para graficarla en el notebook."""
    import os
    from config import OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_path = os.path.join(OUTPUT_DIR, "tabddpm_loss_curve.csv")
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "mse_loss": losses}).to_csv(
        loss_path, index=False
    )
    print(f"[Guardado] Curva de pérdida en: {loss_path}")


# ---------------------------------------------------------------------------
# CARGA Y MUESTREO DE MODELO YA ENTRENADO
# ---------------------------------------------------------------------------

def load_and_sample(n_samples: int = N_SYNTHETIC_SAMPLES, output: str = SYNTH_TABDDPM_PATH):
    """Carga un TabDDPM guardado y genera n_samples filas sin reentrenar."""
    model_path = os.path.join(MODELS_DIR, "tabddpm_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Entrena primero.")

    checkpoint    = torch.load(model_path, map_location="cpu")
    preprocessor  = checkpoint["preprocessor"]
    betas         = checkpoint["schedule_betas"]
    cfg           = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = DiffusionSchedule(betas, device)

    denoiser = DenoisingMLP(
        input_dim=preprocessor.input_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    ).to(device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])

    print(f"[Cargado] TabDDPM desde: {model_path}")
    X_synth  = sample(denoiser, schedule, n_samples, preprocessor.input_dim, device)
    synth_df = preprocessor.inverse_transform(X_synth.cpu().numpy())
    synth_df.to_csv(output, index=False)
    print(f"[Guardado] {n_samples:,} filas en: {output}")
    return synth_df


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
