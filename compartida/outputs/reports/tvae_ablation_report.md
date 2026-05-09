# Estudio de Ablación — TVAE

**Objetivo:** Determinar si la caída del F1-Score del TVAE (−41% respecto al baseline real) es atribuible a hiperparámetros sub-óptimos (`loss_factor`, `embedding_dim`) o si responde a una limitación estructural del modelo VAE frente a distribuciones de clases desbalanceadas.

## Variantes evaluadas

| ID | Descripción | `loss_factor` | `embedding_dim` |
|---|---|---|---|
| V1 | Baseline (configuración original) | 2 | 128 |
| V2 | Mayor peso a reconstrucción | 5 | 128 |
| V3 | Reconstrucción dominante | 10 | 128 |
| V4 | Mayor capacidad latente + reconstrucción | 5 | 256 |

## Resultados

| Variante | Wasserstein ↓ | Corr. Diff ↓ | F1-Score | AUC-ROC | Gap F1 vs. Baseline |
|---|---|---|---|---|---|
| Baseline TRTR (Real → Real) | 0.0000 | 0.0000 | 0.5906 | 0.6879 | +0.00% |
| V1 — Baseline (lf=2, ed=128) | 0.7265 | 0.0491 | 0.4486 | 0.6660 | -24.04% |
| V2 — loss_factor=5, ed=128 | 1.0691 | 0.0627 | 0.4234 | 0.5533 | -28.31% |
| V3 — loss_factor=10, ed=128 | 1.1938 | 0.0553 | 0.4795 | 0.5983 | -18.81% |
| V4 — loss_factor=5, ed=256 | 0.5290 | 0.0584 | 0.3386 | 0.6279 | -42.67% |

## Análisis e interpretación

> **[COMPLETAR tras revisar los resultados]**

### Hipótesis 1: `loss_factor` sub-óptimo
Comparar V1 (lf=2) con V2 (lf=5) y V3 (lf=10). Si el F1 mejora significativamente con lf mayor, el ELBO estaba dominado por la regularización KL, forzando distribuciones latentes demasiado suaves que pierden discriminabilidad entre clases.

### Hipótesis 2: `embedding_dim` insuficiente
Comparar V2 (lf=5, ed=128) con V4 (lf=5, ed=256). Si V4 mejora sobre V2, el espacio latente de 128 dimensiones no tiene capacidad suficiente para representar la complejidad de 39 columnas mixtas.

### Hipótesis 3: Limitación estructural
Si ninguna variante supera el umbral del −5% de gap F1, la conclusión es que la arquitectura VAE tiene una limitación estructural ante este tipo de distribución: al generar desde una gaussiana isotrópica sin mecanismo condicional, colapsa la clase minoritaria `<30` (11.1%) hacia la moda de la distribución latente, perdiendo su señal discriminativa. Este fenómeno es conocido en la literatura (El Emam et al., 2020; Stadler et al., 2022) y es la razón por la que arquitecturas como CTGAN y TabDDPM incorporan mecanismos condicionales.

*Generado: 2026-05-09 19:39:02*