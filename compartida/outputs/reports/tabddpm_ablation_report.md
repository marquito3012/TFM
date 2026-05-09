# Estudio de Ablación — TabDDPM

**Objetivo:** Evaluar el impacto de la parametrización del proceso de difusión (`schedule` y pasos `T`) en la calidad y utilidad predictiva de los datos sintéticos.

## Variantes evaluadas

| ID | Descripción | `schedule` | `T` (pasos) |
|---|---|---|---|
| V1 | Baseline (configuración original) | lineal | 1000 |
| V2 | Schedule Coseno | coseno | 1000 |
| V3 | Reducción de coste (Lineal) | lineal | 500 |
| V4 | Reducción de coste (Coseno) | coseno | 500 |

## Resultados

| Variante | Wasserstein ↓ | Corr. Diff ↓ | F1-Score | AUC-ROC | Gap F1 vs. Baseline |
|---|---|---|---|---|---|
| Baseline TRTR | 0.0000 | 0.0000 | 0.5906 | 0.6879 | +0.00% |
| V1 — Baseline (Linear, T=1000) | 0.4928 | 0.0112 | 0.5679 | 0.6472 | -3.84% |
| V2 — Cosine, T=1000 | 169.9438 | 0.3165 | 0.5349 | 0.6491 | -9.43% |
| V3 — Linear, T=500 | 0.7666 | 0.0113 | 0.5650 | 0.6422 | -4.34% |
| V4 — Cosine, T=500 | 51.3606 | 0.3951 | 0.5565 | 0.6416 | -5.77% |

## Conclusiones Esperadas
- **Impacto del Schedule Coseno:** Suele ralentizar la destrucción de información en pasos iniciales, lo que permite a la red de Denoising preservar detalles finos (mejorando potencialmente la fidelidad).
- **Impacto de T=500:** Reduce a la mitad el coste de inferencia (generación), pero una caída grande en TSTR significaría que se han degradado las fronteras de decisión.