# Recomendaciones de Elementos Visuales y Tabulares para la Memoria

Para alcanzar el volumen requerido de 50.000 palabras y mantener el estándar de una tesis doctoral sin abrumar con bloques de texto puro, es imperativo estructurar la información apoyándonos intensamente en recursos visuales y tablas de datos. 

He explorado la carpeta `/compartida/outputs/reports/` y ya contamos con un arsenal excelente. A continuación, detallo qué vamos a utilizar y qué recomiendo que generemos adicionalmente:

## 1. Elementos Visuales Ya Disponibles (Listos para usar)
Estos archivos se han copiado a `Memoria/images/` para inyectarlos en LaTeX:
- **Matrices de Diferencia de Correlación:** `corr_diff_ctgan.png`, `corr_diff_tvae.png`, `corr_diff_tabddpm.png`. Serán vitales para el Capítulo 5 al demostrar cómo TabDDPM preserva la estructura bivariante mucho mejor que los baselines.
- **Distribución DCR:** `dcr_distribution.png`. Fundamental para la justificación de privacidad en el Capítulo 5 (Riesgo de Memorización).

## 2. Tablas Extendidas (Se generarán directamente en LaTeX)
Convertiremos los archivos `.csv` de resultados en tablas LaTeX muy detalladas. Ocuparán un espacio valioso y aportarán mucha densidad:
- **Métricas TSTR (F1-Score, AUC-ROC):** Extraído de `tstr_results.csv`.
- **Resultados MIA (Membership Inference Attack):** Extraído de `mia_results.csv`.
- **Estudios de Ablación (TVAE y TabDDPM):** Tablas completas extraídas de `tvae_ablation_results.csv` y `tabddpm_ablation_results.csv`.
- **Distancias de Wasserstein:** Como `wasserstein_distances.csv` contiene el valor por cada columna, en el Capítulo 5 pondremos un resumen, y en el **Anexo** volcaremos la tabla completa (¡son 39 filas por modelo, lo que sumará mucha extensión y rigor!).

## 3. Recomendaciones: Nuevos Gráficos a Generar (Opcional pero muy útil)
Si durante la redacción vemos que necesitamos más apoyo visual o extensión, prepararé scripts de Python para generar los siguientes gráficos:
1. **Curva de Pérdida de TabDDPM:** Graficar el histórico de pérdida de la red de Denoising para demostrar la convergencia estable.
2. **Distribuciones Marginales del EDA:** Gráficos de barras para mostrar el desbalanceo del target (`readmitted`) o variables demográficas como `age` o `race`. Esto engrosará muchísimo el **Capítulo 4 (Metodología / EDA)**.
3. **Diagrama de Arquitectura de TabDDPM:** Un esquema hecho con la librería TikZ en LaTeX (o una imagen externa) que muestre el Forward Process y el Reverse Process.
4. **Gráfico de Feature Importance (XGBoost):** Comparar cuáles son las variables más importantes que usa XGBoost para predecir sobre datos reales frente a datos sintéticos. Si coinciden, es una prueba sublime de fidelidad estructural.

> **Acción:** Empezaré redactando con lo que tenemos. Si vemos que nos quedamos cortos de páginas, iremos ejecutando los scripts sugeridos en el punto 3.
