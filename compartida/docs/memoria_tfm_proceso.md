# 📝 Cuaderno de Bitácora: Proceso del TFM
**Proyecto**: Generación de Datos Sintéticos para la Predicción de Readmisión en Pacientes Diabéticos.

---

## 📅 Fase 1: Análisis Exploratorio y Preparación

### Decisiones de Diseño y Preprocesamiento
1. **Identificación de Nulos**: Se sustituyeron los caracteres '?' por `NaN` para un análisis estadístico correcto.
2. **Criterios de Exclusión**:
   - Se decidió excluir a pacientes fallecidos o en hospice (`discharge_disposition_id` 11, 13, 14, 19, 20, 21) del análisis de readmisión.
   - La columna `weight` se marcará para eliminación debido a la pérdida del 97% de los datos.
3. **Agrupación de Diagnósticos**: Durante el análisis se observó una alta cardinalidad en `diag_1`, `diag_2` y `diag_3`. Se planea agrupar los códigos ICD-9 en categorías clínicas (Circulatorio, Digestivo, Respiratorio, etc.) para mejorar la convergencia de los modelos generativos.

### Observaciones para Modelos Generativos (GANs/VAEs)
- **Distribuciones No Normales**: Variables como `number_emergency` o `number_outpatient` tienen una distribución de Poisson muy marcada; TVAE podría manejar esto mejor que una GAN estándar.
- **Cardinalidad**: Las variables de medicación tienen 4 estados (`No`, `Up`, `Down`, `Steady`). La mayoría son `No`, lo que podría causar colapso de modo en GANs si no se preprocesan adecuadamente.

---

## 📅 Fase 2: Ingeniería de Datos y Preprocesamiento

### Hitos Alcanzados
- [x] **Limpieza de Ruido**: Eliminación de 2,423 registros no elegibles (fallecidos).
- [x] **Agrupación ICD-9**: Reducción de dimensionalidad categórica mediante mapeo clínico. Esto facilitará que la GAN aprenda distribuciones de diagnósticos más robustas.
- [x] **Sintetización de Historial**: La variable `prior_visits` captura la recurrencia del paciente en un solo vector.

### Justificación de Decisiones para la Memoria
- **Mapeo ICD-9**: Se utilizó la clasificación estándar de la literatura (Strack et al.) para agrupar códigos. Esto no solo mejora el rendimiento de los modelos, sino que reduce la probabilidad de "Overfitting" en códigos de baja frecuencia.
- **Eliminación de Medicamentos Residuales**: Se eliminaron fármacos como `troglitazone` o `acetohexamide` debido a que su varianza es casi nula y no aportan información discriminativa al modelo.

---

## 📅 Seguimiento de Tareas Técnicas
- [x] Ejecución de EDA detallado.
- [x] Redacción de informe de hallazgos.
- [x] Limpieza definitiva del dataset y Feature Engineering.
- [ ] Implementación de arquitecturas PyTorch (CTGAN/TVAE/TabDDPM).
- [ ] Evaluación preliminar de métricas de readmisión.

---
*Este documento se actualizará conforme avance la investigación.*
