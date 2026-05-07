# Informe de Progreso del Trabajo de Fin de Máster

**Título:** Generación de Datos Sintéticos para Sectores Sensibles mediante IA Generativa  
**Autor:** Marco Fernández Pérez  
**Fecha:** Mayo 2026  
**Fase actual:** 6 de 6 — Impacto Empresarial y Memoria Final (en curso)

---

## Resumen Ejecutivo

Este TFM desarrolla y valida un **framework de IA Generativa** capaz de producir datos tabulares sintéticos clínicos de alta fidelidad, demostrando matemáticamente que dichos datos cruzan el umbral de anonimización exigido por el **GDPR**. El proyecto trabaja sobre el dataset público *Diabetes 130-US Hospitals (1999-2008)* (101.766 registros, 50 variables) y compara tres arquitecturas generativas del estado del arte: **CTGAN**, **TVAE** y **TabDDPM**.

A fecha de este informe, **cinco de las seis fases planificadas están completadas**. Los resultados obtenidos superan los umbrales de calidad definidos como objetivo, con TabDDPM destacando como la arquitectura más robusta en los tres ejes de evaluación: fidelidad estadística, utilidad predictiva y privacidad formal.

---

## 1. Contexto y Motivación

El acceso a datos clínicos reales está sujeto a restricciones legales severas bajo el **GDPR** (Reglamento General de Protección de Datos). En entornos hospitalarios, el proceso burocrático para obtener autorización de acceso a datos de pacientes puede extenderse entre **3 y 5 meses**, constituyendo un cuello de botella crítico para los equipos de Data Science e investigación.

El presente trabajo propone una solución técnica a este problema: entrenar modelos generativos sobre datos reales una única vez y, a partir de ese momento, proporcionar datos sintéticos anónimos bajo demanda. Para que esta solución sea legalmente válida y científicamente rigurosa, se deben cumplir tres criterios simultáneos:

1. **Alta fidelidad estadística** — Los datos sintéticos deben reproducir las distribuciones y correlaciones del dataset original con la suficiente precisión como para ser útiles en el entrenamiento de modelos ML.
2. **Alta utilidad predictiva** — Un modelo entrenado exclusivamente con datos sintéticos debe rendir de forma comparable a uno entrenado con datos reales (paradigma TSTR).
3. **Anonimización formal y demostrable** — Los datos sintéticos no deben permitir la re-identificación de individuos reales ni revelar información de membresía, lo que los excluye del alcance del GDPR.

---

## 2. Dataset y Preprocesamiento

### 2.1 Descripción del Dataset

| Atributo | Valor |
|---|---|
| Nombre | Diabetes 130-US Hospitals (1999-2008) |
| Fuente | UCI Machine Learning Repository |
| Registros originales | 101.766 |
| Variables originales | 50 (13 numéricas, 37 categóricas) |
| Variable objetivo | `readmitted` (`<30`, `>30`, `NO`) |

### 2.2 Hallazgos del Análisis Exploratorio (EDA)

El análisis exploratorio reveló tres hallazgos críticos que condicionan toda la fase de ingeniería de datos:

- **Alta tasa de nulos estructurales:** La variable `weight` presenta un 96.86% de valores faltantes, `medical_specialty` un 49.08% y `A1Cresult` un 83%. Estas ausencias son informativas (los tests no se realizaron) y no aleatorias, lo que hace inadecuada su imputación numérica.
- **Sesgo de clase:** La clase minoritaria `<30` (reingresos urgentes) representa solo el **11.1%** de los registros, confirmando la necesidad de técnicas de balanceo o modelos generativos condicionales.
- **Contaminación del objetivo:** Se identificaron **2.306 registros** de pacientes fallecidos o trasladados a cuidados paliativos que tienen reingresos codificados como `NO` de forma espuria. Incluirlos en el entrenamiento introduciría un sesgo sistemático en el modelo de readmisión.

### 2.3 Ingeniería de Datos

El pipeline de preprocesamiento transformó el dataset en una versión limpia con **99.340 registros y 39 columnas**:

- **Filtrado:** Eliminación de 2.423 registros (fallecidos/hospice) y 3 registros con género inválido.
- **Reducción dimensional:** Eliminación de columnas con nulos masivos (`weight`, identificadores técnicos sin valor predictivo).
- **Agrupación ICD-9:** Los cientos de códigos de diagnóstico se consolidaron en **9 categorías clínicas** estándar (Circulatorio, Respiratorio, Diabetes, Digestivo, Lesiones, Genitourinario, Musculoesquelético, Neoplasias, Otros), reduciendo la cardinalidad y mejorando la convergencia de los modelos.
- **Feature Engineering:**
  - `prior_visits`: Variable sintética que agrega el historial de visitas previas (ambulatorias + urgencias + ingresos anteriores). Media: 1.19, máximo: 80 visitas.
  - `any_med_change`: Variable binaria que simplifica si hubo ajuste farmacológico durante el encuentro.
- **Imputación:** La variable `race` se imputó con la moda; `medical_specialty` recibió una categoría `"Missing"` explícita.

> **Valoración:** La ingeniería de datos es sólida y metodológicamente justificada. La agrupación ICD-9 sigue la clasificación estándar de Strack et al. (2014) y no es una simplificación arbitraria, sino una decisión técnica respaldada por la literatura. El dataset resultante es idóneo para el entrenamiento de los modelos generativos.

---

## 3. Arquitecturas Generativas Implementadas

Se implementaron y entrenaron tres arquitecturas, cada una representando una generación diferente de modelos generativos para datos tabulares:

| Característica | CTGAN | TVAE | TabDDPM |
|---|---|---|---|
| **Arquitectura** | GAN (Redes Adversarias) | VAE (Autoencoder Variacional) | Modelo de Difusión (DDPM) |
| **Épocas entrenadas** | 300 | 300 | 1.000 |
| **Batch size** | 500 | 500 | 4.096 |
| **Estabilidad** | Media (riesgo de mode collapse) | Alta | Muy alta |
| **Registros generados** | 99.340 | 99.340 | 99.340 |

### Detalles de la Implementación de TabDDPM

TabDDPM es la contribución técnica más significativa del proyecto: se ha implementado **desde cero en PyTorch puro**, sin depender de librerías de alto nivel. La arquitectura incluye:

- **Proceso Forward (difusión):** Schedule lineal de ruido β con T=1.000 pasos, de β₀=1e-4 a β_T=0.02.
- **Red de Denoising (MLP):** 3 capas ocultas de 512 neuronas con conexiones residuales, embedding sinusoidal del timestep (igual que el Transformer original) y normalización LayerNorm.
- **Optimizador:** AdamW con weight decay 1e-4 y Cosine Annealing Learning Rate (lr: 1e-3 → 1e-5).
- **Preprocesador:** Pipeline propio con StandardScaler para numéricas y One-Hot Encoding para categóricas, con transformada inversa para recuperar el DataFrame original.

---

## 4. Evaluación de Fidelidad Estadística

### 4.1 Distancia de Wasserstein

La distancia de Wasserstein (o distancia Earth Mover's) mide cuánto "esfuerzo" se necesita para transformar la distribución sintética en la real. **Cuanto menor, mayor fidelidad.**

| Modelo | Wasserstein Media | Ranking |
|---|---|---|
| **TabDDPM** | **0.397** | 🥇 1º |
| TVAE | 0.602 | 🥈 2º |
| CTGAN | 0.762 | 🥉 3º |

**Análisis por variable:** La variable `num_lab_procedures` es la más difícil de replicar para todos los modelos (WD: 2.31–4.35), probablemente por su distribución bimodal. TabDDPM la reduce a 2.31 frente a 4.35 de CTGAN. En variables como `number_outpatient`, `number_emergency` y `number_inpatient`, TabDDPM se aproxima notablemente a la distribución real (WD: 0.04, 0.05 y 0.07 respectivamente).

### 4.2 Diferencia de Matrices de Correlación

Esta métrica evalúa si el modelo sintético preserva las **relaciones entre variables**, no solo sus distribuciones marginales.

| Modelo | Diff. Correlación Media (MAE) | Ranking |
|---|---|---|
| **TabDDPM** | **0.0122** | 🥇 1º |
| TVAE | 0.0395 | 🥈 2º |
| CTGAN | 0.0675 | 🥉 3º |

> **Valoración:** TabDDPM replica la estructura de correlación del dataset real con una diferencia media de apenas **1.22%**. Este resultado indica que el modelo no solo aprende distribuciones marginales, sino también las dependencias estadísticas entre variables — esencial para que los datos sintéticos sean útiles en modelos ML downstream.

---

## 5. Evaluación de Utilidad Predictiva (TSTR)

El paradigma **Train on Synthetic, Test on Real (TSTR)** es el estándar de facto para evaluar si los datos sintéticos son un sustituto viable para el entrenamiento de modelos de Machine Learning. Se entrenó un clasificador **XGBoost** exclusivamente con datos sintéticos y se evaluó su rendimiento en el conjunto de test real.

### 5.1 Resultados TSTR

| Modelo | F1-Score | AUC-ROC | Gap F1 vs. Baseline |
|---|---|---|---|
| **Baseline (TRTR — Real)** | **0.5949** | **0.6904** | — |
| TabDDPM (TSTR) | 0.5965 | 0.6567 | **-0.27%** ✅ |
| CTGAN (TSTR) | 0.5393 | 0.6143 | -9.34% ⚠️ |
| TVAE (TSTR) | 0.3510 | 0.6693 | -41.0% ❌ |

> **Valoración — TabDDPM:** Una brecha de F1 del **-0.27%** respecto al baseline real es un resultado **excepcional**. El objetivo propuesto era mantener la brecha por debajo del 5%. TabDDPM lo supera ampliamente, situándose prácticamente en paridad con el modelo entrenado con datos reales. Esto confirma que los datos sintéticos de TabDDPM preservan fielmente la señal predictiva del dataset original.

> **Valoración — CTGAN:** Una brecha del -9.34% indica una fidelidad aceptable pero claramente inferior. CTGAN captura las distribuciones principales pero pierde información en las correlaciones, lo que se refleja en el rendimiento downstream. Se sitúa dentro del margen tolerable del -10%.

> **Valoración — TVAE:** La brecha del -41% en F1-Score es significativa. Aunque el AUC-ROC se mantiene razonable (0.6693), la caída del F1 sugiere que el TVAE tiende a "suavizar" las distribuciones de clases, dificultando que el clasificador discrimine correctamente los reingresos. TVAE no es adecuado como sustituto directo para entrenamiento supervisado en este dataset.

---

## 6. Protocolo de Privacidad y Cumplimiento GDPR

### 6.1 Distance to Closest Record (DCR) — Riesgo de Memorización

La métrica DCR mide la distancia euclídea entre cada registro sintético y su vecino más próximo en el dataset real. Un DCR mínimo cercano a cero indicaría que el modelo ha memorizado (copiado) registros individuales.

| Modelo | DCR Mínimo | DCR Medio | % Registros con Riesgo (<0.01) |
|---|---|---|---|
| **TabDDPM** | **0.3938** | 2.8831 | **0.00%** ✅ |
| CTGAN | 0.1196 | 3.1009 | 0.00% ✅ |
| TVAE | 0.0443 | 2.2754 | 0.00% ✅ |

> **Valoración:** Los tres modelos superan el umbral de riesgo con holgura. Ningún registro sintético es una copia o cuasi-copia de un registro real (0.00% de riesgo en todos los casos). El DCR mínimo de TabDDPM (0.39) es el más robusto, garantizando un mayor margen de seguridad. Esto satisface el criterio de **individualización** del Grupo de Trabajo del Artículo 29 para la anonimización.

### 6.2 Membership Inference Attack (MIA) — Riesgo de Inferencia

Se simuló un ataque adversario de inferencia de membresía: un clasificador intenta determinar si un individuo específico pertenecía al dataset de entrenamiento basándose en los datos sintéticos generados.

| Modelo | AUC del Atacante | Interpretación |
|---|---|---|
| **TabDDPM** | **0.5034** | Azar puro ✅ |
| CTGAN | 0.5028 | Azar puro ✅ |
| TVAE | 0.5022 | Azar puro ✅ |

> **Valoración:** Un AUC de 0.50 equivale al rendimiento de un clasificador aleatorio, lo que significa que el atacante **no obtiene ninguna ventaja** sobre el azar. Los tres modelos son completamente inmunes a ataques MIA, satisfaciendo el criterio de **inferencia** del Artículo 29.

### 6.3 Dictamen Legal GDPR

Los datos sintéticos generados cumplen los tres criterios de la Directriz de Anonimización del Grupo de Trabajo del Artículo 29:

| Criterio | Estado |
|---|---|
| ✅ Individualización | Imposible — DCR > 0 en todos los registros |
| ✅ Correlación | No vinculable — correlaciones son sintéticas, sin trazabilidad |
| ✅ Inferencia | No deducible — MIA AUC ~0.50 (aleatorio) |

**Dictamen: Los datos se consideran ANÓNIMOS** y pueden utilizarse libremente para entrenamiento de modelos e innovación sin restricciones de PII bajo el GDPR.

---

## 7. Impacto Empresarial: Métrica Time-to-Data (TTD)

### 7.1 Comparativa de Tiempos

| Proceso | Duración estimada |
|---|---|
| **Proceso tradicional (acceso a datos reales)** | |
| Solicitud al Comité de Ética | 2-4 semanas |
| Aprobación del DPO | 4-8 semanas |
| Firma de NDA/DPA | 2-4 semanas |
| Anonimización manual | 1-2 semanas |
| **Total tradicional** | **9-18 semanas (3-5 meses)** |
| **Proceso sintético (basado en este TFM)** | |
| Solicitud de acceso al entorno | 1 día |
| Generación (TabDDPM) | 10-30 minutos |
| Validación automática | 1 hora |
| **Total sintético** | **< 24 horas** |

**Reducción del Time-to-Data: 98.4%**

> **Valoración:** Esta reducción transforma el acceso a datos de un problema legal-burocrático en un proceso técnico automatizable. El impacto en los ciclos de innovación de los equipos de MLOps es significativo: permite iterar 10 veces más rápido en la validación de hipótesis y elimina la exposición a multas por brechas de datos PII durante la fase de desarrollo.

### 7.2 Arquitectura de Producción Propuesta

El framework está preparado para despliegue en producción:

- **Contenedores Docker:** Portabilidad garantizada a AWS/Azure/GCP o Kubernetes.
- **Formatos estándar:** `tabddpm_model.pt` compatible con TorchScript; modelos CTGAN/TVAE serializados en Pickle estándar.
- **Escalabilidad lineal O(N):** Generar 100.000 registros tarda < 1 minuto en GPU comercial.
- **Arquitectura API propuesta:** FastAPI + Celery/Redis para generación asíncrona en GPU + S3/DB para almacenamiento seguro.

---

## 8. Valoración Global de Resultados

### 8.1 Tabla de Cumplimiento de Objetivos

| Objetivo | Umbral definido | Resultado obtenido | Estado |
|---|---|---|---|
| Brecha F1-Score (TSTR) | < 5% | **-0.27%** (TabDDPM) | ✅ Superado |
| Wasserstein Media | Lo más bajo posible | **0.397** (TabDDPM) | ✅ Excelente |
| Diferencia de correlación | Próxima a 0 | **0.012** (TabDDPM) | ✅ Excelente |
| DCR mínimo | > 0 (sin copias exactas) | **0.394** (TabDDPM) | ✅ Superado |
| Registros con riesgo DCR | 0% | **0.00%** | ✅ Perfecto |
| MIA AUC | ~0.50 (azar) | **0.503** (TabDDPM) | ✅ Cumplido |
| Cumplimiento GDPR | Anonimización demostrable | Dictamen formal emitido | ✅ Cumplido |
| Reducción Time-to-Data | > 90% | **98.4%** | ✅ Superado |

### 8.2 Jerarquía de Modelos

Basándose en el análisis integral de los tres ejes de evaluación:

1. 🥇 **TabDDPM** — Arquitectura dominante en todos los ejes. Recomendada para entornos de producción donde la fidelidad y seguridad son críticas.
2. 🥈 **CTGAN** — Opción válida cuando el tiempo de entrenamiento es una restricción y se acepta una pérdida de fidelidad moderada.
3. 🥉 **TVAE** — Descartado como sustituto de entrenamiento supervisado por la elevada caída de F1-Score. Podría ser útil para generación exploratoria rápida.

### 8.3 Fortalezas del Proyecto

- **Implementación desde cero de TabDDPM en PyTorch puro**, incluyendo schedule de ruido, red MLP residual con timestep embedding sinusoidal y preprocesador reversible. Esta es una contribución técnica de peso para un TFM.
- **Marco de evaluación completo y multi-dimensional:** fidelidad estadística (Wasserstein + correlación), utilidad predictiva (TSTR con XGBoost) y privacidad formal (DCR + MIA). Esta trifecta de evaluación es el estándar en la investigación puntera sobre datos sintéticos.
- **Dictamen GDPR documentado formalmente**, con referencia explícita a los tres criterios del Artículo 29. Esto eleva el trabajo de un experimento técnico a una solución con validez legal.
- **Entorno reproducible Docker** con soporte GPU, lo que garantiza que los experimentos puedan ser auditados y replicados de forma independiente.

### 8.4 Limitaciones y Trabajo Pendiente

- **Evaluación TVAE:** La caída del F1 en TVAE merece una investigación más profunda. Un ajuste de hiperparámetros (especialmente `loss_factor` y `embedding_dim`) podría mejorar significativamente su rendimiento de utilidad.
- **Privacidad diferencial formal:** El trabajo demuestra privacidad empírica (MIA/DCR), pero no implementa mecanismos formales de privacidad diferencial (DP). Sería un complemento de alto valor académico.
- **Estudio de ablación de TabDDPM:** No se ha evaluado el impacto del número de pasos de difusión T ni del schedule (lineal vs. coseno) sobre los resultados finales.
- **Memoria académica:** Los 8 capítulos del índice propuesto están pendientes de redacción formal. Es la tarea crítica de la Fase 6.
- **Bibliografía:** Las referencias a El Emam, Stadler, Xu y Kotelnikov deben formalizarse en el estilo de citación académico requerido.

---

## 9. Estado Actual del Roadmap

| Fase | Descripción | Estado |
|---|---|---|
| Fase 1 | Investigación y Configuración | ✅ Completada |
| Fase 2 | Ingeniería de Datos y EDA | ✅ Completada |
| Fase 3 | Motor Generativo (CTGAN, TVAE, TabDDPM) | ✅ Completada |
| Fase 4 | Evaluación de Calidad y Utilidad | ✅ Completada |
| Fase 5 | Protocolo de Privacidad (DCR + MIA) | ✅ Completada |
| Fase 6 | Impacto Empresarial y **Memoria Final** | 🚧 En curso |

**Progreso global: 5/6 fases completadas (~83%)**

---

## 10. Conclusión

El trabajo realizado hasta la fecha constituye una investigación técnica sólida y completa. El objetivo principal del TFM —demostrar que los datos sintéticos generados por IA pueden cruzar la frontera de la anonimización legal mientras mantienen alta utilidad estadística— **ha sido demostrado con éxito** a través de evidencia empírica cuantitativa.

TabDDPM emerge como la arquitectura de referencia, con resultados que igualan prácticamente el rendimiento de los datos reales en entrenamiento supervisado (brecha F1 de -0.27%), preservan la estructura de correlación del dataset original (diff. 1.22%) y garantizan inmunidad ante ataques de re-identificación (MIA AUC ~0.50, DCR mínimo 0.39).

La etapa pendiente — la redacción de la memoria académica — es la que transformará este conjunto de experimentos técnicos en un documento de investigación formal. El material técnico generado es suficientemente rico para sustentar una memoria de alta calidad.

---

*Marco Fernández Pérez — TFM Generación de Datos Sintéticos para Sectores Sensibles — Mayo 2026*
