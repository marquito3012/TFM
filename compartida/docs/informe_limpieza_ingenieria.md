# 🛠️ Informe de Ingeniería de Datos y Preprocesamiento

## 1. Resumen de Ejecución
En esta fase se ha transformado el dataset original **Diabetes 130-US Hospitals** en una versión optimizada y limpia (`diabetic_data_clean.csv`), eliminando ruido estadístico y simplificando la complejidad de las variables para los modelos de IA.

## 2. Limpieza y Filtrado de Registros
- **Eliminación de registros no aptos**: Se han filtrado **2,423 registros** correspondientes a pacientes fallecidos o en cuidados paliativos. Esto es fundamental para no sesgar el modelo de predicción de readmisión.
- **Tratamiento de valores inválidos**: Se eliminaron 3 registros con género "Unknown/Invalid".
- **Dimensionalidad**: El dataset se redujo de 50 a **39 columnas**, eliminando identificadores técnicos e información redundante o con nulos masivos (`weight`).

## 3. Feature Engineering y Transformaciones

### 3.1. Agrupación Clínica de Diagnósticos (ICD-9)
Se ha implementado un mapeo lógico para reducir la cardinalidad de los diagnósticos (`diag_1`, `diag_2`, `diag_3`). Los cientos de códigos ICD-9 se han consolidado en 9 categorías funcionales:
- **Circulatorio**: 29,680 casos (el más frecuente).
- **Respiratorio**: 13,934 casos.
- **Diabetes**: 8,661 casos.
- *Otros*: Digestivo, Lesiones, Genitourinario, Musculoesquelético, Neoplasias y Otros.

### 3.2. Creación de Nuevas Variables
1. **prior_visits**: Nueva métrica que suma visitas previas (ambulatorias, de urgencias e ingresos anteriores). 
   - *Insight*: La media de visitas previas es de 1.19, con casos extremos de hasta 80 visitas.
2. **any_med_change**: Variable binaria que simplifica si hubo algún ajuste en la medicación del paciente durante el encuentro.

## 4. Imputación de Datos Faltantes
- Se ha imputado la variable `race` utilizando la moda.
- La variable `medical_specialty` ha sido tratada con una nueva categoría "Missing" para preservar la información de la ausencia de especialidad asignada.

## 5. Resultado Final
- **Dataset de salida**: `diabetic_data_clean.csv`
- **Registros finales**: 99,340
- **Columnas finales**: 39
- **Estado**: Listo para la fase de motor generativo (Fase 3).
