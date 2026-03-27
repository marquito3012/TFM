# 📊 Informe Profesional de Análisis Exploratorio de Datos (EDA)

## 1. Introducción
El presente análisis se centra en el dataset **Diabetes 130-US Hospitals (1999-2008)**, el cual recoge información sobre encuentros hospitalarios de pacientes diabéticos en 130 centros de EE.UU. El objetivo es comprender las dinámicas de readmisión y la calidad de los datos para la posterior generación de datos sintéticos.

## 2. Descripción del Dataset
- **Registros totales**: 101,766
- **Variables**: 50 (13 numéricas, 37 categóricas)
- **Variable Objetivo**: `readmitted` (<30, >30, NO)

## 3. Calidad de los Datos y Valores Faltantes
Se ha detectado una alta presencia de valores faltantes (representados originalmente como '?'). Las variables más críticas son:
- **Weight**: 96.86% de nulos (se recomienda descartar por falta de representatividad).
- **Medical Specialty**: 49.08% de nulos.
- **Payer Code**: 39.56% de nulos.
- **A1Cresult / Max Glu Serum**: >80% de nulos (indica que las pruebas no se realizaron para la mayoría de pacientes).

## 4. Hallazgos Clave

### 4.1. Filtrado de Pacientes (Crítico para Modelado)
Se han identificado **2,306 registros** donde el paciente falleció o fue trasladado a cuidados paliativos (`discharge_disposition_id` en [11, 13, 14, 19, 20, 21]). 
> [!IMPORTANT]
> Estos registros no son elegibles para una readmisión y deben ser excluidos del entrenamiento del modelo predictivo para evitar sesgos ("falsos negativos" de readmisión).

### 4.2. Perfil del Paciente y Hospitalización
- **Edad**: La distribución está sesgada hacia pacientes de entre 60 y 90 años.
- **Tiempo en hospital**: La mediana es de aproximadamente 4 días, con una cola larga de hasta 14 días.
- **Lab Procedures**: Existe una fuerte concentración en torno a 40-50 procedimientos.

### 4.3. Medicación y Tratamiento
- La **Insulina** es el medicamento más frecuente (54,382 prescripciones).
- La **Metformina** es el segundo más común (20,548 prescripciones).
- Se observa una gran cantidad de fármacos de nicho con frecuencia casi nula (ej. `acetohexamide`, `troglitazone`, `examide`), lo que sugiere una posible simplificación del espacio de características.

### 4.4. Multingresos
- Se detectó un paciente con hasta **40 ingresos** diferentes.
- La frecuencia de ingresos por paciente es un indicador relevante del estado crónico del mismo.

## 5. Conclusiones para las Siguientes Fases
1. **Limpieza**: Eliminar columnas con >90% de nulos y filtrar registros de fallecidos.
2. **Feature Engineering**: Agremiar diagnósticos (ICD-9) y crear una variable de "número de hospitalizaciones previas".
3. **Modelado**: El desequilibrio de la clase `<30` sugiere que se necesitarán técnicas de balanceo o modelos generativos (CTGAN) para robustecer la minoría.
