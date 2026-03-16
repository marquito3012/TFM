# TFM

## 🟦 Fase 1: Investigación y Configuración (Semanas 1-2)
- [ ] [cite_start]**Selección del Dataset:** Identificar y descargar un dataset público de alta sensibilidad (Finanzas o Salud)[cite: 64].
- [ ] [cite_start]**Revisión Bibliográfica:** Estudiar a fondo las arquitecturas CTGAN, TVAE y Tabular Diffusion[cite: 36, 72].
- [ ] [cite_start]**Análisis de Privacidad:** Definir los límites de la normativa GDPR aplicados al proyecto[cite: 20, 32].
- [ ] [cite_start]**Entorno de Desarrollo:** Configurar el entorno virtual con **PyTorch** y dependencias necesarias.

## 🟨 Fase 2: Ingeniería de Datos y Pre-procesamiento (Semanas 3-4)
- [ ] [cite_start]**Limpieza de Datos:** Tratamiento de valores nulos y outliers en el dataset original[cite: 65].
- [ ] [cite_start]**Transformaciones RDT:** Implementar Reversible Data Transforms para columnas no gaussianas.
- [ ] [cite_start]**Codificación:** Manejar variables categóricas de alta cardinalidad y normalización de continuas[cite: 37, 65].

## 🟩 Fase 3: Implementación del Motor Generativo (Semanas 5-7)
- [ ] [cite_start]**Desarrollo del Modelo:** Programar la arquitectura seleccionada (CTGAN o TVAE) en PyTorch[cite: 66, 75].
- [ ] [cite_start]**Entrenamiento:** Configurar el bucle de entrenamiento y funciones de pérdida[cite: 67, 75].
- [ ] [cite_start]**Optimización:** Ajuste de hiperparámetros para evitar *mode collapse* y *overfitting*[cite: 41, 67].
- [ ] [cite_start]**Espacio Latente:** Validar la transformación de datos brutos al espacio latente[cite: 39].

## 🟧 Fase 4: Evaluación de Calidad y Utilidad (Semanas 8-9)
- [ ] [cite_start]**Fidelidad Estadística:** Calcular distancias de Wasserstein y matrices de correlación (Pearson/Spearman)[cite: 44, 45, 77].
- [ ] [cite_start]**Validación TSTR:** Entrenar modelos de ML con datos sintéticos y testear con reales[cite: 46, 48, 68, 78].
- [ ] [cite_start]**Métricas de Éxito:** Verificar que la brecha de rendimiento (F1/AUC) sea inferior al 5%[cite: 49, 50].

## 🟥 Fase 5: Protocolo de Privacidad y Ataques (Semana 10)
- [ ] [cite_start]**Métricas DCR:** Ejecutar pruebas de *Distance to Closest Record* para evitar copias exactas[cite: 53, 69].
- [ ] [cite_start]**Simulación MIA:** Implementar Ataques de Inferencia de Membresía para validar la robustez[cite: 54, 80].
- [ ] [cite_start]**Informe de Riesgos:** Documentar formalmente la resiliencia ante ataques de re-identificación[cite: 21, 80].

## 🟪 Fase 6: Impacto Empresarial y Memoria Final (Semanas 11-12)
- [ ] [cite_start]**Análisis Time-to-Data:** Cuantificar el ahorro de tiempo frente a procesos burocráticos[cite: 60, 86].
- [ ] [cite_start]**Escalabilidad:** Evaluar la integración en flujos MLOps y manejo de datos PII[cite: 61, 86].
- [ ] [cite_start]**Redacción de Memoria:** Completar los 8 capítulos del índice propuesto[cite: 70, 71, 86].
- [ ] [cite_start]**Bibliografía:** Asegurar que todas las referencias (El Emam, Stadler, Xu, etc.) estén correctamente citadas[cite: 87, 88, 90, 91].
