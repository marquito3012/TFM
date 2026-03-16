# TFM

## 🟦 Fase 1: Investigación y Configuración (Semanas 1-2)
- [x] **Selección del Dataset:** Identificar y descargar un dataset público de alta sensibilidad (Finanzas o Salud). **Decidí elegir el conjunto de datos [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- [ ] **Revisión Bibliográfica:**
    - [ ] **Lectura profunda de los papers de la propuesta**
    - [ ] **Estudio de las arquitecturas**: CTGAN, TVAE y Tabular Diffusion.
- [ ] **Análisis de Privacidad:** Definir los límites de la normativa GDPR aplicados al proyecto.
- [ ] **Entorno de Desarrollo:** Configurar el entorno virtual con **PyTorch** y dependencias necesarias.

## 🟨 Fase 2: Ingeniería de Datos y Pre-procesamiento (Semanas 3-4)
- [ ] **Limpieza de Datos:** Tratamiento de valores nulos y outliers en el dataset original.
- [ ] **Transformaciones RDT:** Implementar Reversible Data Transforms para columnas no gaussianas.
- [ ] **Codificación:** Manejar variables categóricas de alta cardinalidad y normalización de continuas.

## 🟩 Fase 3: Implementación del Motor Generativo (Semanas 5-7)
- [ ] **Desarrollo del Modelo:** Programar la arquitectura seleccionada (CTGAN o TVAE) en PyTorch.
- [ ] **Entrenamiento:** Configurar el bucle de entrenamiento y funciones de pérdida.
- [ ] **Optimización:** Ajuste de hiperparámetros para evitar *mode collapse* y *overfitting*.
- [ ] **Espacio Latente:** Validar la transformación de datos brutos al espacio latente.

## 🟧 Fase 4: Evaluación de Calidad y Utilidad (Semanas 8-9)
- [ ] **Fidelidad Estadística:** Calcular distancias de Wasserstein y matrices de correlación (Pearson/Spearman).
- [ ] **Validación TSTR:** Entrenar modelos de ML con datos sintéticos y testear con reales.
- [ ] **Métricas de Éxito:** Verificar que la brecha de rendimiento (F1/AUC) sea inferior al 5%.

## 🟥 Fase 5: Protocolo de Privacidad y Ataques (Semana 10)
- [ ] **Métricas DCR:** Ejecutar pruebas de *Distance to Closest Record* para evitar copias exactas.
- [ ] **Simulación MIA:** Implementar Ataques de Inferencia de Membresía para validar la robustez.
- [ ] **Informe de Riesgos:** Documentar formalmente la resiliencia ante ataques de re-identificación.

## 🟪 Fase 6: Impacto Empresarial y Memoria Final (Semanas 11-12)
- [ ] **Análisis Time-to-Data:** Cuantificar el ahorro de tiempo frente a procesos burocráticos.
- [ ] **Escalabilidad:** Evaluar la integración en flujos MLOps y manejo de datos PII.
- [ ] **Redacción de Memoria:** Completar los 8 capítulos del índice propuesto.
- [ ] **Bibliografía:** Asegurar que todas las referencias (El Emam, Stadler, Xu, etc.) estén correctamente citadas.

Para construir la imágen de docker, usamos (`--build` fuerza a Docker a leer la `Dockerfile` e instalar todas las dependencias del `requirements.txt`):
```
docker compose up --build -d
```
Para comprobar si se detecta la tarjeta gráfica (En mi caso RX9070), ejecutar:
```
docker compose exec tfm python3 -c "import torch; print(f'¿GPU detectada?: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Ninguno\"}')"
```
Para ejecutar el contenedor
```
docker compose exec tfm /bin/bash
```
Para entrar a Jupyter Lab, entrar en un buscado a:
```
http://localhost:8888
```