# TFM

## 🟦 Fase 1: Investigación y Configuración (Semanas 1-2)
- [x] **Selección del Dataset:** Identificar y descargar un dataset público de alta sensibilidad (Finanzas o Salud). **Decidí elegir el conjunto de datos [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- [x] **Revisión Bibliográfica:**
    - [x] **Lectura profunda de los papers de la propuesta**
    - [x] **Estudio de las arquitecturas**: CTGAN, TVAE y Tabular Diffusion.
     
      * **CTGAN (Conditional Tabular GAN):** Se basa en un juego de suma cero entre un generador y un discriminador. Su innovación es el **entrenamiento condicional**, que permite balancear clases minoritarias al "forzar" al modelo a generar ejemplos de categorías poco comunes.
      * **TVAE (Tabular Variational Autoencoder):** Utiliza un codificador para comprimir los datos en un espacio latente y un decodificador para reconstruirlos. Es mucho más estable y rápido de entrenar que las GAN, siendo ideal para datasets masivos donde la eficiencia es clave.
      * **Tabular Diffusion (TabDDPM):** Representa el estado del arte actual. Aprende a revertir un proceso de degradación de datos (ruido). Aunque es computacionalmente costoso, captura correlaciones complejas y distribuciones multimodales con una precisión que supera a los modelos anteriores.

        | Característica | CTGAN | TVAE | Tabular Diffusion (TabDDPM) |
        | :--- | :--- | :--- | :--- |
        | **Arquitectura Base** | GAN (Redes Adversarias) | VAE (Autoencoder Variacional) | Modelo de Difusión Probabilística |
        | **Mecanismo Clave** | Generador Condicional | Espacio Latente Probabilístico | Difusión Reversa (Denoising) |
        | **Estabilidad** | Baja (propenso a colapso de modo) | Alta (convergencia rápida) | Muy Alta (proceso iterativo robusto) |
        | **Velocidad de Entrenamiento** | Media | **Muy Rápida** | Lenta |
        | **Calidad Estadística** | Buena | Media (tiende a "suavizar" bordes) | **Excelente (Estado del arte)** |
        | **Manejo de Categorías** | Excelente (vía muestreo condicional) | Bueno (vía Softmax/Cross-entropy) | Excelente (vía Cadenas de Markov) |
        | **Complejidad de Uso** | Media (ajuste de hiperparámetros) | Baja | Alta (requiere más recursos de GPU) |

        **Usa CTGAN si:** Tienes un dataset con un desequilibrio de clases muy marcado (ej. detección de fraudes).
        **Usa TVAE si:** Necesitas generar datos rápidamente o tienes recursos computacionales limitados.
        **Usa Tabular Diffusion si:** La fidelidad de los datos es crítica y no te importa el tiempo de cómputo (ej. investigación médica o financiera).
- [x] **Análisis de Privacidad:** Definir los límites de la normativa GDPR aplicados al proyecto.
  Diferencia entre Datos Personales y Datos Anónimos:
    - Datos personales: se refiere a cualquier información relacionada con una persona física identificada o identificable. Aquí el GDPR aplica con toda su dureza.
    - Datos anónimos: información que no se refiere a una persona identificable. El GDPR estipula explícitamente que no se aplica a los datos anónimos.

  El objetivo legal del TFM es demostrar matemáticamente que los datos sintéticos cruzan esa frontera y se convierten en datos anónimos, evadiendo así las restricciones del GDPR.

  Se pueden establecer 3 límites legales para que los datos sintéticos estén fuera del alcance del GDPR:

  1. Riesgo de Re-identificación: ¿Puede un atacante aislar un registro sintético y vincularlo a una persona real? **Solución** -> Utilizar la métrica Distance to Closest Record (DCR). Con ella, se demuestra que el modelo no ha memorizado los datos de entrenamiento y que ningún registro sintético es una copia exacta de uno real.
  2. Riesgo de Inferencia: ¿Puede alguien deducir información sensible cruzando el dataset sintético con otras bases de datos públicas? **Solución** -> La simulación de Ataques de Inferencia de Membresía (MIA). Se evalúa si un atacante externo podría llegar a determinar si un individuo específico formó parte del dataset original usado para entrenar. Esto garantiza un nivel robusto ante auditorías.
  3. Minimización de Datos y Propósito: Las empresas no pueden dar acceso a datos sensibles (PII) indiscriminadamente para hacer pruebas. **Solución** -> El análisis de viabilidad empresarial. Se demostrará que la solución permite a los equipos de MLOps innovar sin tocar datos PII, mejorando drásticamente el "Time-to-Data".
- [x] **Entorno de Desarrollo:** Configurar el entorno virtual con **PyTorch** y dependencias necesarias.

## 🟨 Fase 2: Ingeniería de Datos y Pre-procesamiento (Semanas 3-4)
- [x] **Análisis exploratorio de los Datos:** Finalizado. Se ha generado un informe detallado con los siguientes hallazgos:
    - **Calidad de Datos:** Variables con alta tasa de nulos (`weight` 97%, `max_glu_serum` 95%, `A1Cresult` 83%).
    - **Criterio de Exclusión:** Identificación de 2,306 registros correspondientes a fallecimientos/hospice que deben filtrarse para el modelo de readmisión.
    - **Distribución de Clases:** La clase menoritaria `<30` (11.1%) confirma la necesidad de SMOTE o modelos generativos balanceados.
    - **Medicación:** Análisis de prevalencia de Insulina y Metformina como ejes del tratamiento.
- [x] **Limpieza de Datos:** Eliminación de 2,423 registros (fallecidos/hospice) y depuración de columnas irrelevantes o con nulos extremos.
- [x] **Agrupación ICD-9:** Consolidación de diagnósticos en 9 categorías clínicas para optimizar el aprendizaje del motor generativo.
- [x] **Ingeniería de Variables:** Creación de variables sintéticas como `prior_visits` y normalización visual del dataset.

## 🟩 Fase 3: Implementación del Motor Generativo (Semanas 5-7)
- [ ] **Desarrollo del Modelo:** Programar las diferentes arquitecturas seleccionadas (CTGAN, TVAE y Tabular Diffusion) en PyTorch.
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
