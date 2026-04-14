# Informe de Seguimiento: TFM "Generación de Datos Sintéticos para Sectores Sensibles"

**Autor:** Marco Fernández Pérez

## 1. Introducción y Objetivo Principal
El Trabajo de Fin de Máster avanza en el desarrollo y evaluación de un framework de Inteligencia Artificial Generativa capaz de crear datos tabulares sintéticos de alta fidelidad. El objetivo central es demostrar matemáticamente que dichos datos cruzan la barrera legal para catalogarse como **datos anónimos**, cumpliendo así con las restricciones impuestas por el **GDPR**. Esto resulta esencial para entrenar modelos de Machine Learning en entornos altamente regulados, tales como salud o finanzas, garantizando que no exista exposición de datos personales identificables (PII).

## 2. Tecnologías y Arquitecturas Evaludadas
Se han implementado y entrenado con éxito tres enfoques de estado del arte en la generación tabular:
- **CTGAN**: Redes Adversarias Generativas Condicionales, óptimas para datasets con fuerte desajuste de clases.
- **TVAE**: Autoencoders Variacionales para datos tabulares, que ofrecen una alta estabilidad y excelente rapidez de convergencia.
- **Tabular Diffusion (TabDDPM)**: Un modelo de difusión de vanguardia que invierte el proceso de adición de ruido para crear distribuciones complejas con altísima fidelidad estadística.

## 3. Estado de Avance por Fases

A continuación se detalla el progreso actual en base al roadmap del proyecto:

### ✅ Fase 1: Investigación y Configuración (Completada)
- Se ha seleccionado como base práctica el dataset **Diabetes 130-US Hospitals (1999-2008)**.
- Se completó un estudio a fondo de las tres arquitecturas generativas mencionadas.
- Se definieron los límites normativos del GDPR, marcando la diferencia entre Datos Personales y Datos Anónimos. Además, se establecieron tres barreras de auditoría clave:
  1. **DCR (Distance to Closest Record)** para validar el riesgo de re-identificación.
  2. **MIA (Membership Inference Attacks)** para asegurar la protección contra ataques inferenciales.
  3. **Minimización de datos** para optimizar la métrica empresarial *Time-to-Data*.
- El entorno de trabajo se consolidó en un framework de contenedores Docker con aceleración GPU mediante PyTorch.

### ✅ Fase 2: Ingeniería de Datos (Completada)
- Se llevó a cabo un análisis exploratorio intensivo: detección y tratamiento de nulos extremos, limpieza y depuración de registros correspondientes a fallecimientos.
- Se consolidó y agrupó la variable diagnóstica según el estándar **ICD-9**, reduciéndola a 9 macro-categorías clave.
- Se completó la limpieza final y una profunda ingeniería de variables, tales como `prior_visits`, habilitando el dataset de entrenamiento.

### ✅ Fase 3: Implementación del Motor Generativo (Completada)
- **Implementación finalizada:** Se han desarrollado al completo las arquitecturas CTGAN, TVAE y Tabular Diffusion en PyTorch.
- Se ha ajustado la función de pérdida y optimizado los hiperparámetros de los modelos de forma efectiva.
- **Entregables técnicos cumplidos:** Todos los modelos ya se encuentran entrenados y guardados (`ctgan_model.pkl`, `tvae_model.pkl`, `tabddpm_model.pt`). Igualmente, los **datasets sintéticos ya han sido renderizados** y se almacenan exitosamente en csv.

### 🚧 Próximos Pasos (Fases 4-6)
El proyecto entra ahora en la fase de evaluación, donde las siguientes tareas son prioritarias:
- **Calidad y Utilidad Estadística (Fase 4):** Validación mediante entrenamiento TSTR (Train on Synthetic, Test on Real), calculando distancias de Wasserstein y midiendo brechas de rendimiento.
- **Auditoría de Privacidad (Fase 5):** Ejecución de métricas DCR y pruebas ofensivas MIA.
- **Memoria y Conclusión Empresarial (Fase 6):** Redacción del informe final, análisis costo-beneficio del *Time-to-Data* y ensamblado bibliográfico final.

---

**Conclusión General:** El TFM avanza según lo previsto. El hito técnico fundamental —el entrenamiento y generación con difusores y redes adversarias— se ha completado favorablemente. El foco actual pivota hacia las validaciones de las barreras de privacidad relativas al GDPR y a la validación de la utilidad estadística del dato final.
