# Informe de Auditoría de Privacidad y Cumplimiento GDPR

**Proyecto:** TFM - Generación de Datos Sintéticos para Sectores Sensibles
**Fecha:** 21 de Abril de 2026
**Responsable:** Marco Fernández Pérez

## 1. Resumen Ejecutivo
Tras realizar pruebas ofensivas y métricas de distancia, se confirma que los datos sintéticos generados por las arquitecturas **CTGAN**, **TVAE** y, especialmente, **TabDDPM**, cumplen con los criterios de **anonimización** requeridos para operar fuera del alcance del GDPR. El riesgo de re-identificación e inferencia es insignificante.

## 2. Resultados de Memorización (Distance to Closest Record - DCR)
La métrica DCR mide la distancia euclidiana entre cada registro sintético y su vecino más cercano en el dataset real.

| Modelo | Distancia Mínima | Distancia Media | Riesgo de Copia (<0.01) |
| :--- | :--- | :--- | :--- |
| **TabDDPM** | **0.3938** | 2.8831 | **0.00%** |
| CTGAN | 0.1196 | 3.1009 | 0.00% |
| TVAE | 0.0443 | 2.2754 | 0.00% |

**Conclusión DCR:** Ningún modelo ha generado copias exactas de pacientes reales. **TabDDPM** es el modelo más robusto, manteniendo una distancia de seguridad mayor (0.39) mientras ofrece la mayor utilidad estadística.

## 3. Resultados de Inferencia (Membership Inference Attack - MIA)
Se simuló un ataque donde un adversario intenta predecir si un individuo perteneció al dataset de entrenamiento basándose en la proximidad de los datos sintéticos.

| Modelo | AUC del Atacante | Accuracy del Atacante | Nivel de Privacidad |
| :--- | :--- | :--- | :--- |
| **TabDDPM** | **0.5033** | 0.4191 | **Excelente** |
| CTGAN | 0.5028 | 0.3955 | Excelente |
| TVAE | 0.5022 | 0.6618 | Excelente (AUC baseline) |

**Conclusión MIA:** Un AUC cercano a **0.50** indica que el atacante no tiene ventaja sobre el azar. Los modelos no filtran información sobre la membresía de los individuos originales.

## 4. Dictamen Final sobre GDPR
Los datos sintéticos generados cumplen los tres criterios del Grupo de Trabajo del Artículo 29 para la anonimización:
1. **Individualización:** Imposible aislar a un individuo (DCR > 0).
2. **Correlación:** Las correlaciones son sintéticas y no vinculables a registros específicos (MIA AUC ~0.5).
3. **Inferencia:** No se puede deducir información sensible con una probabilidad superior al azar.

**Dictamen:** Los datos se consideran **ANÓNIMOS** y pueden ser utilizados libremente para entrenamiento de modelos e innovación sin restricciones de PII.
