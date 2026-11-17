# Apuntes — Casos de Análisis de Fallo Real y Baja Confianza (Unidad 4)

## Caso A — Fallo Real de Alta Confianza (Falso Positivo)
- **Origen:** Dataset de test IMDB, índice 5220 (`notebooks/entrenamiento_u4.ipynb`, celda 57).
- **Texto completo:** `Primary plot!Primary direction!Poor interpretation.`
- **Valor real:** 0 (negativo)
- **Predicción del modelo:** 1 (positivo)
- **Probabilidad positiva:** 1.000000
- **Confianza en predicción:** 1.000000
- **Tipo de fallo:** Falso positivo con alta confianza (el modelo sobreponderó "primary" frente a "poor interpretation").

---

## Caso B — Caso de Baja Confianza (Zona Gris / Revisión Humana)
- **Origen:** Evaluación de expresiones mixtas/ambiguas frente al umbral `UMBRAL_BAJA_CONFIANZA = 0.70`.
- **Texto completo:** `Not bad, but not great either. It sits right in the middle.`
- **Predicción del modelo:** negativo
- **Confianza en predicción:** 0.5835
- **Baja confianza (`baja_confianza`):** `true` (confianza < 0.70)
- **Protocolo de triaje aplicable:** `POL_REV` (Prioridad Media: Revisión Humana, no automatizar decisión).
