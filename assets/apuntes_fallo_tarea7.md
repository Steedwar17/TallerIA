# Apuntes — Tarea 7 (Error Analysis)
- Modelo analizado: modelo mejorado (Embedding + GlobalAveragePooling1D + Dropout + Dense + Dropout + Dense)
- Conjunto: test (25000 reseñas), umbral 0.5
- Índice en el dataset de test: 5220
- Valor real: 0 (negativo)
- Predicción: 1 (positivo)
- Probabilidad de clase positiva: 1.000000
- Confianza en la clase predicha: 1.000000
- Tipo de fallo: falso positivo
- Número de palabras de la reseña: 4
- Accuracy en test del modelo mejorado: 0.8650
- Errores totales en test: 3375 (FP: 1351, FN: 2024)
- Textos de test repetidos en train (auditoría tarea 1): 99 ; errores que caen en ellos: 0
- Notebook y celda de origen: notebooks/entrenamiento_u4.ipynb, celda 57

## Texto completo de la reseña (tal cual en el dataset)
Primary plot!Primary direction!Poor interpretation.

## Mis notas (las completa Steedwar)
