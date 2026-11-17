"""Constantes compartidas (contrato de nombres del taller).

NOTA (insumo provisional creado en la Tarea 9): este archivo no existía en el repo.
SEQ_LEN = 585 y no 300: es el MAX_LEN (percentil 95 de train_final) con el que se
entrenó el modelo de la tarea 5; la capa Embedding/entrada del modelo guardado depende de él.
"""

SEED = 42
MAX_TOKENS = 20000
SEQ_LEN = 585
UMBRAL_BAJA_CONFIANZA = 0.70
RUTA_MODELO = "models/sentimiento_imdb.keras"
GEMINI_MODEL = None  # [ABIERTO] lo define la tarea 6 (agente); no se usa en la tarea 9
