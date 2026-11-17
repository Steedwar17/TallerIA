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
GEMINI_MODEL = "gemini-3.5-flash-lite"
MAX_CARACTERES_RESENA = 4000
RPM_GRATUITO = 15
MAX_MENSAJES_HISTORIAL = 8
