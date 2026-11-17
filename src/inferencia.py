"""Inferencia local del modelo de sentimiento (Unidad 4). No importa Streamlit."""
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.config import RUTA_MODELO, UMBRAL_BAJA_CONFIANZA
from src.preprocesamiento import limpiar_texto

RAIZ_REPO = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga una sola vez por proceso el modelo extremo a extremo (.keras)."""
    ruta = RAIZ_REPO / RUTA_MODELO
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta la sección 5 de "
            "notebooks/entrenamiento_u4.ipynb o descarga los pesos (assets/apuntes/descarga_pesos.md)."
        )
    import keras  # import diferido: el módulo se importa rápido y falla claro si falta el archivo

    return keras.saving.load_model(ruta)


def predecir_sentimiento(texto: str) -> dict:
    """Clasifica una reseña (en inglés). Devuelve sentimiento, confianza, p_positivo y baja_confianza."""
    if not isinstance(texto, str):
        raise ValueError(f"El texto debe ser str, se recibió {type(texto).__name__}.")
    if not texto.strip():
        raise ValueError("El texto está vacío o solo contiene espacios.")
    limpio = limpiar_texto(texto)  # el vectorizador exportado no estandariza: la limpieza va aquí
    modelo = cargar_modelo()
    # El texto largo no falla: TextVectorization trunca a SEQ_LEN tokens
    p = float(modelo(tf.constant([[limpio]]), training=False).numpy()[0][0])
    confianza = max(p, 1.0 - p)
    return {
        "sentimiento": "positivo" if p >= 0.5 else "negativo",
        "confianza": round(confianza, 4),
        "p_positivo": round(p, 4),
        "baja_confianza": bool(confianza < UMBRAL_BAJA_CONFIANZA),
    }
