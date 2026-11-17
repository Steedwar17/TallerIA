"""@tool #1: inferencia de la red neuronal local (Unidad 4)."""
import json

from langchain_core.tools import tool

from src.inferencia import predecir_sentimiento


@tool
def clasificar_sentimiento_resena(texto: str) -> str:
    """Ejecuta la red neuronal local entrenada con reseñas IMDB en inglés y clasifica su sentimiento.

    Úsala cuando el usuario entregue el texto de una reseña y pida clasificarla o hacer triaje.
    NO la uses para saludos, preguntas sobre políticas ni para un texto ya clasificado en este turno.

    Args:
        texto: la reseña literal y completa, sin resumir ni traducir.

    Returns:
        JSON de una línea con sentimiento ("positivo"/"negativo"), confianza (0-1) y baja_confianza (bool).
    """
    try:
        r = predecir_sentimiento(texto)
        return json.dumps(
            {"sentimiento": r["sentimiento"], "confianza": r["confianza"], "baja_confianza": r["baja_confianza"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception as e:  # nunca se lanza la excepción hacia el agente
        return json.dumps({"error": str(e)[:90]}, ensure_ascii=False, separators=(",", ":"))
