"""
policies.py — Documentos de políticas internas de triaje y respuesta.
Ruta 3: PLN con Embeddings + Agente de Triaje Inteligente (IMDB Reviews).
"""
from typing import List, Dict

POLITICAS_TRIAJE: List[Dict[str, str]] = [
    {
        "id": "POLITICA_NEGATIVO",
        "category": "negativo",
        "titulo": "Protocolo Reseña Negativa",
        "contenido": (
            "Prioridad: ALTA. "
            "Acción: Generar ticket de insatisfacción. "
            "Pauta: Empatizar con el malestar, agradecer la franqueza y ofrecer canal directo de soporte."
        ),
    },
    {
        "id": "POLITICA_POSITIVO",
        "category": "positivo",
        "titulo": "Protocolo Reseña Positiva",
        "contenido": (
            "Prioridad: ESTÁNDAR. "
            "Acción: Registrar como reseña destacada/promotor. "
            "Pauta: Agradecer el apoyo, celebrar su entusiasmo y recomendar títulos afines del catálogo."
        ),
    },
    {
        "id": "POLITICA_INCIERTO",
        "category": "neutral",
        "titulo": "Protocolo Reseña Incierta / Baja Confianza",
        "contenido": (
            "Prioridad: MEDIA (Revisión Humana). "
            "Acción: Derivar a moderación humana sin categorizar. "
            "Pauta: Solicitar con cortesía mayor detalle o clarificación al usuario."
        ),
    },
]
