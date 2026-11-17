"""
src/tool_politicas.py — @tool #2: Consulta de políticas internas de triaje (Unidad 5).
Contrato del taller: consultar_politica_triaje(consulta, sentimiento=None, confianza=None)
"""
import json
from typing import Optional
from langchain_core.tools import tool

# Catálogo oficial de políticas de triaje (streaming platform)
POLITICAS_TRIAJE = {
    "negativo": {
        "pol_id": "POL_NEG",
        "prioridad": "ALTA",
        "accion": "Ticket de insatisfaccion",
        "pauta": "Empatizar con malestar, agradecer franqueza y ofrecer soporte directo."
    },
    "positivo": {
        "pol_id": "POL_POS",
        "prioridad": "ESTANDAR",
        "accion": "Registrar promotor",
        "pauta": "Agradecer apoyo, celebrar entusiasmo y sugerir titulos afines."
    },
    "incierto": {
        "pol_id": "POL_REV",
        "prioridad": "MEDIA",
        "accion": "Derivar a moderacion",
        "pauta": "Solicitar cortesmente mayor detalle o clarificacion al usuario."
    }
}


@tool
def consultar_politica_triaje(
    consulta: str,
    sentimiento: Optional[str] = None,
    confianza: Optional[float] = None
) -> str:
    """Consulta la politica interna y protocolo de respuesta para una resena de cliente.

    Usala cuando conozcas el sentimiento/confianza de la resena o cuando el usuario pregunte
    directamente sobre lineamientos, politicas de triaje o protocolos de escalamiento.
    NO la uses para clasificar texto no analizado (usa clasificar_sentimiento_resena) ni
    para saludos o preguntas genericas no relacionadas con politicas.

    Args:
        consulta: pregunta o texto de consulta sobre el protocolo o la resena.
        sentimiento: 'positivo', 'negativo' o 'incierto' (si ya fue inferido).
        confianza: valor flotante de certeza (0.0 a 1.0).

    Returns:
        JSON compacto (max 600 caracteres) con pol_id, prioridad, accion y pauta.
    """
    sent = (sentimiento or "").strip().lower()
    conf = float(confianza) if confianza is not None else 1.0

    if conf < 0.70 or sent in ["incierto", "neutral", "duda"]:
        categoria = "incierto"
    elif sent in ["negativo", "negative", "malo", "critica"]:
        categoria = "negativo"
    elif sent in ["positivo", "positive", "bueno", "promotor"]:
        categoria = "positivo"
    else:
        # Analisis de palabras clave en la consulta si no se paso sentimiento explicito
        q_lower = consulta.lower()
        if any(w in q_lower for w in ["negativ", "mal", "queja", "reclam", "pesim"]):
            categoria = "negativo"
        elif any(w in q_lower for w in ["positiv", "buen", "excelent", "promot"]):
            categoria = "positivo"
        else:
            categoria = "incierto"

    resultado = POLITICAS_TRIAJE.get(categoria, POLITICAS_TRIAJE["incierto"])
    salida_json = json.dumps(resultado, ensure_ascii=False, separators=(",", ":"))
    assert len(salida_json) <= 600, f"Salida excede limite duro de 600 chars: {len(salida_json)}"
    return salida_json
