"""
agent.py — Módulo del Agente de Triaje Inteligente.
Ruta 3: PLN con Embeddings + Agente de Triaje (IMDB Reviews).

Decisiones de diseño (Tarea 6):
- Agente: create_react_agent (langgraph.prebuilt) — API actual 2025 (LangGraph 1.2.x)
- Modelo: gemini-2.0-flash (Google AI Studio, plan gratuito)
- Memoria: MemorySaver (langgraph.checkpoint.memory) como checkpointer
- Sesión: thread_id pasado en config = {"configurable": {"thread_id": "..."}}
- Modularidad: crear_agente(tools) recibe la lista de herramientas como parámetro
  → Permite usar @tool #1 provisional ahora y la real de Steedwar en la Tarea 10
"""

import os
import uuid
from typing import List

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Cargar GOOGLE_API_KEY desde .env (nunca se imprime ni se versiona)
load_dotenv()


# ===========================================================================
# @tool #1 PROVISIONAL — SIMULADA
# REEMPLAZAR EN TAREA 10 por la función real de Steedwar (tarea 9).
# Firma acordada: inferir_sentimiento_resena(texto: str) -> str
# Retorna: "SENTIMIENTO: X | CONFIANZA: Y" con X ∈ {positivo, negativo, incierto}
# ===========================================================================
@tool
def inferir_sentimiento_resena(texto: str) -> str:
    """[PROVISIONAL — SIMULADA] Infiere el sentimiento de una reseña de película usando
    la red neuronal de la Unidad 4.

    Úsala cuando el usuario proporcione una reseña o texto de opinión sobre una película
    y necesites determinar si el sentimiento es positivo, negativo o incierto antes de
    consultar el protocolo de acción correspondiente.

    Args:
        texto: Texto de la reseña o mensaje del usuario a clasificar.

    Returns:
        Resumen compacto con el sentimiento inferido y la confianza del modelo.
        Formato: "SENTIMIENTO: X | CONFIANZA: Y"
        donde X ∈ {positivo, negativo, incierto} e Y es un float entre 0.0 y 1.0.
        Se reporta 'incierto' cuando la confianza es < 0.60.

    Nota: Esta implementación es PROVISIONAL y retorna valores simulados.
    En la Tarea 10 se reemplazará por la inferencia real del modelo entrenado.
    """
    # --- SIMULACIÓN DETERMINISTA PARA PRUEBAS ---
    # Lógica simplificada basada en palabras clave para simular la red neuronal.
    # La función real de Steedwar (Tarea 9) reemplazará este bloque completo.
    texto_lower = texto.lower()

    palabras_negativas = [
        "malo", "pésimo", "terrible", "horrible", "decepcionante",
        "aburrido", "basura", "mediocre", "worst", "bad", "boring",
        "awful", "terrible", "disappointing", "hate", "odio", "mala"
    ]
    palabras_positivas = [
        "bueno", "excelente", "fantástico", "increíble", "genial",
        "maravilloso", "good", "great", "excellent", "amazing",
        "wonderful", "love", "best", "brilliant", "perfect", "encanta"
    ]

    conteo_neg = sum(1 for p in palabras_negativas if p in texto_lower)
    conteo_pos = sum(1 for p in palabras_positivas if p in texto_lower)
    total = conteo_neg + conteo_pos

    if total == 0:
        # Sin señal clara → incierto
        confianza = 0.52
        sentimiento = "incierto"
    elif conteo_neg > conteo_pos:
        confianza = min(0.65 + 0.07 * (conteo_neg - conteo_pos), 0.98)
        sentimiento = "negativo" if confianza >= 0.60 else "incierto"
    elif conteo_pos > conteo_neg:
        confianza = min(0.65 + 0.07 * (conteo_pos - conteo_neg), 0.98)
        sentimiento = "positivo" if confianza >= 0.60 else "incierto"
    else:
        confianza = 0.55
        sentimiento = "incierto"

    return f"SENTIMIENTO: {sentimiento} | CONFIANZA: {round(confianza, 2)}"


# ===========================================================================
# GESTIÓN DE SESIÓN (thread_id)
# ===========================================================================

def nuevo_thread_id() -> str:
    """Genera un UUID4 único para iniciar una nueva sesión conversacional.
    Al usar un thread_id nuevo, el MemorySaver no encuentra historial previo
    y la memoria del agente comienza completamente en blanco.
    """
    return str(uuid.uuid4())


def obtener_config(thread_id: str) -> dict:
    """Retorna el dict de configuración que LangGraph usa para enrutar la memoria
    al hilo de conversación correcto.

    Args:
        thread_id: Identificador único de la sesión actual.

    Returns:
        Dict con clave 'configurable' requerida por LangGraph.
    """
    return {"configurable": {"thread_id": thread_id}}


# ===========================================================================
# CREACIÓN DEL AGENTE (MODULAR)
# ===========================================================================

def crear_agente(tools: List):
    """Crea y retorna el agente de triaje inteligente con memoria conversacional.

    Función modular: recibe la lista de herramientas como parámetro para permitir
    intercambiar la @tool #1 provisional por la real de Steedwar en la Tarea 10
    sin modificar ningún otro código.

    Uso actual (Tarea 6):
        agente = crear_agente([inferir_sentimiento_resena, consultar_politicas_triaje])

    Uso en Tarea 10 (con @tool #1 real de Steedwar):
        agente = crear_agente([tool_1_real, consultar_politicas_triaje])

    Args:
        tools: Lista de herramientas (@tool) que el agente puede invocar.

    Returns:
        Agente compilado de LangGraph con MemorySaver como checkpointer.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_clave_aqui":
        raise EnvironmentError(
            "GOOGLE_API_KEY no configurada. "
            "Crea un archivo .env en la raíz del proyecto con tu clave real de Google AI Studio."
        )

    modelo = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )

    checkpointer = MemorySaver()

    agente = create_react_agent(
        model=modelo,
        tools=tools,
        checkpointer=checkpointer,
    )

    return agente


# ===========================================================================
# INVOCACIÓN DEL AGENTE
# ===========================================================================

def invocar_agente(agente, mensaje: str, thread_id: str) -> dict:
    """Invoca el agente con un mensaje de usuario y retorna la respuesta final
    junto con las herramientas invocadas en ese turno.

    Args:
        agente: Agente compilado retornado por crear_agente().
        mensaje: Texto del mensaje del usuario.
        thread_id: ID de sesión para enrutar la memoria correcta.

    Returns:
        Dict con:
          - 'respuesta': str — texto de la respuesta final del agente.
          - 'tools_invocadas': list[str] — nombres de las herramientas invocadas.
    """
    config = obtener_config(thread_id)
    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": mensaje}]},
        config=config,
    )

    # Extraer respuesta final (último AIMessage sin tool_calls)
    respuesta = ""
    tools_invocadas = []

    for msg in resultado["messages"]:
        tipo = type(msg).__name__
        # Detectar tool_calls en mensajes del agente (AIMessage)
        if tipo == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    nombre = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if nombre:
                        tools_invocadas.append(nombre)
            # La respuesta final es el último AIMessage sin tool_calls pendientes
            # Algunos modelos Gemini retornan content como lista de dicts: [{"type": "text", "text": "..."}]
            if hasattr(msg, "content") and msg.content:
                contenido = msg.content
                if isinstance(contenido, list):
                    # Extraer texto de cada bloque y concatenar
                    partes = []
                    for bloque in contenido:
                        if isinstance(bloque, dict) and "text" in bloque:
                            partes.append(bloque["text"])
                        elif isinstance(bloque, str):
                            partes.append(bloque)
                    respuesta = " ".join(partes)
                else:
                    respuesta = str(contenido)

    return {
        "respuesta": respuesta,
        "tools_invocadas": tools_invocadas,
    }
