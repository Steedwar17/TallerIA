"""
src/agente.py — Agente de Triaje Inteligente optimizado (Unidad 5).
Contrato de nombres del taller:
- construir_agente(tools, ...)
- nuevo_thread_id()
- config_sesion(thread_id)
- mensajes_del_turno(mensajes)
- herramientas_invocadas(mensajes)
- responder(agente, texto, thread_id)
"""

import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.config import (
    GEMINI_MODEL,
    MAX_CARACTERES_RESENA,
    MAX_MENSAJES_HISTORIAL,
    RPM_GRATUITO,
)

load_dotenv()

# ===========================================================================
# SYSTEM PROMPTS (Base para medicion y Compacto optimizado)
# ===========================================================================

SYSTEM_PROMPT_BASE = (
    "Eres un agente inteligente y asistente experto en la plataforma de streaming encargado de la mesa de triaje de reseñas y comentarios de los clientes.\n"
    "Tu objetivo principal es recibir comentarios, reseñas y opiniones de los usuarios sobre películas y contenidos de la plataforma, y determinar la mejor acción a seguir.\n"
    "Debes seguir estrictamente las siguientes seis reglas fundamentales en cada interacción:\n"
    "Regla 1: Cuando el usuario te salude amablemente o haga una pregunta general sobre tus capacidades o sobre la plataforma, responde directamente con cortesía sin invocar ninguna de las herramientas disponibles.\n"
    "Regla 2: Cuando el usuario entregue el texto de una reseña u opinión sobre una película en idioma inglés que no haya sido clasificada todavía, debes invocar obligatoriamente la herramienta clasificar_sentimiento_resena para predecir si el sentimiento es positivo o negativo y conocer la certeza del modelo.\n"
    "Regla 3: Una vez obtenido el resultado de sentimiento y confianza de la reseña, o si el usuario pregunta explícitamente sobre directrices y políticas de respuesta, debes invocar la herramienta consultar_politica_triaje pasando los parámetros necesarios para recuperar el protocolo oficial.\n"
    "Regla 4: Si el resultado de la clasificación arroja una confianza inferior a 0.70 o un sentimiento incierto o ambiguo, debes asegurar que se aplique la política de revisión humana y moderación (POL_REV) y no automatizar una respuesta definitiva.\n"
    "Regla 5: Luego de consultar las políticas correspondientes, debes redactar una respuesta estructurada que informe al usuario o al moderador la categoría detectada, el nivel de prioridad, la acción requerida y la pauta de respuesta corporativa recomendada.\n"
    "Regla 6: Mantén siempre un tono profesional, empático y formal, evitando redundancias, texto de relleno innecesario o información que no haya sido solicitada por el usuario en la sesión."
)

SYSTEM_PROMPT_COMPACTO = (
    "Asistente de triaje para resenas de streaming.\n"
    "Reglas:\n"
    "1. Saludos/dudas generales: responde directo sin tools.\n"
    "2. Resena en ingles no clasificada: usa clasificar_sentimiento_resena.\n"
    "3. Con sentimiento o al pedir directriz: usa consultar_politica_triaje.\n"
    "4. Si confianza < 0.70 o incierto: aplica politica POL_REV.\n"
    "5. Con la salida de tools: menciona el sentimiento y confianza detectados, prioridad, accion y pauta.\n"
    "6. Responde breve y estructurado; sin relleno ni repetir el texto analizado."
)


from src.tool_sentimiento import clasificar_sentimiento_resena
from src.tool_politicas import consultar_politica_triaje


# ===========================================================================
# GESTIÓN DE SESIÓN (thread_id)
# ===========================================================================

def nuevo_thread_id() -> str:
    """Genera un UUID4 único para iniciar una nueva sesión conversacional."""
    return str(uuid.uuid4())


def config_sesion(thread_id: str) -> dict:
    """Retorna el diccionario de configuración para aislar la memoria de la sesión."""
    return {"configurable": {"thread_id": thread_id}}


# Alias de compatibilidad con tareas previas
obtener_config = config_sesion


# ===========================================================================
# RATE LIMITER (Singleton para respetar RPM gratuito de Google AI Studio)
# ===========================================================================

_RATE_LIMITER_GLOBAL: Optional[InMemoryRateLimiter] = None


def obtener_rate_limiter() -> InMemoryRateLimiter:
    """Retorna la instancia global única de InMemoryRateLimiter configurada para RPM_GRATUITO."""
    global _RATE_LIMITER_GLOBAL
    if _RATE_LIMITER_GLOBAL is None:
        # 15 RPM = 0.25 requests/sec. Intervalo 0.1s. Capacidad de ráfaga 2.
        rps = max(RPM_GRATUITO / 60.0, 0.05)
        _RATE_LIMITER_GLOBAL = InMemoryRateLimiter(
            requests_per_second=rps,
            check_every_n_seconds=0.1,
            max_bucket_size=2,
        )
    return _RATE_LIMITER_GLOBAL


# ===========================================================================
# RECORTE DE HISTORIAL (Middleware @before_model / prompt modifier)
# ===========================================================================

def crear_prompt_modifier(
    system_prompt: str,
    max_mensajes: int = MAX_MENSAJES_HISTORIAL,
) -> Callable[[Dict[str, Any]], List[BaseMessage]]:
    """Crea la función modificadora de estado para el agente.
    Aplica trim_messages asegurando:
    1. Máximo max_mensajes (últimos).
    2. Comienza siempre en un HumanMessage.
    3. No separa tool_calls de sus ToolMessage.
    4. Conserva intacto el SystemMessage al inicio.
    """
    sys_msg = SystemMessage(content=system_prompt)

    def modifier(state: Dict[str, Any]) -> List[BaseMessage]:
        mensajes = state.get("messages", [])
        if not mensajes:
            return [sys_msg]

        recortados = trim_messages(
            mensajes,
            max_tokens=max_mensajes,
            token_counter=len,
            strategy="last",
            start_on="human",
            allow_partial=False,
            include_system=False,
        )
        return [sys_msg] + recortados

    return modifier


# ===========================================================================
# CONSTRUCCIÓN DEL AGENTE (MODULAR)
# ===========================================================================

def construir_agente(
    tools: List,
    system_prompt: Optional[str] = None,
    usar_optimizaciones: bool = True,
    checkpointer: Optional[Any] = None,
):
    """Construye y compila el agente LangGraph.

    Args:
        tools: Lista de herramientas (@tool).
        system_prompt: Texto de instrucciones del sistema (si es None, usa SYSTEM_PROMPT_COMPACTO).
        usar_optimizaciones: Si True aplica rate limiter, max_output_tokens y recorte de historial.
        checkpointer: Checkpointer de memoria (por defecto MemorySaver).

    Returns:
        Agente compilado de LangGraph.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_clave_aqui":
        raise EnvironmentError(
            "GOOGLE_API_KEY no configurada. Agrega tu clave en el archivo .env."
        )

    s_prompt = system_prompt or (SYSTEM_PROMPT_COMPACTO if usar_optimizaciones else SYSTEM_PROMPT_BASE)

    kwargs_modelo: Dict[str, Any] = {
        "model": GEMINI_MODEL or "gemini-3.5-flash-lite",
        "google_api_key": api_key,
        "temperature": 0,
        "max_retries": 2,
    }

    if usar_optimizaciones:
        kwargs_modelo["max_output_tokens"] = 512
        kwargs_modelo["rate_limiter"] = obtener_rate_limiter()

    modelo = ChatGoogleGenerativeAI(**kwargs_modelo)
    cp = checkpointer or MemorySaver()

    if usar_optimizaciones:
        prompt_fn = crear_prompt_modifier(s_prompt, max_mensajes=MAX_MENSAJES_HISTORIAL)
        agente = create_react_agent(
            model=modelo,
            tools=tools,
            prompt=prompt_fn,
            checkpointer=cp,
        )
    else:
        # Modo baseline sin optimizaciones
        agente = create_react_agent(
            model=modelo,
            tools=tools,
            prompt=SystemMessage(content=s_prompt),
            checkpointer=cp,
        )

    return agente


# Alias de compatibilidad
crear_agente = construir_agente


# ===========================================================================
# EXTRACCIÓN Y RESPUESTA DEL AGENTE
# ===========================================================================

def herramientas_invocadas(mensajes: List[BaseMessage]) -> List[str]:
    """Extrae la lista de nombres de herramientas invocadas en los mensajes del turno."""
    tools = []
    for msg in mensajes:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                nombre = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                if nombre and nombre not in tools:
                    tools.append(nombre)
    return tools


def mensajes_del_turno(mensajes: List[BaseMessage]) -> List[BaseMessage]:
    """Retorna los mensajes generados desde el último HumanMessage."""
    idx_humano = -1
    for i in range(len(mensajes) - 1, -1, -1):
        if isinstance(mensajes[i], HumanMessage):
            idx_humano = i
            break
    if idx_humano >= 0:
        return mensajes[idx_humano:]
    return mensajes


def responder(agente, texto: str, thread_id: str) -> Dict[str, Any]:
    """Ejecuta un turno conversacional y retorna el resultado detallado con métricas.

    Args:
        agente: Agente compilado.
        texto: Mensaje del usuario (se recorta a MAX_CARACTERES_RESENA si excede).
        thread_id: ID de sesión para la memoria.

    Returns:
        Dict con 'respuesta', 'tools_invocadas', 'input_tokens', 'output_tokens',
        'total_tokens', 'num_llamadas_modelo'.
    """
    # Tope de entrada defensivo
    texto_procesado = (texto or "").strip()
    if len(texto_procesado) > MAX_CARACTERES_RESENA:
        texto_procesado = texto_procesado[:MAX_CARACTERES_RESENA]

    cfg = config_sesion(thread_id)
    resultado = agente.invoke(
        {"messages": [{"role": "user", "content": texto_procesado}]},
        config=cfg,
    )

    mensajes_todos = resultado.get("messages", [])
    msgs_turno = mensajes_del_turno(mensajes_todos)

    # Métricas del turno
    inp_tok = 0
    out_tok = 0
    tot_tok = 0
    llamadas_modelo = 0
    respuesta = ""

    for msg in msgs_turno:
        if isinstance(msg, AIMessage):
            llamadas_modelo += 1
            meta = getattr(msg, "usage_metadata", None)
            if meta:
                inp_tok += meta.get("input_tokens", 0)
                out_tok += meta.get("output_tokens", 0)
                tot_tok += meta.get("total_tokens", 0)

            # Contenido final
            if hasattr(msg, "content") and msg.content:
                c = msg.content
                if isinstance(c, list):
                    partes = []
                    for b in c:
                        if isinstance(b, dict) and "text" in b:
                            partes.append(b["text"])
                        elif isinstance(b, str):
                            partes.append(b)
                    respuesta = " ".join(partes)
                else:
                    respuesta = str(c)

    # Si usage_metadata vino vacío en algún mensaje, calcular fallback total
    if tot_tok == 0 and (inp_tok + out_tok) > 0:
        tot_tok = inp_tok + out_tok

    tools_usadas = herramientas_invocadas(msgs_turno)
    return {
        "respuesta": respuesta,
        "tools_invocadas": tools_usadas,
        "herramientas": tools_usadas,
        "input_tokens": inp_tok,
        "output_tokens": out_tok,
        "total_tokens": tot_tok,
        "num_llamadas_modelo": llamadas_modelo,
        "mensajes": msgs_turno,
    }


# Alias de compatibilidad
invocar_agente = responder
