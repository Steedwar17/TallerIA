"""app_agent.py — Módulo Streamlit del Agente de Triaje (Tarea 12).

Proporciona la interfaz conversacional con memoria por sesión, trazabilidad
de herramientas invocadas, detalle crudo de invocaciones y métricas de tokens.
"""

import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Asegurar raíz del proyecto en sys.path
RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Cargar variables de entorno
load_dotenv()

from src.config import MAX_CARACTERES_RESENA
from src.herramientas import obtener_tools
from src.agente import construir_agente, nuevo_thread_id, responder
from langchain_core.messages import AIMessage, ToolMessage


@st.cache_resource(show_spinner="Iniciando el agente...")
def obtener_agente():
    """Compila y almacena en caché el agente con sus herramientas y checkpointer.

    NOTA DE SEGURIDAD / ARQUITECTURA:
    El agente en caché se comparte entre sesiones del navegador y eso es seguro
    porque cada sesión usa su propio thread_id independiente para aislar la
    memoria conversacional en el checkpointer.
    """
    tools = obtener_tools()
    return construir_agente(tools)


def formatear_indicador_herramientas(tools_invocadas):
    """Genera el indicador visible de herramientas a partir de la lista de tool_calls."""
    if not tools_invocadas:
        return "ℹ️ *Sin herramientas invocadas en este turno.*"

    nombres_formateados = []
    for t in tools_invocadas:
        if t == "clasificar_sentimiento_resena":
            nombres_formateados.append("`@tool #1 clasificar_sentimiento_resena`")
        elif t == "consultar_politica_triaje":
            nombres_formateados.append("`@tool #2 consultar_politica_triaje`")
        else:
            nombres_formateados.append(f"`@{t}`")

    return "🛠️ **Herramientas invocadas:** " + " ➔ ".join(nombres_formateados)


def extraer_detalle_herramientas(mensajes):
    """Extrae argumentos enviados y salidas crudas asociando tool_calls con ToolMessages."""
    detalles = []
    mapa_respuestas = {}

    for msg in mensajes:
        if isinstance(msg, ToolMessage):
            tool_id = getattr(msg, "tool_call_id", None)
            if tool_id:
                mapa_respuestas[tool_id] = msg.content

    for msg in mensajes:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                t_id = tc.get("id")
                t_nombre = tc.get("name", "desconocida")
                t_args = tc.get("args", {})
                t_salida = mapa_respuestas.get(t_id, "Sin respuesta registrada")
                detalles.append({
                    "nombre": t_nombre,
                    "args": t_args,
                    "salida": t_salida,
                })

    return detalles


def procesar_mensaje(agente, texto_usuario):
    """Valida, invoca al agente y registra el turno en ag_historial."""
    texto_limpio = (texto_usuario or "").strip()
    if not texto_limpio:
        return

    aviso_recorte = False
    if len(texto_limpio) > MAX_CARACTERES_RESENA:
        texto_limpio = texto_limpio[:MAX_CARACTERES_RESENA]
        aviso_recorte = True

    # Registrar mensaje del usuario en pantalla
    st.session_state.ag_historial.append({
        "role": "user",
        "contenido": texto_limpio,
        "aviso_recorte": aviso_recorte,
    })

    # Invocar al agente (únicamente con el mensaje nuevo, la memoria la aporta el checkpointer)
    try:
        with st.spinner("El agente está procesando la consulta..."):
            res = responder(agente, texto_limpio, st.session_state.ag_thread_id)

        # Extraer detalle técnico crudo de herramientas
        msgs_turno = res.get("mensajes", [])
        detalle_tools = extraer_detalle_herramientas(msgs_turno)

        st.session_state.ag_historial.append({
            "role": "assistant",
            "contenido": res.get("respuesta", ""),
            "herramientas": res.get("herramientas", []),
            "detalle_tools": detalle_tools,
            "tokens": {
                "input": res.get("input_tokens", 0),
                "output": res.get("output_tokens", 0),
                "total": res.get("total_tokens", 0),
            },
        })

    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
            st.session_state.ag_historial.append({
                "role": "assistant",
                "contenido": "⚠️ **Límite de cuota alcanzado (429):** La API de Google AI Studio ha alcanzado su límite por minuto. Por favor espera unos segundos antes de enviar otro mensaje.",
                "herramientas": [],
                "detalle_tools": [],
                "tokens": {"input": 0, "output": 0, "total": 0},
            })
        else:
            st.session_state.ag_historial.append({
                "role": "assistant",
                "contenido": f"⚠️ **Error al procesar la solicitud:** {type(exc).__name__}. Por favor intenta de nuevo.",
                "herramientas": [],
                "detalle_tools": [],
                "tokens": {"input": 0, "output": 0, "total": 0},
            })


def render_panel_agente():
    """Renderiza el panel interactivo del agente conversacional."""
    # Verificación de clave de entorno
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "❌ **GOOGLE_API_KEY no encontrada.**\n\n"
            "Para activar el agente:\n"
            "1. Copia `.env.example` como `.env`.\n"
            "2. Configura tu clave válida de Google AI Studio: `GOOGLE_API_KEY=tu_clave_aqui`.\n"
            "3. Reinicia la aplicación."
        )
        st.stop()

    # Inicialización de estado de sesión con prefijo ag_
    if "ag_thread_id" not in st.session_state:
        st.session_state.ag_thread_id = nuevo_thread_id()

    if "ag_historial" not in st.session_state:
        st.session_state.ag_historial = []

    # Obtener agente desde caché
    agente = obtener_agente()

    st.subheader("🤖 Agente de Triaje Inteligente")
    st.caption("Asistente neuro-agéntico con clasificación neuronal de sentimiento y protocolos corporativos.")

    # Barra de control de sesión
    col_info, col_btn = st.columns([2, 1])
    with col_info:
        tid_abrev = st.session_state.ag_thread_id[:8] + "..."
        st.markdown(f"**Sesión activa:** `{tid_abrev}` *(Memoria en checkpointer)*")

    with col_btn:
        if st.button("🔄 Reiniciar conversación", help="Genera un nuevo thread_id y vacía el historial de pantalla"):
            st.session_state.ag_thread_id = nuevo_thread_id()
            st.session_state.ag_historial = []
            st.rerun()

    # Puente opcional con el panel del modelo neuronal
    texto_panel_modelo = st.session_state.get("nn_texto", "").strip()
    if texto_panel_modelo:
        st.info(f"📋 **Reseña disponible desde el panel del modelo:** *\"{texto_panel_modelo[:80]}...\"*")
        if st.button("🚀 Enviar al agente la reseña del panel del modelo"):
            prompt_puente = (
                f"Reseña del cliente: '{texto_panel_modelo}'. "
                "Por favor clasifica el sentimiento con la red neuronal y determina la acción correspondiente según las políticas corporativas."
            )
            procesar_mensaje(agente, prompt_puente)
            st.rerun()

    st.divider()

    # Renderizar historial de conversación
    for turno in st.session_state.ag_historial:
        role = turno.get("role", "user")
        with st.chat_message(role):
            if role == "user":
                st.markdown(turno.get("contenido", ""))
                if turno.get("aviso_recorte"):
                    st.warning(f"⚠️ El texto superó los {MAX_CARACTERES_RESENA} caracteres y fue recortado.")
            else:
                st.markdown(turno.get("contenido", ""))

                # Indicador visible de herramientas invocadas (desde tool_calls)
                tools_usadas = turno.get("herramientas", [])
                st.markdown(formatear_indicador_herramientas(tools_usadas))

                # Expander con detalles técnicos crudos de invocación
                detalles = turno.get("detalle_tools", [])
                if detalles:
                    with st.expander("🔍 Detalle técnico de herramientas"):
                        for i, d in enumerate(detalles, 1):
                            st.markdown(f"**Llamada #{i}: `{d['nombre']}`**")
                            st.markdown(f"- **Argumentos enviados:** `{d['args']}`")
                            st.markdown(f"- **Salida cruda (`ToolMessage`):** `{d['salida']}`")

                # Métricas de consumo de tokens
                tokens = turno.get("tokens", {})
                inp = tokens.get("input", 0)
                out = tokens.get("output", 0)
                tot = tokens.get("total", 0)
                st.caption(f"📊 Consumo del turno: **{inp}** in | **{out}** out | **{tot}** total tokens")

    # Entrada de chat
    entrada_usuario = st.chat_input("Escribe una consulta, saludo o reseña para triaje...")
    if entrada_usuario:
        procesar_mensaje(agente, entrada_usuario)
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Triaje de Reseñas — Agente",
        page_icon="🤖",
        layout="wide",
    )
    render_panel_agente()
