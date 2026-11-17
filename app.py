"""app.py — Aplicación Unificada: Red Neuronal de Sentimiento + Agente de Triaje Inteligente.

Taller Evaluativo Unidades 4 y 5: Sistemas Neuro-Agénticos Interactivos (Universidad de Boyacá).
Ruta 3: PLN con Embeddings + Agente de Triaje Inteligente.
Integrantes: Juan Diego Castillo Jiménez y Michael Steedwar Parra Daza.
"""

import logging
import os
import re
import sys
import time
from pathlib import Path

# Asegurar raíz del proyecto en sys.path
RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

from src.config import MAX_CARACTERES_RESENA, SEQ_LEN, UMBRAL_BAJA_CONFIANZA
from src.inferencia import cargar_modelo, predecir_sentimiento
from src.herramientas import obtener_tools
from src.agente import construir_agente, nuevo_thread_id, responder
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger("app_unificada")

RUTA_FALLO_REAL = RAIZ / "assets" / "apuntes" / "fallo_real.md"
RUTA_NOTEBOOK = RAIZ / "notebooks" / "entrenamiento_u4.ipynb"

EJEMPLO_POSITIVO = (
    "This movie was absolutely wonderful. The acting was superb, the story kept me "
    "engaged from start to finish, and the ending left me smiling. Highly recommended!"
)
EJEMPLO_NEGATIVO = (
    "This was a terrible waste of time. The plot made no sense, the acting was awful "
    "and I was bored the whole way through. I regret watching it."
)


# ==============================================================================
# CACHÉ DE RECURSOS (Modelo y Agente cargados una sola vez por proceso)
# ==============================================================================
@st.cache_resource(show_spinner="Cargando la red neuronal...")
def obtener_modelo():
    """Único punto de la app que carga pesos neuronales; se ejecuta una vez por proceso."""
    t0 = time.perf_counter()
    modelo = cargar_modelo()
    logger.info("Modelo cargado de disco en %.0f ms", (time.perf_counter() - t0) * 1000)
    return modelo


@st.cache_resource(show_spinner="Iniciando el agente de triaje...")
def obtener_agente():
    """Compila y almacena en caché el agente con sus herramientas y checkpointer.

    NOTA DE SEGURIDAD / ARQUITECTURA:
    El agente en caché se comparte entre sesiones del navegador y eso es seguro
    porque cada sesión usa su propio thread_id independiente para aislar la
    memoria conversacional en el checkpointer.
    """
    tools = obtener_tools()
    return construir_agente(tools)


# ==============================================================================
# UTILIDADES DEL PANEL DE LA RED NEURONAL (Unidad 4)
# ==============================================================================
@st.cache_data(show_spinner=False)
def leer_caso_b() -> str | None:
    """Texto del Caso B leído de assets/apuntes/fallo_real.md (no pegado en el código)."""
    if not RUTA_FALLO_REAL.is_file():
        return None
    md = RUTA_FALLO_REAL.read_text(encoding="utf-8")
    m = re.search(r"## Caso B.*?\*\*Texto completo:\*\*\s*`([^`]+)`", md, flags=re.S)
    return m.group(1).strip() if m else None


@st.cache_data(show_spinner=False)
def leer_info_modelo() -> str | None:
    """Arquitectura final y accuracy de test, leídas del notebook de entrenamiento."""
    if not RUTA_NOTEBOOK.is_file():
        return None
    crudo = RUTA_NOTEBOOK.read_text(encoding="utf-8")
    arq = re.search(r"Modelo analizado: modelo mejorado \(([^)]+)\)", crudo)
    acc = re.search(r"modelo mejorado \(Dropout\)\s*: test loss = [\d.]+ \| test accuracy = ([\d.]+)", crudo)
    if not (arq and acc):
        return None
    return f"Arquitectura final: {arq.group(1)} · Accuracy en test (25.000 reseñas): {float(acc.group(1)) * 100:.2f} %"


def _usar_ejemplo(texto: str | None):
    if texto:
        st.session_state["nn_entrada"] = texto


def render_panel_modelo():
    """Renderiza el panel de inferencia directa de la red neuronal (Unidad 4)."""
    st.subheader("📊 Inferencia Neuronal Directa (Unidad 4)")
    info = leer_info_modelo()
    st.caption(info or "Información del modelo no disponible.")
    st.caption("Este panel muestra la clasificación directa del modelo local sin invocar agentes ni la API de Google.")

    caso_b = leer_caso_b()
    c1, c2, c3 = st.columns(3)
    c1.button("Ejemplo positivo", key="nn_btn_pos", use_container_width=True,
              on_click=_usar_ejemplo, args=(EJEMPLO_POSITIVO,))
    c2.button("Ejemplo negativo", key="nn_btn_neg", use_container_width=True,
              on_click=_usar_ejemplo, args=(EJEMPLO_NEGATIVO,))
    c3.button("Caso B (baja confianza)", key="nn_btn_casob", use_container_width=True,
              on_click=_usar_ejemplo, args=(caso_b,), disabled=caso_b is None,
              help="Texto leído de assets/apuntes/fallo_real.md")

    texto = st.text_area("Reseña en inglés para evaluación:", key="nn_entrada", height=140,
                         placeholder="Write a movie review in English...")
    n_palabras = len(texto.split())
    st.caption(f"{len(texto)} caracteres · {n_palabras} palabras")
    if n_palabras > SEQ_LEN:
        st.warning(f"La reseña tiene {n_palabras} palabras: el modelo solo lee las primeras {SEQ_LEN}.")

    if st.button("🔍 Clasificar sentimiento", key="nn_btn_clasificar", type="primary"):
        if not texto.strip():
            st.session_state["nn_resultado"] = None
            st.warning("Escribe una reseña antes de clasificar.")
        else:
            obtener_modelo()  # Carga o reutiliza en caché
            t0 = time.perf_counter()
            resultado = predecir_sentimiento(texto)
            st.session_state["nn_ms"] = (time.perf_counter() - t0) * 1000
            st.session_state["nn_texto"] = texto
            st.session_state["nn_resultado"] = resultado

    resultado = st.session_state.get("nn_resultado")
    if resultado:
        p_pos = resultado["p_positivo"]
        m1, m2 = st.columns(2)
        m1.metric("Categoría predicha", resultado["sentimiento"].capitalize())
        m2.metric("Confianza de predicción", f"{resultado['confianza'] * 100:.2f} %")
        st.progress(min(max(float(resultado["confianza"]), 0.0), 1.0))
        st.write(f"Probabilidad positiva ($p$) = **{p_pos:.4f}** · Probabilidad negativa ($1-p$) = **{1 - p_pos:.4f}**")
        st.caption(f"Tiempo de inferencia local: **{st.session_state.get('nn_ms', 0.0):.1f} ms**")

        if resultado["baja_confianza"]:
            st.warning(f"⚠️ La certeza está por debajo del umbral de {UMBRAL_BAJA_CONFIANZA:.2f}: "
                       "se recomienda derivar a moderación / revisión humana.")

        with st.expander("Detalle JSON crudo de la red local"):
            st.json(resultado)

        # Invitación a enviar al agente
        st.info("💡 Puedes enviar esta reseña al **Agente de Triaje** (Pestaña 2) para determinar la política corporativa y acción de respuesta.")


# ==============================================================================
# UTILIDADES DEL PANEL DEL AGENTE CONVERSACIONAL (Unidad 5)
# ==============================================================================
def formatear_indicador_herramientas(tools_invocadas):
    """Genera el indicador visible de herramientas a partir de los tool_calls."""
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


def procesar_mensaje_agente(agente, texto_usuario):
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

    # Invocar al agente (únicamente con el mensaje nuevo; memoria provista por MemorySaver)
    try:
        with st.spinner("El agente está procesando la consulta..."):
            res = responder(agente, texto_limpio, st.session_state.ag_thread_id)

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
    """Renderiza el panel interactivo del agente conversacional (Unidad 5)."""
    # Verificación defensiva de API KEY
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "❌ **GOOGLE_API_KEY no encontrada.**\n\n"
            "Para activar el agente:\n"
            "1. Copia `.env.example` como `.env`.\n"
            "2. Configura tu clave válida de Google AI Studio: `GOOGLE_API_KEY=tu_clave_aqui`.\n"
            "3. Reinicia la aplicación."
        )
        st.stop()

    # Inicialización de estado con prefijo ag_
    if "ag_thread_id" not in st.session_state:
        st.session_state.ag_thread_id = nuevo_thread_id()

    if "ag_historial" not in st.session_state:
        st.session_state.ag_historial = []

    agente = obtener_agente()

    st.subheader("🤖 Agente de Triaje Inteligente (Unidad 5)")
    st.caption("Chat neuro-agéntico con trazabilidad de tools, memoria de corto plazo y políticas corporativas.")

    # Barra superior de sesión
    col_info, col_btn = st.columns([2, 1])
    with col_info:
        tid_abrev = st.session_state.ag_thread_id[:8] + "..."
        st.markdown(f"**Sesión activa:** `{tid_abrev}` *(Memoria en checkpointer)*")

    with col_btn:
        if st.button("🔄 Reiniciar conversación", key="ag_btn_reinicio", help="Genera un nuevo thread_id y vacía el historial de pantalla"):
            st.session_state.ag_thread_id = nuevo_thread_id()
            st.session_state.ag_historial = []
            st.rerun()

    # Puente interactivo con el panel del modelo neuronal
    texto_panel_modelo = st.session_state.get("nn_texto", "").strip()
    if texto_panel_modelo:
        st.info(f"📋 **Reseña cargada desde el panel del modelo:** *\"{texto_panel_modelo[:80]}...\"*")
        if st.button("🚀 Enviar al agente la reseña del panel del modelo", key="ag_btn_puente"):
            prompt_puente = (
                f"Reseña del cliente: '{texto_panel_modelo}'. "
                "Por favor clasifica el sentimiento con la red neuronal y determina la acción correspondiente según las políticas corporativas."
            )
            procesar_mensaje_agente(agente, prompt_puente)
            st.rerun()

    st.divider()

    # Historial de conversación
    for turno in st.session_state.ag_historial:
        role = turno.get("role", "user")
        with st.chat_message(role):
            if role == "user":
                st.markdown(turno.get("contenido", ""))
                if turno.get("aviso_recorte"):
                    st.warning(f"⚠️ El texto superó los {MAX_CARACTERES_RESENA} caracteres y fue recortado.")
            else:
                st.markdown(turno.get("contenido", ""))

                # Indicador visible de herramientas
                tools_usadas = turno.get("herramientas", [])
                st.markdown(formatear_indicador_herramientas(tools_usadas))

                # Expander técnico de invocación
                detalles = turno.get("detalle_tools", [])
                if detalles:
                    with st.expander("🔍 Detalle técnico de herramientas"):
                        for i, d in enumerate(detalles, 1):
                            st.markdown(f"**Llamada #{i}: `{d['nombre']}`**")
                            st.markdown(f"- **Argumentos enviados:** `{d['args']}`")
                            st.markdown(f"- **Salida cruda (`ToolMessage`):** `{d['salida']}`")

                # Métricas de tokens del turno
                tokens = turno.get("tokens", {})
                inp = tokens.get("input", 0)
                out = tokens.get("output", 0)
                tot = tokens.get("total", 0)
                st.caption(f"📊 Consumo del turno: **{inp}** in | **{out}** out | **{tot}** total tokens")

    # Entrada de chat
    entrada_usuario = st.chat_input("Escribe una consulta, saludo o reseña para triaje...", key="ag_chat_input")
    if entrada_usuario:
        procesar_mensaje_agente(agente, entrada_usuario)
        st.rerun()


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
def main():
    st.set_page_config(
        page_title="Sistema Neuro-Agéntico de Triaje",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎬 Sistema Neuro-Agéntico de Triaje de Reseñas de Streaming")
    st.caption("Ruta 3: PLN con Embeddings + Agente de Triaje Inteligente · Universidad de Boyacá")
    st.markdown("---")

    tab_modelo, tab_agente = st.tabs([
        "📊 Clasificación Neuronal Directa (Unidad 4)",
        "🤖 Agente de Triaje con Memoria (Unidad 5)",
    ])

    with tab_modelo:
        render_panel_modelo()

    with tab_agente:
        render_panel_agente()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
