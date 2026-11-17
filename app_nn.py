"""Panel Streamlit de la red neuronal (Unidad 4). Copia de trabajo: se fusiona en app.py en la tarea 18.

No usa el agente ni la API de Google. La predicción la hace src.inferencia.predecir_sentimiento.
Estado compartido para la fusión: st.session_state["nn_texto"] y st.session_state["nn_resultado"].
"""
import logging
import re
import time
from pathlib import Path

import streamlit as st

from src.config import SEQ_LEN, UMBRAL_BAJA_CONFIANZA
from src.inferencia import cargar_modelo, predecir_sentimiento

RAIZ = Path(__file__).resolve().parent
RUTA_FALLO_REAL = RAIZ / "assets" / "apuntes" / "fallo_real.md"
RUTA_NOTEBOOK = RAIZ / "notebooks" / "entrenamiento_u4.ipynb"

logger = logging.getLogger("nn_app")

EJEMPLO_POSITIVO = (
    "This movie was absolutely wonderful. The acting was superb, the story kept me "
    "engaged from start to finish, and the ending left me smiling. Highly recommended!"
)
EJEMPLO_NEGATIVO = (
    "This was a terrible waste of time. The plot made no sense, the acting was awful "
    "and I was bored the whole way through. I regret watching it."
)


@st.cache_resource(show_spinner="Cargando la red neuronal...")
def obtener_modelo():
    """Único punto de la app que carga pesos; se ejecuta una vez por proceso."""
    t0 = time.perf_counter()
    modelo = cargar_modelo()
    logger.info("Modelo cargado de disco en %.0f ms", (time.perf_counter() - t0) * 1000)
    return modelo


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
    st.header("Red neuronal de sentimiento (Unidad 4)")
    info = leer_info_modelo()
    st.caption(info or "Información del modelo no disponible (no se encontró el notebook).")
    st.caption("Este panel muestra la salida directa del modelo; no usa la API de Google.")

    caso_b = leer_caso_b()
    c1, c2, c3 = st.columns(3)
    c1.button("Ejemplo positivo", key="nn_btn_pos", width="stretch",
              on_click=_usar_ejemplo, args=(EJEMPLO_POSITIVO,))
    c2.button("Ejemplo negativo", key="nn_btn_neg", width="stretch",
              on_click=_usar_ejemplo, args=(EJEMPLO_NEGATIVO,))
    c3.button("Caso B (baja confianza)", key="nn_btn_casob", width="stretch",
              on_click=_usar_ejemplo, args=(caso_b,), disabled=caso_b is None,
              help="Texto leído de assets/apuntes/fallo_real.md")

    texto = st.text_area("Reseña (en inglés)", key="nn_entrada", height=160,
                         placeholder="Write a movie review in English...")
    n_palabras = len(texto.split())
    st.caption(f"{len(texto)} caracteres · {n_palabras} palabras")
    if n_palabras > SEQ_LEN:
        st.warning(f"La reseña tiene {n_palabras} palabras: el modelo solo lee las primeras {SEQ_LEN}.")

    if st.button("Clasificar", key="nn_btn_clasificar", type="primary"):
        if not texto.strip():
            st.session_state["nn_resultado"] = None
            st.warning("Escribe una reseña antes de clasificar.")
        else:
            obtener_modelo()  # carga (o reutiliza) el modelo en caché
            t0 = time.perf_counter()
            resultado = predecir_sentimiento(texto)
            st.session_state["nn_ms"] = (time.perf_counter() - t0) * 1000
            st.session_state["nn_texto"] = texto
            st.session_state["nn_resultado"] = resultado

    resultado = st.session_state.get("nn_resultado")
    if resultado:
        p_pos = resultado["p_positivo"]
        m1, m2 = st.columns(2)
        m1.metric("Categoría", resultado["sentimiento"])
        m2.metric("Confianza", f"{resultado['confianza'] * 100:.2f} %")
        st.progress(min(max(float(resultado["confianza"]), 0.0), 1.0))
        st.write(f"p_positivo = **{p_pos:.4f}** · p_negativo = **{1 - p_pos:.4f}**")
        st.caption(f"Tiempo de inferencia: {st.session_state.get('nn_ms', 0.0):.1f} ms")
        if resultado["baja_confianza"]:
            st.warning(f"La certeza está por debajo del umbral de {UMBRAL_BAJA_CONFIANZA:.2f}: "
                       "conviene revisión humana.")
        with st.expander("Salida cruda del modelo (Unidad 4)"):
            st.json(resultado)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    st.set_page_config(page_title="Panel red neuronal", page_icon="🧠", layout="centered")
    render_panel_modelo()
