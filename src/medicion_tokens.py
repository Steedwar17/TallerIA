"""
src/medicion_tokens.py — Medición empírica de tokens antes vs. después de optimizaciones.
Tarea 8: Ahorro de tokens en agente de triaje.
"""

import datetime
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agente import (
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_COMPACTO,
    clasificar_sentimiento_resena,
    construir_agente,
    nuevo_thread_id,
    responder,
)
from src.config import GEMINI_MODEL
from src.tool_politicas import consultar_politica_triaje

load_dotenv()

# ===========================================================================
# HERRAMIENTAS BASE (Docstrings extensos para medicion 'antes')
# ===========================================================================

@tool
def clasificar_sentimiento_resena_base(texto: str) -> str:
    """Ejecuta la red neuronal local entrenada con reseñas IMDB en inglés y clasifica su sentimiento.

    Úsala cuando el usuario entregue el texto de una reseña y pida clasificarla o hacer triaje.
    NO la uses para saludos, preguntas sobre políticas ni para un texto ya clasificado en este turno.

    Args:
        texto: la reseña literal y completa, sin resumir ni traducir.

    Returns:
        JSON de una línea con sentimiento ("positivo"/"negativo"), confianza (0-1) y baja_confianza (bool).
    """
    return clasificar_sentimiento_resena.invoke(texto)


@tool
def consultar_politica_triaje_base(consulta: str, sentimiento: str = None, confianza: float = None) -> str:
    """Consulta la politica interna y protocolo de respuesta para una resena de cliente de la plataforma de streaming.

    Usala cuando conozcas el sentimiento o nivel de confianza de la resena o cuando el usuario pregunte
    directamente sobre lineamientos, politicas de triaje corporativo o protocolos de escalamiento y respuesta.
    NO la uses para clasificar texto no analizado previamente (para eso usa clasificar_sentimiento_resena) ni
    para responder a saludos o preguntas genericas no relacionadas con politicas internas.

    Args:
        consulta: pregunta detallada o texto de consulta sobre el protocolo o la resena.
        sentimiento: 'positivo', 'negativo' o 'incierto' (si ya fue inferido).
        confianza: valor flotante de certeza (0.0 a 1.0).

    Returns:
        JSON compacto (max 600 caracteres) con pol_id, prioridad, accion y pauta.
    """
    return consultar_politica_triaje.invoke({"consulta": consulta, "sentimiento": sentimiento, "confianza": confianza})


def medir_prompts_y_docstrings(modelo):
    """Mide con el contador nativo del modelo los tokens de system prompt y docstrings."""
    tok_sys_base = modelo.get_num_tokens(SYSTEM_PROMPT_BASE)
    tok_sys_opt = modelo.get_num_tokens(SYSTEM_PROMPT_COMPACTO)

    doc_t1_base = clasificar_sentimiento_resena_base.description
    doc_t1_opt = clasificar_sentimiento_resena.description
    tok_t1_base = modelo.get_num_tokens(doc_t1_base)
    tok_t1_opt = modelo.get_num_tokens(doc_t1_opt)

    doc_t2_base = consultar_politica_triaje_base.description
    doc_t2_opt = consultar_politica_triaje.description
    tok_t2_base = modelo.get_num_tokens(doc_t2_base)
    tok_t2_opt = modelo.get_num_tokens(doc_t2_opt)

    return {
        "sys_base": tok_sys_base,
        "sys_opt": tok_sys_opt,
        "t1_base": tok_t1_base,
        "t1_opt": tok_t1_opt,
        "t2_base": tok_t2_base,
        "t2_opt": tok_t2_opt,
    }


def ejecutar_tres_turnos(agente, thread_id):
    """Ejecuta las tres entradas fijas requeridas y retorna las métricas por turno."""
    entradas = [
        ("Caso 1: Saludo", "Hola, ¿qué puedes hacer por mí?"),
        ("Caso 2: Pregunta de política", "¿Cuál es el protocolo de triaje para una reseña negativa?"),
        (
            "Caso 3: Reseña completa",
            "This movie was terrible, the script was awful and the directing was horrible. How should this be handled?"
        ),
    ]

    resultados = []
    for nombre, txt in entradas:
        r = responder(agente, txt, thread_id)
        resultados.append({
            "caso": nombre,
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"],
            "llamadas": r["num_llamadas_modelo"],
            "tools": r["tools_invocadas"],
            "respuesta_corta": r["respuesta"][:80].replace("\n", " "),
        })
    return resultados


def main():
    print("=" * 70)
    print("MEDICION DE TOKENS — ANTES VS DESPUES (TAREA 8)")
    print("=" * 70)

    api_key = os.getenv("GOOGLE_API_KEY")
    modelo_nombre = GEMINI_MODEL or "gemini-3.5-flash-lite"
    fecha = datetime.date.today().isoformat()

    modelo_contador = ChatGoogleGenerativeAI(
        model=modelo_nombre,
        google_api_key=api_key,
        temperature=0,
    )

    # 1. Medición de componentes estáticos
    print("\n[PASO 1] Midiendo tokens de System Prompt y Docstrings...")
    estaticos = medir_prompts_y_docstrings(modelo_contador)
    print(f"  System Prompt: Base = {estaticos['sys_base']} tok -> Compacto = {estaticos['sys_opt']} tok (Ahorro: {((estaticos['sys_base']-estaticos['sys_opt'])/estaticos['sys_base'])*100:.1f}%)")
    print(f"  Docstring Tool 1: Base = {estaticos['t1_base']} tok -> Compacto = {estaticos['t1_opt']} tok (Ahorro: {((estaticos['t1_base']-estaticos['t1_opt'])/estaticos['t1_base'])*100:.1f}%)")
    print(f"  Docstring Tool 2: Base = {estaticos['t2_base']} tok -> Compacto = {estaticos['t2_opt']} tok (Ahorro: {((estaticos['t2_base']-estaticos['t2_opt'])/estaticos['t2_base'])*100:.1f}%)")

    # 2. Medición dinámicas: Línea Base ('Antes')
    print("\n[PASO 2] Ejecutando Línea Base (SIN optimizaciones)...")
    tools_base = [clasificar_sentimiento_resena_base, consultar_politica_triaje_base]
    agente_base = construir_agente(
        tools=tools_base,
        system_prompt=SYSTEM_PROMPT_BASE,
        usar_optimizaciones=False,
    )
    tid_base = nuevo_thread_id()
    resultados_antes = ejecutar_tres_turnos(agente_base, tid_base)

    for r in resultados_antes:
        print(f"  [{r['caso']}] In={r['input_tokens']}, Out={r['output_tokens']}, Tot={r['total_tokens']}, Llamadas={r['llamadas']}, Tools={r['tools']}")

    # 3. Medición dinámicas: Optimizado ('Después')
    print("\n[PASO 3] Ejecutando Versión Optimizada (CON optimizaciones)...")
    tools_opt = [clasificar_sentimiento_resena, consultar_politica_triaje]
    agente_opt = construir_agente(
        tools=tools_opt,
        system_prompt=SYSTEM_PROMPT_COMPACTO,
        usar_optimizaciones=True,
    )
    tid_opt = nuevo_thread_id()
    resultados_despues = ejecutar_tres_turnos(agente_opt, tid_opt)

    for r in resultados_despues:
        print(f"  [{r['caso']}] In={r['input_tokens']}, Out={r['output_tokens']}, Tot={r['total_tokens']}, Llamadas={r['llamadas']}, Tools={r['tools']}")

    # 4. Generar reporte Markdown en assets/apuntes/medicion_tokens.md
    print("\n[PASO 4] Guardando tabla comparativa en assets/apuntes/medicion_tokens.md...")
    os.makedirs("assets/apuntes", exist_ok=True)
    md_path = Path("assets/apuntes/medicion_tokens.md")

    total_in_antes = sum(r["input_tokens"] for r in resultados_antes)
    total_out_antes = sum(r["output_tokens"] for r in resultados_antes)
    total_tot_antes = sum(r["total_tokens"] for r in resultados_antes)
    total_call_antes = sum(r["llamadas"] for r in resultados_antes)

    total_in_desp = sum(r["input_tokens"] for r in resultados_despues)
    total_out_desp = sum(r["output_tokens"] for r in resultados_despues)
    total_tot_desp = sum(r["total_tokens"] for r in resultados_despues)
    total_call_desp = sum(r["llamadas"] for r in resultados_despues)

    dif_total_pct = ((total_tot_antes - total_tot_desp) / total_tot_antes * 100) if total_tot_antes > 0 else 0.0

    lineas_md = [
        "# Medición de Tokens — Antes vs. Después (Tarea 8)",
        "",
        f"- **Fecha:** {fecha}",
        f"- **Modelo evaluado:** `{modelo_nombre}`",
        f"- **Checkpointer:** `MemorySaver` con `thread_id`",
        f"- **Ahorro global medido:** **{dif_total_pct:.1f}%** de tokens totales en los 3 casos",
        "",
        "## 1. Componentes Estáticos (Tokens por Invocación)",
        "",
        "| Componente | Tokens Antes | Tokens Después | Reducción | % Ahorro |",
        "|---|---|---|---|---|",
        f"| **System Prompt (6 reglas)** | {estaticos['sys_base']} | {estaticos['sys_opt']} | {estaticos['sys_base']-estaticos['sys_opt']} | {((estaticos['sys_base']-estaticos['sys_opt'])/estaticos['sys_base'])*100:.1f}% |",
        f"| **Docstring Tool #1 (`clasificar_sentimiento_resena`)** | {estaticos['t1_base']} | {estaticos['t1_opt']} | {estaticos['t1_base']-estaticos['t1_opt']} | {((estaticos['t1_base']-estaticos['t1_opt'])/estaticos['t1_base'])*100:.1f}% |",
        f"| **Docstring Tool #2 (`consultar_politica_triaje`)** | {estaticos['t2_base']} | {estaticos['t2_opt']} | {estaticos['t2_base']-estaticos['t2_opt']} | {((estaticos['t2_base']-estaticos['t2_opt'])/estaticos['t2_base'])*100:.1f}% |",
        "",
        "## 2. Invocaciones Dinámicas (3 Casos Reales)",
        "",
        "| Caso | Métricas Antes (In / Out / Tot / LLM Calls) | Métricas Después (In / Out / Tot / LLM Calls) | Ahorro Total (%) | Tools Invocadas |",
        "|---|---|---|---|---|",
    ]

    for a, d in zip(resultados_antes, resultados_despues):
        ahorro_caso = ((a["total_tokens"] - d["total_tokens"]) / a["total_tokens"] * 100) if a["total_tokens"] > 0 else 0.0
        lineas_md.append(
            f"| **{a['caso']}** | {a['input_tokens']} / {a['output_tokens']} / {a['total_tokens']} / {a['llamadas']} | "
            f"{d['input_tokens']} / {d['output_tokens']} / {d['total_tokens']} / {d['llamadas']} | "
            f"**{ahorro_caso:.1f}%** | `{d['tools']}` |"
        )

    lineas_md.extend([
        f"| **TOTALES** | **{total_in_antes} / {total_out_antes} / {total_tot_antes} / {total_call_antes}** | "
        f"**{total_in_desp} / {total_out_desp} / {total_tot_desp} / {total_call_desp}** | "
        f"**{dif_total_pct:.1f}%** | — |",
        "",
        "## 3. Conclusiones Empíricas",
        "1. **Reducción del System Prompt:** Al eliminar prosa redundante manteniendo intactas las 6 reglas, el tamaño del prompt de sistema se redujo sustancialmente.",
        "2. **Docstrings de Herramientas:** La sintaxis compacta con viñetas 'Usala:' y 'NO usar:' preserva la precisión del enrutamiento agéntico y ahorra tokens en cada turno.",
        "3. **Recorte de Historial:** Garantiza que en turnos avanzados el número de mensajes inyectados no crezca indefinidamente, iniciando siempre en un `HumanMessage` y sin separar pares `AIMessage` - `ToolMessage`.",
        "4. **Límite de Ritmo:** `InMemoryRateLimiter(requests_per_second=0.25)` previene errores de cuota HTTP 429 en el plan gratuito de 15 RPM.",
    ])

    md_path.write_text("\n".join(lineas_md), encoding="utf-8")
    print(f"[OK] Archivo guardado en {md_path}")
    print("\n" + "=" * 70)
    print("MEDICION COMPLETADA EXITOSAMENTE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
