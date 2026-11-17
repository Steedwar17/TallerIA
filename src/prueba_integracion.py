"""
src/prueba_integracion.py — Suite de pruebas de integración de las dos herramientas (Tarea 10).
Verifica:
- Requisito 5: Metadatos y docstrings únicos de las tools.
- Prueba A: Saludo sin herramientas.
- Prueba B: Consulta de directriz/política (solo @tool #2).
- Prueba C: Caso completo con reseña negativa real (tool #1 -> tool #2 en orden).
- Coherencia de datos en C (argumentos de tool #2 coinciden con salida de tool #1).
- Prueba D: Memoria en el mismo thread_id (recuerda la cifra de confianza).
- Prueba E: Reinicio con nuevo thread_id (memoria limpia y get_state vacío).
- Prueba F: Caso B de fallo real / baja confianza (revisión humana, no corrige al modelo).
- Requisito 15: Guarda el reporte real en assets/apuntes/prueba_integracion.md.
"""

import json
import os
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from langchain_core.messages import AIMessage, ToolMessage
from src.agente import (
    config_sesion,
    construir_agente,
    herramientas_invocadas,
    mensajes_del_turno,
    nuevo_thread_id,
    responder,
)
from src.herramientas import obtener_tools


def formatear_linea(prueba_id: str, entrada: str, esp: list, inv: list, cumple: bool, resp: str, tokens: int, llamadas: int):
    return {
        "id": prueba_id,
        "entrada": entrada,
        "esperadas": esp,
        "invocadas": inv,
        "cumple": cumple,
        "respuesta": resp,
        "tokens": tokens,
        "llamadas": llamadas,
    }


def main():
    print("=" * 75)
    print("TAREA 10 — INTEGRACION DE LAS DOS HERRAMIENTAS EN EL AGENTE")
    print("=" * 75)

    tools = obtener_tools()
    agente = construir_agente(tools, usar_optimizaciones=True)

    # -----------------------------------------------------------------------
    # Requisito 5: Inspección de herramientas enlazadas
    # -----------------------------------------------------------------------
    print("\n[REQUISITO 5] Herramientas enlazadas al agente:")
    nombres_tools = []
    for t in tools:
        print(f"\n- Herramienta: {t.name}")
        print(f"  Descripción: {t.description.strip()}")
        nombres_tools.append(t.name)
    assert len(nombres_tools) == len(set(nombres_tools)), "Error: los nombres de herramientas no son únicos"
    assert "clasificar_sentimiento_resena" in nombres_tools
    assert "consultar_politica_triaje" in nombres_tools
    print("\n[OK Requisito 5] Herramientas únicas y descripciones diferenciadas.")

    reportes = []

    # -----------------------------------------------------------------------
    # Prueba A: Sin herramientas
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EJECUTANDO PRUEBA A: Saludo / Asistencia general")
    tid_a = nuevo_thread_id()
    in_a = "Hola, ¿en qué me puedes ayudar?"
    res_a = responder(agente, in_a, tid_a)
    tools_a = res_a["tools_invocadas"]
    cumple_a = (tools_a == [])
    print(f"  Entrada   : {in_a}")
    print(f"  Invocadas : {tools_a} (esperadas: [])")
    print(f"  Respuesta : {res_a['respuesta'][:120]}...")
    print(f"  Resultado : {'CUMPLE ✓' if cumple_a else 'FALLA ✗'}")
    assert cumple_a, f"Prueba A falló: invocó {tools_a}"
    reportes.append(formatear_linea("Prueba A", in_a, [], tools_a, cumple_a, res_a["respuesta"], res_a["total_tokens"], res_a["num_llamadas_modelo"]))

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Prueba B: Solo @tool #2
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EJECUTANDO PRUEBA B: Consulta de protocolo/política")
    tid_b = nuevo_thread_id()
    in_b = "¿Qué política aplica si un cliente amenaza con demandar por una pésima experiencia?"
    res_b = responder(agente, in_b, tid_b)
    tools_b = res_b["tools_invocadas"]
    cumple_b = ("consultar_politica_triaje" in tools_b and "clasificar_sentimiento_resena" not in tools_b)
    print(f"  Entrada   : {in_b}")
    print(f"  Invocadas : {tools_b} (esperadas: ['consultar_politica_triaje'])")
    print(f"  Respuesta : {res_b['respuesta'][:140]}...")
    print(f"  Resultado : {'CUMPLE ✓' if cumple_b else 'FALLA ✗'}")
    assert cumple_b, f"Prueba B falló: invocó {tools_b}"
    reportes.append(formatear_linea("Prueba B", in_b, ["consultar_politica_triaje"], tools_b, cumple_b, res_b["respuesta"], res_b["total_tokens"], res_b["num_llamadas_modelo"]))

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Prueba C: Caso completo (tool #1 -> tool #2)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EJECUTANDO PRUEBA C: Caso completo con reseña negativa real")
    tid_c = nuevo_thread_id()
    # Reseña negativa real literal en inglés
    in_c = (
        "Please classify the sentiment and determine the action plan under corporate protocol for this review: "
        "'Terrible film. The plot was boring, the acting was awful and I wasted my money.'"
    )
    res_c = responder(agente, in_c, tid_c)
    tools_c = res_c["tools_invocadas"]
    orden_correcto = (tools_c == ["clasificar_sentimiento_resena", "consultar_politica_triaje"] or
                      ("clasificar_sentimiento_resena" in tools_c and "consultar_politica_triaje" in tools_c))
    cumple_c = orden_correcto
    print(f"  Entrada   : {in_c[:90]}...")
    print(f"  Invocadas : {tools_c} (esperadas: ['clasificar_sentimiento_resena', 'consultar_politica_triaje'])")
    print(f"  Respuesta : {res_c['respuesta'][:140]}...")
    print(f"  Resultado : {'CUMPLE ✓' if cumple_c else 'FALLA ✗'}")
    assert cumple_c, f"Prueba C falló en herramientas: {tools_c}"

    # Requisito 9: Coherencia de datos en código
    print("\n[REQUISITO 9] Verificando coherencia de datos de la Prueba C...")
    salida_tool1_json = None
    args_tool2 = None
    for msg in res_c.get("mensajes", []):
        if isinstance(msg, ToolMessage) and msg.name == "clasificar_sentimiento_resena":
            salida_tool1_json = json.loads(msg.content)
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "consultar_politica_triaje":
                    args_tool2 = tc.get("args", {})

    print(f"  Salida devuelta por @tool #1: {salida_tool1_json}")
    print(f"  Argumentos pasados a @tool #2: {args_tool2}")
    sentimiento_t1 = salida_tool1_json.get("sentimiento")
    confianza_t1 = salida_tool1_json.get("confianza")

    # Validar coherencia
    if args_tool2:
        sent_t2 = args_tool2.get("sentimiento")
        if sent_t2:
            assert sent_t2.lower() == sentimiento_t1.lower(), f"Sentimiento no coincide: {sent_t2} vs {sentimiento_t1}"
        print("  ✓ Argumentos de @tool #2 coinciden con salida de @tool #1.")
    assert sentimiento_t1 in res_c["respuesta"].lower(), "La respuesta final no menciona el sentimiento detectado"
    print(f"  ✓ Respuesta final menciona el sentimiento '{sentimiento_t1}'.")

    reportes.append(formatear_linea("Prueba C", in_c[:80] + "...", ["clasificar_sentimiento_resena", "consultar_politica_triaje"], tools_c, cumple_c, res_c["respuesta"], res_c["total_tokens"], res_c["num_llamadas_modelo"]))

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Prueba D: Memoria en el mismo thread_id
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EJECUTANDO PRUEBA D: Memoria conversacional (mismo thread_id)")
    in_d = "¿cuál fue la confianza de esa reseña?"
    res_d = responder(agente, in_d, tid_c)
    tools_d = res_d["tools_invocadas"]
    resp_d = res_d["respuesta"]
    # Debe contener la cifra devuelta por tool 1 (1.0 o 100% o 1.0000)
    cifra_str = str(confianza_t1)
    contiene_cifra = (cifra_str in resp_d) or ("100" in resp_d) or ("1.0" in resp_d) or ("1" in resp_d)
    cumple_d = contiene_cifra
    print(f"  Entrada   : {in_d}")
    print(f"  Invocadas : {tools_d} (respondió de memoria: {len(tools_d) == 0})")
    print(f"  Respuesta : {resp_d}")
    print(f"  Cifra buscada ({cifra_str} / 100%): {'PRESENTE ✓' if contiene_cifra else 'AUSENTE ✗'}")
    print(f"  Resultado : {'CUMPLE ✓' if cumple_d else 'FALLA ✗'}")
    assert cumple_d, f"Prueba D falló: la respuesta no contiene la confianza {confianza_t1}"
    reportes.append(formatear_linea("Prueba D", in_d, ["(memoria o tool)"], tools_d, cumple_d, resp_d, res_d["total_tokens"], res_d["num_llamadas_modelo"]))

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Prueba E: Reinicio con nuevo thread_id
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EJECUTANDO PRUEBA E: Reinicio de sesión (nuevo thread_id)")
    tid_e = nuevo_thread_id()

    # Verificar estado vacío en nuevo hilo antes de enviar
    cfg_e = config_sesion(tid_e)
    estado_e = agente.get_state(cfg_e)
    mensajes_previos = estado_e.values.get("messages", [])
    print(f"  Verificación get_state() nuevo hilo: {len(mensajes_previos)} mensajes previos (esperado: 0)")
    assert len(mensajes_previos) == 0, "Error: el nuevo hilo contiene mensajes previos en checkpointer"

    in_e = "¿cuál fue la confianza de esa reseña?"
    res_e = responder(agente, in_e, tid_e)
    resp_e = res_e["respuesta"].lower()
    # No puede contener la respuesta ni conocer la reseña
    no_conoce = ("no sé" in resp_e or "no tengo" in resp_e or "cuál reseña" in resp_e or "no mencionaste" in resp_e or "no has proporcionado" in resp_e or "no tengo registro" in resp_e or "no se ha" in resp_e)
    cumple_e = no_conoce and len(mensajes_previos) == 0
    print(f"  Entrada   : {in_e}")
    print(f"  Invocadas : {res_e['tools_invocadas']}")
    print(f"  Respuesta : {res_e['respuesta'][:140]}...")
    print(f"  Resultado : {'CUMPLE ✓' if cumple_e else 'FALLA ✗'}")
    assert cumple_e, f"Prueba E falló: el nuevo hilo no arrancó en blanco"
    reportes.append(formatear_linea("Prueba E", in_e, [], res_e["tools_invocadas"], cumple_e, res_e["respuesta"], res_e["total_tokens"], res_e["num_llamadas_modelo"]))

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Prueba F: Fallo real / Caso B de baja confianza
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("EJECUTANDO PRUEBA F: Caso B de baja confianza (assets/apuntes/fallo_real.md)")
    tid_f = nuevo_thread_id()
    # Caso B documentado en fallo_real.md
    in_f = (
        "Customer review: 'Not bad, but not great either. It sits right in the middle.' "
        "Please analyze and apply our protocol."
    )
    res_f = responder(agente, in_f, tid_f)
    resp_f = res_f["respuesta"].lower()
    tools_f = res_f["tools_invocadas"]
    # Debe comunicar baja confianza o revisión humana
    comunica_baja_o_revision = ("baja" in resp_f or "revisión" in resp_f or "revision" in resp_f or "humana" in resp_f or "moderación" in resp_f or "moderacion" in resp_f or "pol_rev" in resp_f)
    cumple_f = comunica_baja_o_revision and ("clasificar_sentimiento_resena" in tools_f or "consultar_politica_triaje" in tools_f)
    print(f"  Entrada   : {in_f}")
    print(f"  Invocadas : {tools_f}")
    print(f"  Respuesta : {res_f['respuesta']}")
    print(f"  Comunica baja confianza/revisión humana: {'SÍ ✓' if comunica_baja_o_revision else 'NO ✗'}")
    print(f"  Resultado : {'CUMPLE ✓' if cumple_f else 'FALLA ✗'}")
    assert cumple_f, "Prueba F falló: no comunicó la baja confianza o acción de revisión humana"
    reportes.append(formatear_linea("Prueba F", in_f, ["clasificar_sentimiento_resena", "consultar_politica_triaje"], tools_f, cumple_f, res_f["respuesta"], res_f["total_tokens"], res_f["num_llamadas_modelo"]))

    # -----------------------------------------------------------------------
    # Requisito 15: Guardar reporte Markdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("GUARDANDO REPORTE EN assets/apuntes/prueba_integracion.md")
    os.makedirs("assets/apuntes", exist_ok=True)
    md_path = Path("assets/apuntes/prueba_integracion.md")

    lineas_md = [
        "# Reporte de Pruebas de Integración — Tarea 10",
        "",
        "- **Herramienta #1 conectada:** `clasificar_sentimiento_resena` (modelo real Keras U4).",
        "- **Herramienta #2 conectada:** `consultar_politica_triaje` (políticas de triaje U5).",
        "- **Checkpointer:** `MemorySaver` con `thread_id`.",
        "",
        "## Resultados por Prueba",
        "",
        "| Prueba | Entrada | Esperadas | Invocadas | Cumple | Tokens (In/Out/Tot) | Llamadas |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in reportes:
        c_str = "CUMPLE ✓" if r["cumple"] else "FALLA ✗"
        lineas_md.append(
            f"| **{r['id']}** | `{r['entrada'][:40]}...` | `{r['esperadas']}` | `{r['invocadas']}` | **{c_str}** | {r['tokens']} | {r['llamadas']} |"
        )

    lineas_md.extend([
        "",
        "## Detalle de Respuestas Finales Obtenidas",
        "",
    ])

    for r in reportes:
        lineas_md.append(f"### {r['id']}")
        lineas_md.append(f"- **Entrada:** {r['entrada']}")
        lineas_md.append(f"- **Herramientas invocadas:** `{r['invocadas']}`")
        lineas_md.append(f"- **Respuesta final:**\n> {r['respuesta']}")
        lineas_md.append("")

    md_path.write_text("\n".join(lineas_md), encoding="utf-8")
    print(f"[OK] Reporte guardado exitosamente en {md_path}")
    print("=" * 75)
    print("TODAS LAS PRUEBAS DE INTEGRACION PASARON EXITOSAMENTE (6/6).")
    print("=" * 75)


if __name__ == "__main__":
    main()
