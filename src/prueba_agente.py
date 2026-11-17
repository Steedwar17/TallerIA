"""
src/prueba_agente.py — Pruebas de integración y regresión del agente de triaje.
Verifica:
1. Saludo sin herramientas.
2. Reseña completa con tools.
3. Memoria reciente con mismo thread_id ("me llamo Ana").
4. Reinicio de memoria con nuevo thread_id.
5. Conversación de 12 turnos sin error de secuencia de mensajes por el recorte.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.agente import (
    MAX_MENSAJES_HISTORIAL,
    config_sesion,
    construir_agente,
    crear_prompt_modifier,
    nuevo_thread_id,
    responder,
)
from src.herramientas import obtener_tools


def ejecutar_pruebas():
    print("=" * 70)
    print("EJECUTANDO PRUEBAS DE REGRESION: src/prueba_agente.py")
    print("=" * 70)

    tools = obtener_tools()
    agente = construir_agente(tools, usar_optimizaciones=True)
    print("[INIT] Agente construido con herramientas:", [t.name for t in tools])

    # 1. Saludo (0 tools)
    tid1 = nuevo_thread_id()
    r1 = responder(agente, "Hola, buenos dias", tid1)
    print("\n[TEST 1 — Saludo]")
    print("  Respuesta:", r1["respuesta"][:100], "...")
    print("  Tools invocadas:", r1["tools_invocadas"])
    assert len(r1["tools_invocadas"]) == 0, f"Error: saludo no debia invocar tools: {r1['tools_invocadas']}"
    print("  Validacion: PASS ✓ (0 tools invocadas)")

    # 2. Reseña completa (debe invocar tools)
    tid2 = nuevo_thread_id()
    msg_resena = (
        "This film was absolutely terrible and boring, awful script and horrible acting. "
        "What is the required action and corporate protocol?"
    )
    r2 = responder(agente, msg_resena, tid2)
    print("\n[TEST 2 — Resena]")
    print("  Respuesta:", r2["respuesta"][:120], "...")
    print("  Tools invocadas:", r2["tools_invocadas"])
    assert "clasificar_sentimiento_resena" in r2["tools_invocadas"] or "consultar_politica_triaje" in r2["tools_invocadas"], "Error: debia invocar al menos una tool"
    print("  Validacion: PASS ✓ (tools activadas correctamente)")

    # 3. Memoria reciente ("me llamo Ana")
    tid3 = nuevo_thread_id()
    r3_t1 = responder(agente, "Hola, me llamo Ana y soy supervisora de atencion.", tid3)
    r3_t2 = responder(agente, "¿Como me llamo y que cargo tengo?", tid3)
    print("\n[TEST 3 — Memoria 'me llamo Ana']")
    print("  Turno 2 respuesta:", r3_t2["respuesta"])
    recuerda_ana = "ana" in r3_t2["respuesta"].lower()
    assert recuerda_ana, f"Error: no recordo el nombre Ana en turno 2: {r3_t2['respuesta']}"
    print("  Validacion: PASS ✓ (recuerda contexto entre turnos con mismo thread_id)")

    # 4. Reinicio de memoria con nuevo thread_id
    tid4 = nuevo_thread_id()
    r4 = responder(agente, "¿Recuerdas mi nombre o mi cargo?", tid4)
    print("\n[TEST 4 — Reinicio de thread_id]")
    print("  Respuesta con nuevo thread_id:", r4["respuesta"][:120], "...")
    olvido = "ana" not in r4["respuesta"].lower()
    assert olvido, "Error: recordo datos de otra sesion en un thread_id nuevo"
    print("  Validacion: PASS ✓ (nuevo thread_id inicia en blanco)")

    # 5. Conversación de 12 turnos (prueba de robustez del recorte de historial)
    print("\n[TEST 5 — Conversacion de 12 turnos con recorte de historial]")
    # Verificamos la secuencia estructural de 12 turnos con tool_calls y mensajes humanos
    mod_fn = crear_prompt_modifier("Prompt de sistema", max_mensajes=MAX_MENSAJES_HISTORIAL)
    historial_12_turnos = []
    for t in range(1, 13):
        historial_12_turnos.append(HumanMessage(content=f"Mensaje usuario {t}"))
        if t % 3 == 0:
            historial_12_turnos.append(AIMessage(content="", tool_calls=[{"name": "tool_x", "args": {}, "id": f"tc_{t}"}]))
            historial_12_turnos.append(ToolMessage(content="resultado", tool_call_id=f"tc_{t}"))
        historial_12_turnos.append(AIMessage(content=f"Respuesta agente {t}"))

    # Aplicar el recorte sobre la conversación de 12 turnos (más de 24 mensajes acumulados)
    estado_simulado = {"messages": historial_12_turnos}
    mensajes_preparados = mod_fn(estado_simulado)

    # Validaciones estructurales del recorte
    assert len(mensajes_preparados) >= 2, "Debe tener al menos SystemMessage + 1 mensaje"
    assert type(mensajes_preparados[0]).__name__ == "SystemMessage", "El primero debe ser SystemMessage"
    assert type(mensajes_preparados[1]).__name__ == "HumanMessage", f"El primer mensaje tras SystemMessage debe ser HumanMessage, fue {type(mensajes_preparados[1]).__name__}"
    # Verificar que ningún ToolMessage quedó huérfano
    tool_calls_ids = {tc["id"] for m in mensajes_preparados if isinstance(m, AIMessage) and m.tool_calls for tc in m.tool_calls}
    for m in mensajes_preparados:
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in tool_calls_ids, f"ToolMessage huérfano detectado: {m.tool_call_id}"

    # Adicionalmente ejecutamos 2 turnos reales adicionales en el agente live para confirmar que la API lo acepta
    r_live1 = responder(agente, "Continuacion turno 11 de prueba", tid3)
    r_live2 = responder(agente, "Continuacion turno 12 de prueba", tid3)
    assert r_live1["respuesta"] and r_live2["respuesta"], "El agente debe responder con éxito"
    print(f"  Total mensajes generados en 12 turnos simulados: {len(historial_12_turnos)}")
    print(f"  Mensajes enviados al modelo tras recorte: {len(mensajes_preparados)} (System + {len(mensajes_preparados)-1} mensajes)")
    print("  Validacion: PASS ✓ (12 turnos procesados sin tool_calls rotos ni errores de secuencia)")

    print("\n" + "=" * 70)
    print("TODAS LAS PRUEBAS DE PRUEBA_AGENTE PASARON EXITOSAMENTE (5/5).")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_pruebas()
