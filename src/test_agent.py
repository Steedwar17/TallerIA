"""
test_agent.py — Pruebas de integración del Agente de Triaje Inteligente.
Verifica las 4 demos obligatorias del spec (Tarea 6):
  1. Saludo sin invocar herramientas.
  2. Reseña que activa @tool #1 y @tool #2.
  3. Memoria entre turnos con el mismo thread_id.
  4. Reinicio de memoria con un thread_id nuevo.
"""
import sys
from pathlib import Path

# Agregar directorio raíz al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.agent import (
    crear_agente,
    nuevo_thread_id,
    invocar_agente,
    inferir_sentimiento_resena,
)
from src.tools import consultar_politicas_triaje


def separador(titulo: str):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def imprimir_resultado(turno: int, mensaje: str, resultado: dict):
    print(f"\n[Turno {turno}] Usuario: {mensaje}")
    print(f"[Turno {turno}] Agente:   {resultado['respuesta']}")
    tools = resultado["tools_invocadas"]
    if tools:
        print(f"[Turno {turno}] Tools invocadas: {tools}")
    else:
        print(f"[Turno {turno}] Tools invocadas: (ninguna)")


def main():
    print("\n" + "=" * 70)
    print("  PRUEBAS DE INTEGRACIÓN — AGENTE DE TRIAJE INTELIGENTE (Tarea 6)")
    print("=" * 70)

    # --- Crear el agente con ambas tools (provisional + tool #2) ---
    print("\n[INIT] Creando agente con @tool #1 provisional y @tool #2...")
    try:
        agente = crear_agente([inferir_sentimiento_resena, consultar_politicas_triaje])
        print("[INIT] Agente creado correctamente.")
    except EnvironmentError as e:
        print(f"\n[ERROR] {e}")
        print("Configura tu GOOGLE_API_KEY en el archivo .env y vuelve a ejecutar.")
        sys.exit(1)

    # ===================================================================
    # DEMO 1: Saludo — el agente NO debe invocar ninguna herramienta
    # ===================================================================
    separador("DEMO 1: Saludo (esperado: 0 tools invocadas)")

    tid_demo1 = nuevo_thread_id()
    res1 = invocar_agente(agente, "Hola, ¿qué puedes hacer por mí?", tid_demo1)
    imprimir_resultado(1, "Hola, ¿qué puedes hacer por mí?", res1)

    tools_demo1 = res1["tools_invocadas"]
    print(f"\n[VALIDACIÓN Demo 1] Tools invocadas: {tools_demo1}")
    print(f"[VALIDACIÓN Demo 1] Sin herramientas: {'PASS ✓' if len(tools_demo1) == 0 else 'FAIL ✗'}")

    # ===================================================================
    # DEMO 2: Reseña negativa — debe invocar @tool #1 y @tool #2
    # ===================================================================
    separador("DEMO 2: Reseña negativa (esperado: tool #1 + tool #2 invocadas)")

    tid_demo2 = nuevo_thread_id()
    mensaje_resena = (
        "Acabo de ver la película y fue terrible. El guión es malo, "
        "los actores son pésimos y la dirección es horrible. "
        "¿Cómo debería manejarse esta reseña según los protocolos?"
    )
    res2 = invocar_agente(agente, mensaje_resena, tid_demo2)
    imprimir_resultado(1, mensaje_resena, res2)

    tools_demo2 = res2["tools_invocadas"]
    tiene_tool1 = "inferir_sentimiento_resena" in tools_demo2
    tiene_tool2 = "consultar_politicas_triaje" in tools_demo2
    print(f"\n[VALIDACIÓN Demo 2] Tools invocadas: {tools_demo2}")
    print(f"[VALIDACIÓN Demo 2] @tool #1 invocada: {'PASS ✓' if tiene_tool1 else 'FAIL ✗'}")
    print(f"[VALIDACIÓN Demo 2] @tool #2 invocada: {'PASS ✓' if tiene_tool2 else 'FAIL ✗'}")

    # ===================================================================
    # DEMO 3: Memoria entre turnos — mismo thread_id
    # ===================================================================
    separador("DEMO 3: Memoria entre turnos (mismo thread_id)")

    tid_memoria = nuevo_thread_id()
    print(f"[INFO] thread_id de sesión: {tid_memoria}\n")

    # Turno 1: presentarse
    msg_t1 = "Hola, mi nombre es Juan Diego y trabajo en el área de moderación de contenido."
    res_t1 = invocar_agente(agente, msg_t1, tid_memoria)
    imprimir_resultado(1, msg_t1, res_t1)

    # Turno 2: preguntar algo que requiere memoria del turno 1
    msg_t2 = "¿Recuerdas cómo me llamo y en qué área trabajo?"
    res_t2 = invocar_agente(agente, msg_t2, tid_memoria)
    imprimir_resultado(2, msg_t2, res_t2)

    recuerda = "juan" in res_t2["respuesta"].lower() or "moderación" in res_t2["respuesta"].lower()
    print(f"\n[VALIDACIÓN Demo 3] Agente recuerda contexto previo: {'PASS ✓' if recuerda else 'FAIL ✗'}")

    # ===================================================================
    # DEMO 4: Nuevo thread_id — la memoria debe reiniciarse
    # ===================================================================
    separador("DEMO 4: Nuevo thread_id (memoria en blanco)")

    tid_nuevo = nuevo_thread_id()
    print(f"[INFO] thread_id NUEVO (distinto): {tid_nuevo}\n")

    msg_reset = "¿Recuerdas cómo me llamo o en qué área trabajo?"
    res_reset = invocar_agente(agente, msg_reset, tid_nuevo)
    imprimir_resultado(1, msg_reset, res_reset)

    respuesta_reset = res_reset["respuesta"].lower()
    # El agente no debería mencionar "Juan Diego" ni "moderación" sin que se lo hayan dicho
    no_recuerda = (
        "juan" not in respuesta_reset
        and "moderación" not in respuesta_reset
        and "moderacion" not in respuesta_reset
    )
    print(f"\n[VALIDACIÓN Demo 4] Memoria reiniciada (no recuerda datos previos): {'PASS ✓' if no_recuerda else 'FAIL ✗'}")

    # ===================================================================
    # RESUMEN FINAL
    # ===================================================================
    separador("RESUMEN DE VALIDACIONES")
    resultados = [
        ("Demo 1 — Sin herramientas en saludo", len(tools_demo1) == 0),
        ("Demo 2 — @tool #1 invocada en reseña", tiene_tool1),
        ("Demo 2 — @tool #2 invocada en reseña", tiene_tool2),
        ("Demo 3 — Memoria persiste en mismo thread_id", recuerda),
        ("Demo 4 — Memoria limpia en thread_id nuevo", no_recuerda),
    ]
    for nombre, resultado in resultados:
        estado = "PASS ✓" if resultado else "FAIL ✗"
        print(f"  {estado}  {nombre}")

    total_pass = sum(1 for _, r in resultados if r)
    print(f"\n  Resultado: {total_pass}/{len(resultados)} pruebas pasaron.")


if __name__ == "__main__":
    main()
