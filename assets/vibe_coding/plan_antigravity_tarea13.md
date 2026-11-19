# Plan generado por el agente de Antigravity: Tarea 13 — Evaluación del agente, casos 1 y 2

> Evidencia de uso de IA (Vibe Coding). Texto copiado tal cual del plan que produjo el agente de Antigravity; no es redacción propia del equipo.

## Hallazgos de la inspección (reales)
- Rama actual `main` (incluye tarea 10, commit 93749b6). `steedwar/modelo` no existe. **Decisión tuya: no crear ramas ni commits**; trabajo en el árbol actual, sin `git add/commit`.
- Versiones (`uv pip list --python .venv/bin/python`; el venv no tiene `pip`): langchain 1.5.0, langgraph 1.2.22, nbconvert 7.17.2, nbformat 5.11.1, pandas 3.1.1, Python 3.12.14. `langchain-google-genai` no apareció en el grep (empieza con `langchain-g…`, lo confirmaré en ejecución).
- `src/config.py`: `GEMINI_MODEL="gemini-3.5-flash-lite"`, `RPM_GRATUITO=15` → pausa entre casos ≥ 60/15 = 4 s (uso 5 s). `SEQ_LEN=585` (difiere del contrato por la tarea 5; ya documentado, no lo toco).
- `responder()` devuelve dict con `respuesta`, `input/output/total_tokens` y `mensajes` (mensajes del turno). Uso `mensajes` para trayectoria y ToolMessage.
- `herramientas_invocadas()` deduplica nombres: limitación conocida, se anotará en el reporte (decisión tuya, no se toca `src/agente.py`).
- Catálogo real de políticas: `POL_NEG`, `POL_POS`, `POL_REV` (no existe POL-06 ni política legal). Para el caso 2 la tool devolverá probablemente `POL_REV`; el verificador solo comprueba pol_id/acción en la respuesta. Si el catálogo no cubre el caso, es hallazgo para Juan Diego.
- `assets/vibe_coding/` no existe (lo crearé solo si hay que dejar captura; la captura la tomas tú).

## Cambios (solo archivos permitidos)
### [NEW] src/evaluacion.py
- `evaluar_caso(agente, nombre, entrada, esperadas, verificar_resultado) -> dict`:
  - `thread_id = nuevo_thread_id()` por caso; `out = responder(agente, entrada, thread_id)`.
  - `msgs = mensajes_del_turno(out["mensajes"])`; `invocadas = herramientas_invocadas(msgs)` (de `AIMessage.tool_calls`).
  - `trayectoria_ok = (invocadas == esperadas)` (listas, orden); no lee la respuesta.
  - `tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]`; `resultado_ok, detalle = verificar_resultado(respuesta, tool_msgs)`.
  - Devuelve: nombre, entrada, esperadas, invocadas, trayectoria_ok, resultado_ok, detalle_resultado, respuesta, tokens (total_tokens).
  - Reintento ante 429: captura `ResourceExhausted`/`ClientError 429`, espera el retardo indicado en el error (fallback 60 s) y repite solo ese caso (máx. 2 reintentos).
- `imprimir_caso(resultado)` con rótulos exactos: Caso, Entrada, Herramientas esperadas, Herramientas invocadas, Trayectoria: CUMPLE/NO CUMPLE, Resultado: CUMPLE/NO CUMPLE (motivo), Respuesta final, Tokens.
- Verificadores deterministas reutilizables: `verificar_caso_1` (respuesta no vacía, 0 ToolMessage, sin `\d+\s*%` ni "confianza" cerca de una cifra) y `verificar_caso_2` (extrae `pol_id` y `accion` del JSON del ToolMessage; CUMPLE si alguno aparece en la respuesta, sin distinguir mayúsculas; también acepta `POL-06` ↔ `POL_06`).
- `pausa_entre_casos()` usando `RPM_GRATUITO`; `guardar_resultados(lista, ruta)`.

### [NEW] notebooks/evaluacion_casos_1_2.ipynb (desde cero)
1. Celda de entorno: versiones langchain/langgraph/langchain-google-genai (importlib.metadata), `GEMINI_MODEL`, nombre+descripción de las 2 tools de `obtener_tools()`; no imprime la clave (solo confirma si existe, sin valor).
2. `## Caso 1` (markdown) + celda: `evaluar_caso` + `imprimir_caso`.
3. `## Caso 2` + celda análoga (pausa antes).
4. Tabla resumen pandas (caso, esperadas, invocadas, trayectoria, resultado) + guardado del JSON.
Sin lógica de evaluación en el notebook.

### [NEW] assets/apuntes/resultados_casos_1_2.json
Generado por la ejecución real.

### [MODIFY] assets/apuntes/bitacora_vibe_coding.md
Solo si cometo un error real (te aviso en ese momento para la captura). Si un caso da NO CUMPLE, el intento fallido se anota tras el ajuste y re-ejecución.

## Verificación
```
.venv/bin/python -c "import src.evaluacion"
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/evaluacion_casos_1_2.ipynb
.venv/bin/python - <<'EOF'   # nbformat: todas las celdas de código con salida y sin error
EOF
```
Tests offline del verificador/evaluar_caso con un agente simulado (sin cuota) antes de la ejecución real. Una sola ejecución real por caso (2 llamadas al agente, más el 429 si ocurre).
Si un caso da NO CUMPLE: queda así; diagnóstico con tool_calls y argumentos reales; te informo la causa y a quién (Juan Diego: docstring/system prompt).

## Reporte final
Tabla "Cumplimiento Tarea 13" (13 requisitos) con evidencia real, totales, pendientes y comandos.
