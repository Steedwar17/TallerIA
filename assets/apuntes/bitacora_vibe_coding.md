# Bitácora de Vibe Coding — Registro del Equipo

## Episodio 1 — 2026-11-17 — Tarea 9 (src/inferencia.py - Steedwar)
- **Qué propuse:** en `predecir_sentimiento`, llamar `modelo.predict(np.array([[limpio]], dtype=object), verbose=0)`.
- **Por qué estaba mal:** Keras 3 rechaza arreglos NumPy `dtype=object` (`ValueError: Invalid dtype: object`); `src.prueba_inferencia` falló en la primera llamada. Error de API/mala práctica con entradas string.
- **Corrección:** `modelo(tf.constant([[limpio]]), training=False)`. Verificado ejecutando `src.prueba_inferencia`.
- **Captura:** en `assets/vibe_coding/`.

## Decisión de plan (no es error): Plan A vs B (Steedwar)
- Plan A (modelo extremo a extremo con `TextVectorization(standardize=None)`) quedó vigente: diferencia máxima 0.0 en proceso nuevo (notebook sección 5, celda 62). El Plan B no fue necesario.

## Episodio 2 — 2026-11-15 / 2026-11-17 — Tarea 6 (src/agente.py - Juan Diego)
- **Qué propuse:** Utilizar el identificador de modelo `gemini-2.0-flash` y posteriormente `gemini-2.0-flash-lite` para el agente LangGraph.
- **Por qué estaba mal:** Ambos modelos fueron dados de baja en el API de Google AI Studio, arrojando excepción `ClientError: 404 NOT_FOUND` indicando migrar a la familia `gemini-3.x`. Adicionalmente, el nuevo endpoint retorna bloques de contenido estructurados en listas de diccionarios en lugar de un string plano, lo que provocó un `AttributeError: 'list' object has no attribute 'lower'`.
- **Corrección:** Se configuró el modelo vigente `gemini-3.5-flash-lite` y se implementó un extractor normalizador en `src/agente.py:responder()` que concatena los bloques `text` de la lista de contenido devuelta por el modelo. Verificado con `src/prueba_agente.py`.
- **Captura:** en `assets/vibe_coding/`.

## Episodio 3 — 2026-11-17 — Tarea 10 (src/agente.py - Juan Diego)
- **Qué propuse:** Regla 5 del system prompt que solo indicaba *"indica prioridad, accion y pauta de forma concisa"*.
- **Por qué estaba mal:** El agente ejecutaba correctamente la trayectoria `@tool #1` -> `@tool #2`, pero en la respuesta final omitía redactar explícitamente el sentimiento predicho ("negativo"), fallando el assert de coherencia de datos del Requisito 9.
- **Corrección:** Se ajustó la Regla 5 del system prompt para exigir explícitamente: *"menciona el sentimiento y confianza detectados, prioridad, accion y pauta"*.
- **Captura:** en `assets/vibe_coding/`.
