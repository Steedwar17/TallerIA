# Plan de Implementación — Tarea 12: Aplicación Streamlit, Parte del Agente

## Objetivo
Implementar el módulo interactivo del agente en Streamlit (`app_agent.py`), que proporcione una interfaz de chat para triaje de reseñas con memoria de corto plazo gestionada por `thread_id` (vía `MemorySaver`), visualización de herramientas invocadas y sus detalles crudos (`tool_calls` y `ToolMessage`), consumo de tokens por turno, botón de reinicio de conversación, puente con el panel del modelo y manejo robusto de excepciones (sin clave, cuota 429), acompañado de pruebas unitarias automatizadas con `streamlit.testing.v1.AppTest` en `src/prueba_app_agent.py`.

---

## User Review Required
> [!IMPORTANT]
> **1. Rama de trabajo:** El prompt de la Tarea 12 menciona `juandiego/agente`, pero en la tarea anterior solicitaste explícitamente unificar todo en `main` y borrar `juandiego/agente`. El plan propone trabajar directamente en `main` (o crear la rama si prefieres aislamiento).
> **2. Capturas reales de pantalla:** Los requisitos 14 y 16 exigen ejecutar la app en el navegador y capturar `assets/app_panel_agente.png` y `assets/app_reinicio_thread.png`. El plan contempla arrancar Streamlit en headless, interactuar con el agente en turnos reales y documentar los pasos.

---

## Open Questions
1. **Rama de Git:** ¿Trabajamos directamente sobre `main` como en la tarea 10, o deseas que creemos la rama `juandiego/agente`? (Recomendado: seguir en `main` dado que ya se unificó).
2. **Navegador para capturas:** ¿Deseas que usemos el subagente de navegador (`browser`) para interactuar con la app en `localhost:8501` y tomar las capturas requeridas, o prefieres tomarlas manualmente desde tu navegador?

---

## Proposed Changes

```
c:\Users\USUARIO\Documents\Inteligencia Artificial\TallerIA\
├── app_agent.py                 [NEW] Interfaz del agente en Streamlit
├── src/
│   ├── agente.py                [MODIFY] Añadir alias 'herramientas' en el retorno de responder()
│   └── prueba_app_agent.py      [NEW] Suite de pruebas con streamlit.testing.v1.AppTest
└── assets/
    ├── app_panel_agente.png     [NEW] Captura del panel del agente con 2 tools
    └── app_reinicio_thread.png  [NEW] Captura del reinicio de thread_id
```

---

### Componente 1: Núcleo del Agente (`src/agente.py`)

#### [MODIFY] `src/agente.py`
- Añadir en el diccionario retornado por `responder()` la clave `"herramientas": herramientas_invocadas(msgs_turno)` además de `"tools_invocadas"`, asegurando compatibilidad estricta con el contrato de la Tarea 12.

---

### Componente 2: Módulo Streamlit del Agente (`app_agent.py`)

#### [NEW] `app_agent.py`
Implementará:
1. **Verificación defensiva de entorno:** Comprueba existencia de `GOOGLE_API_KEY`. Si falta, muestra `st.error` con pasos para configurar `.env` y llama `st.stop()`.
2. **Caché de agente y checkpointer:**
   ```python
   @st.cache_resource(show_spinner="Iniciando el agente...")
   def obtener_agente():
       # El agente y su checkpointer se crean una sola vez por proceso.
       # Se comparte de forma segura entre sesiones porque cada una tiene su propio thread_id.
       return construir_agente(obtener_tools())
   ```
3. **Inicialización de estado (`st.session_state` con prefijo `ag_`):**
   - `ag_thread_id`: generado con `nuevo_thread_id()`.
   - `ag_historial`: lista de turnos `[{"role": "user"|"assistant", "contenido": ..., "herramientas": ..., "tokens": ..., "detalle_tools": ...}]` exclusivamente para pintar la UI.
4. **Función principal `render_panel_agente()`:**
   - Barra superior con visualización del `thread_id` abreviado (ej. primeros 8 caracteres) y botón *"🔄 Reiniciar conversación (nuevo thread_id)"* que reinicia el id, limpia `ag_historial` y ejecuta `st.rerun()`.
   - Puente con el panel del modelo: si existe `st.session_state.get("nn_texto")`, muestra botón *"📋 Enviar al agente la reseña del panel del modelo"* para inyectarla de inmediato con la instrucción de triaje.
   - Renderizado del historial con `st.chat_message("user")` y `st.chat_message("assistant")`.
   - Para cada mensaje del asistente:
     - Texto de la respuesta.
     - Indicador de herramientas invocadas: badge o markdown visible con `@tool #1 clasificar_sentimiento_resena`, `@tool #2 consultar_politica_triaje` o `"Sin herramientas"`.
     - `st.expander("Detalle de herramientas")` mostrando argumentos enviados y salida cruda (`ToolMessage.content`).
     - `st.caption` con métricas de tokens (`input`, `output`, `total`).
   - Entrada con `st.chat_input("Escribe una consulta o reseña...")`.
   - Validación de longitud: si supera `MAX_CARACTERES_RESENA` (4.000 chars), se recorta y se muestra `st.warning`.
   - Manejo de excepciones: captura errores de cuota (HTTP 429 / ResourceExhausted) mostrando `st.warning` amigable sin traceback ni fuga de credenciales.
5. **Punto de entrada independiente:**
   ```python
   if __name__ == "__main__":
       st.set_page_config(page_title="Triaje de Reseñas — Agente", layout="wide")
       render_panel_agente()
   ```

---

### Componente 3: Suite de Pruebas Automatizadas (`src/prueba_app_agent.py`)

#### [NEW] `src/prueba_app_agent.py`
Pruebas usando `streamlit.testing.v1.AppTest`:
- **Test 1: Render inicial y estado por defecto:**
  - Verifica que `ag_thread_id` exista y tenga formato UUID.
  - Verifica que `ag_historial` empiece vacío.
  - Verifica que no haya excepciones (`at.exception` vacío).
- **Test 2: Aislamiento y seguridad de caché:**
  - Verifica que dos llamadas a `obtener_agente()` devuelvan la misma instancia en memoria, pero con distintos `thread_id` operen independientemente.
- **Test 3: Envío de mensaje con doble de prueba (Mock sin consumir cuota):**
  - Simula `st.chat_input` enviando una consulta.
  - Con un agente mock que simula invocación de `@tool #1` y `@tool #2`, valida que el historial muestre el mensaje, el indicador visual de tools y el expander con los argumentos.
- **Test 4: Reinicio de conversación:**
  - Pulsa el botón de reinicio.
  - Valida que `ag_thread_id` cambie a un nuevo valor y `ag_historial` se limpie completamente.
- **Test 5: Puente con panel de modelo:**
  - Inyecta `st.session_state["nn_texto"] = "Great movie"`.
  - Verifica que el botón de puente aparezca y al accionarlo despache el turno al agente.

---

## Verification Plan

### Automated Tests
1. Ejecutar la suite de pruebas Streamlit:
   ```powershell
   & "C:\Users\USUARIO\anaconda3\python.exe" src/prueba_app_agent.py
   ```
2. Ejecutar con pytest si está disponible:
   ```powershell
   & "C:\Users\USUARIO\anaconda3\python.exe" -m unittest src/prueba_app_agent.py
   ```

### Manual Verification & Screenshots
1. Iniciar Streamlit en background:
   ```powershell
   streamlit run app_agent.py --server.headless true
   ```
2. Ejecutar los 4 flujos requeridos en el navegador:
   - Turno 1: Saludo inicial (*"Hola, ¿en qué me puedes ayudar?"*) -> Indicador *"Sin herramientas"*.
   - Turno 2: Reseña completa para triaje -> Indicador con ambas herramientas y expander desplegado.
   - Turno 3: Pregunta con memoria (*"¿cuál fue la confianza?"*) -> Responde recordando el valor previo.
   - Turno 4: Reinicio de sesión con botón -> `thread_id` cambia; repetición de la pregunta no recuerda el dato previo.
3. Guardar las capturas en `assets/app_panel_agente.png` y `assets/app_reinicio_thread.png`.
