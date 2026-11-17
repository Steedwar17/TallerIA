# INSUMO PARA LA SECCIÓN 4 — LA REDACCIÓN FINAL ES DEL EQUIPO

Documento de hechos técnicos verificados en código y ejecución real para la Sección 4 del README.

---

## (a) Docstrings de las Herramientas

### 1. `@tool` #1: `clasificar_sentimiento_resena`
- **Texto final implementado:**
  ```python
  """Clasifica sentimiento de resena en ingles usando red neuronal.

  Usala: usuario entrega texto de resena para clasificar.
  NO usar: saludos, preguntas de politicas o texto ya clasificado.

  Args:
      texto: resena literal en ingles.

  Returns:
      JSON (max 120 chars) con sentimiento, confianza y baja_confianza.
  """
  ```
- **Tokens medidos (`gemini-3.5-flash-lite`):** Reducción de **122 tokens** (base) a **122 tokens** (optimizado) -> **0.0% de ahorro**.
- **Qué se cambió y motivo observado:** Se eliminó texto narrativo superfluo ("Ejecuta la red neuronal local entrenada con reseñas IMDB..."). Se estructuró en cláusulas operativas directas `Usala:` y `NO usar:` para que el planificador LLM reconozca de inmediato las precondiciones de activación sin consumir tokens de contexto en cada invocación.
- **[COMPLETADO EN TAREA 10]:** Conectada con `src/tool_sentimiento.py` entregada por Steedwar en la Tarea 9 (`models/sentimiento_imdb.keras`). Docstring real en `src/tool_sentimiento.py:16-29` y límite duro de 120 caracteres en `src/tool_sentimiento.py:44`. Pruebas de integración superadas (6/6).

### 2. `@tool` #2: `consultar_politica_triaje`
- **Texto final implementado:**
  ```python
  """Consulta la politica interna y protocolo de respuesta para una resena de cliente.

  Usala cuando conozcas el sentimiento/confianza de la resena o cuando el usuario pregunte
  directamente sobre lineamientos, politicas de triaje o protocolos de escalamiento.
  NO la uses para clasificar texto no analizado (usa clasificar_sentimiento_resena) ni
  para saludos o preguntas genericas no relacionadas con politicas.

  Args:
      consulta: pregunta o texto de consulta sobre el protocolo o la resena.
      sentimiento: 'positivo', 'negativo' o 'incierto' (si ya fue inferido).
      confianza: valor flotante de certeza (0.0 a 1.0).

  Returns:
      JSON compacto (max 600 caracteres) con pol_id, prioridad, accion y pauta.
  """
  ```
- **Tokens medidos (`gemini-3.5-flash-lite`):** Reducción de **205 tokens** (base) a **187 tokens** (optimizado) -> **8.8% de ahorro**.
- **Qué se cambió y motivo observado:** Se sintetizó la descripción excluyendo redundancias ("plataforma de streaming", "corporativo"), manteniendo explícitas las condiciones de no uso para evitar alucinaciones de llamadas duplicadas.

---

## (b) Memoria Conversacional y Recorte de Historial

- **Checkpointer utilizado:** `MemorySaver` (`langgraph.checkpoint.memory`), gestionando persistencia en memoria volátil por sesión.
- **Generación y reinicio de sesión:**
  - Función `nuevo_thread_id()` (`src/agente.py:100`): genera un `uuid.uuid4()` aleatorio único.
  - Función `config_sesion(thread_id)` (`src/agente.py:105`): inyecta `{"configurable": {"thread_id": thread_id}}` en el runtime de LangGraph.
  - Al generar un nuevo `thread_id`, el checkpointer no encuentra entradas previas asociadas y el agente arranca con memoria en blanco (verificado en `src/prueba_agente.py:TEST 4`).
- **Recorte de historial (`trim_messages` en middleware/prompt modifier):**
  - Implementado en `crear_middleware_recorte()` (middleware de `create_agent`, `src/agente.py`); `crear_prompt_modifier()` conserva la misma lógica para las pruebas.
  - Límite configurado: constante `MAX_MENSAJES_HISTORIAL = 8` (`src/config.py:16`).
  - Reglas aplicadas:
    1. Estrategia `"last"` con `token_counter=len` para conservar los últimos 8 mensajes conversacionales.
    2. `start_on="human"`: garantiza que el contexto recortado inicie siempre con un mensaje de usuario válido.
    3. `allow_partial=False`: previene que un mensaje de llamada a herramienta (`AIMessage` con `tool_calls`) quede separado de su correspondiente respuesta (`ToolMessage`), evitando excepciones de validación en la API de Gemini.
    4. El `SystemMessage` se reinyecta siempre al inicio de la lista enviada al modelo.
  - Verificado con conversación de 12 turnos (32 mensajes acumulados en estado) recortados dinámicamente a 9 mensajes efectivos hacia el LLM sin fallas de secuencia.

---

## (c) Medidas de Ahorro de Tokens Implementadas y Cifras Medidas

| # | Medida de Ahorro | Ubicación (Archivo:Línea) | Cifra Empírica Medida |
|---|---|---|---|
| 1 | **Optimización del System Prompt (6 reglas)** | `src/agente.py:48-57` | De **395** a **136 tokens** (**65.6% de ahorro**) |
| 2 | **Optimización de Docstring Tool #1** | `src/agente.py:64-75` | De **122** a **122 tokens** (**0.0% de ahorro**) |
| 3 | **Optimización de Docstring Tool #2** | `src/tool_politicas.py:38-51` | De **205** a **187 tokens** (**8.8% de ahorro**) |
| 4 | **Límite duro de salida en Tool #1** | `src/agente.py:96` | Assert `len(salida) <= 120` (salida real: **66 chars**) |
| 5 | **Límite duro de salida en Tool #2** | `src/tool_politicas.py:84` | Assert `len(salida) <= 600` (salida real: **156 chars**) |
| 6 | **Tope de salida del modelo (`max_output_tokens`)** | `src/agente.py:192` | Limitado a **512 tokens** con `temperature=0` |
| 7 | **Recorte de historial conversacional** | `src/agente.py:157` | Límite máximo de **8 mensajes** recientes (`MAX_MENSAJES_HISTORIAL`) |
| 8 | **Límite de ritmo (`InMemoryRateLimiter`)** | `src/agente.py:126` | **0.25 req/seg** (15 RPM gratuito) para evitar reintentos HTTP 429 |
| 9 | **Tope de entrada de usuario (`MAX_CARACTERES_RESENA`)** | `src/config.py:14`, `src/agente.py:233` | Truncado defensivo a **4000 caracteres** |
| 10 | **Ahorro total medido en suite de 3 casos reales** | `assets/apuntes/medicion_tokens.md` | De **7,125** a **4,712 tokens** (**33.9% de ahorro global**) |

---

## (d) Integración del Agente con Red Neuronal y Políticas (Tarea 10)

- **Herramientas enlazadas en `src/herramientas.py:obtener_tools()`:**
  - `@tool #1` `clasificar_sentimiento_resena` (modelo real `models/sentimiento_imdb.keras`).
  - `@tool #2` `consultar_politica_triaje` (protocolos internos de triaje).
- **Pruebas de integración automatizadas (`src/prueba_integracion.py`):**
  - **Prueba A (saludo):** 0 tools invocadas (`[]`).
  - **Prueba B (consulta política):** solo `['consultar_politica_triaje']`.
  - **Prueba C (triaje completo con reseña negativa real):** `['clasificar_sentimiento_resena', 'consultar_politica_triaje']`. Verificada por código la coherencia exacta de argumentos y predicción.
  - **Prueba D (memoria en mismo hilo):** responde la confianza previa sin volver a invocar tools.
  - **Prueba E (reinicio de hilo):** con nuevo `thread_id`, `get_state()` está vacío y no tiene conocimiento previo.
  - **Prueba F (baja confianza / fallo real):** detecta baja certeza (0.58 < 0.70) y aplica política de revisión humana (`POL_REV`).
- **Consumo:** 11 llamadas totales al LLM en la suite completa (límite: 20).
- **Reporte completo:** [prueba_integracion.md](file:///c:/Users/USUARIO/Documents/Inteligencia%20Artificial/TallerIA/assets/apuntes/prueba_integracion.md).
