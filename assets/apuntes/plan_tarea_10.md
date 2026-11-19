# Plan de Implementación — Tarea 10: Integración de las Dos Herramientas en el Agente

## Objetivo
Integrar las dos herramientas reales en el agente ReAct: la red neuronal local de Steedwar (`@tool #1: clasificar_sentimiento_resena`) y las políticas corporativas (`@tool #2: consultar_politica_triaje`), eliminando la herramienta provisional y comprobando de forma rigurosa el enrutamiento selectivo, la coherencia de datos, la memoria persistente en sesión y el reinicio de contexto mediante `thread_id`.

---

## Contexto y Alcance (según `spec.md` §7.2 y `prompts.md` § Tarea 10)
* **Responsable:** Juan Diego Castillo Jiménez.
* **Insumo necesario:** Módulo de red local terminado por Steedwar (`models/sentimiento_imdb.keras`, `src/inferencia.py`, `src/tool_sentimiento.py`) y caso de fallo real en `assets/apuntes/fallo_real.md`.
* **Archivos intervenidos:**
  - `src/herramientas.py`: Única fuente de herramientas (`obtener_tools()`).
  - `src/agente.py`: Ajustes al system prompt y normalización de turnos si una prueba lo exige.
  - `src/tool_politicas.py`: Refinamiento de docstring contractual.
  - `src/tool_provisional.py`: Eliminación definitiva.
  - `src/prueba_integracion.py`: Suite automatizada de pruebas A a F.
  - `assets/apuntes/prueba_integracion.md`: Evidencia de ejecución real.
  - `assets/apuntes/apuntes_agente.md`: Documentación de docstrings reales.
  - `assets/apuntes/bitacora_vibe_coding.md`: Registro de incidentes.

---

## User Review Required
> [!IMPORTANT]
> **1. Prohibición de lógica condicional para enrutamiento:** El enrutamiento de herramientas debe ser 100% autónomo por parte del planificador del LLM (Gemini 3.5 Flash Lite) guiado por los docstrings y el system prompt. Queda prohibido añadir condicionales `if/else` en Python que fuercen la llamada a una herramienta.
> **2. Validación de trayectoria por `tool_calls`:** Ninguna prueba puede validar la trayectoria buscando palabras clave en el texto final de la respuesta; debe inspeccionarse directamente el campo `AIMessage.tool_calls`.
> **3. Cuidado de cuota gratuita (15 RPM):** Limitar la suite a un máximo de 20 llamadas al modelo, con pausas automáticas entre pruebas para evitar errores HTTP 429.

---

## Proposed Changes

```
TallerIA/
├── src/
│   ├── herramientas.py          [MODIFY] Importar @tool #1 real y @tool #2
│   ├── tool_provisional.py      [DELETE] Eliminar archivo provisional
│   ├── tool_politicas.py        [MODIFY] Ajustar docstring contractual y límite <= 600 chars
│   ├── agente.py                [MODIFY] Refinar system prompt y normalizador
│   └── prueba_integracion.py    [NEW] Suite de pruebas automatizadas A a F
└── assets/apuntes/
    ├── prueba_integracion.md    [NEW] Reporte de salidas reales y tokens medidos
    ├── apuntes_agente.md        [MODIFY] Actualizar bloque de docstrings reales
    └── bitacora_vibe_coding.md  [MODIFY] Registrar episodios de bitácora
```

---

### Componente 1: Fuente Única de Herramientas (`src/herramientas.py`)
* Reemplazar la importación de `src.tool_provisional` por `src.tool_sentimiento.clasificar_sentimiento_resena`.
* Garantizar que `obtener_tools()` retorne exactamente la lista:
  ```python
  from src.tool_sentimiento import clasificar_sentimiento_resena
  from src.tool_politicas import consultar_politica_triaje

  def obtener_tools():
      """Retorna las 2 herramientas reales del agente de triaje."""
      return [clasificar_sentimiento_resena, consultar_politica_triaje]
  ```
* Eliminar `src/tool_provisional.py` y verificar con búsqueda global que no queden referencias a "provisional" en el código fuente.

---

### Componente 2: Docstrings Contractuales y Prompt del Sistema

#### 1. `@tool #1: clasificar_sentimiento_resena` ([src/tool_sentimiento.py](src/tool_sentimiento.py))
* Docstring compacto con directriz de activación positiva (`Usala cuando...`) y exclusión negativa taxativa:
  ```python
  @tool
  def clasificar_sentimiento_resena(texto: str) -> str:
      """Clasifica sentimiento de resena en ingles usando red neuronal.

      Usala: usuario entrega texto de resena para clasificar.
      NO usar: saludos, preguntas de politicas o texto ya clasificado.
      """
  ```

#### 2. `@tool #2: consultar_politica_triaje` ([src/tool_politicas.py](src/tool_politicas.py))
* Docstring adaptado para recibir `sentimiento` y `confianza` opcionales, delimitando la frontera operativa para no colisionar con la Tool #1.
* Límite duro de salida: `assert len(salida) <= 600`.

#### 3. System Prompt Compacto ([src/agente.py](src/agente.py))
* Mantener las 6 directrices densas:
  1. Rol de triaje de streaming.
  2. Prohibición de re-clasificar texto ya analizado en el hilo.
  3. Activación exclusiva de Tool #1 ante texto sin clasificar.
  4. Activación de Tool #2 pasando los argumentos de la inferencia previa.
  5. Estructura obligatoria de respuesta (sentimiento, confianza, política, prioridad, acción y pauta).
  6. Mandato de no contradecir al modelo local en casos de baja confianza (derivar a `POL_REV`).

---

### Componente 3: Suite de Pruebas de Integración (`src/prueba_integracion.py`)

Diseñar y ejecutar 6 pruebas funcionales con validaciones programáticas por `assert`:

1. **Prueba A (Sin herramientas):**
   * Entrada: `"Hola, ¿en qué me puedes ayudar?"`
   * Validación: `herramientas_invocadas == []` (0 herramientas).
2. **Prueba B (Solo `@tool #2`):**
   * Entrada: `"¿Qué política aplica si un cliente amenaza con demandar?"`
   * Validación: `herramientas_invocadas == ["consultar_politica_triaje"]`.
3. **Prueba C (Caso completo con ambas herramientas):**
   * Entrada: Reseña negativa real de test IMDB pegada literal.
   * Validación: `herramientas_invocadas == ["clasificar_sentimiento_resena", "consultar_politica_triaje"]` en ese orden exacto.
4. **Coherencia de datos en Prueba C:**
   * Validación por código: `sentimiento` y `confianza` enviados en los argumentos del `tool_call` de la Tool #2 deben coincidir con los valores devueltos en el `ToolMessage` de la Tool #1, y la respuesta final debe mencionar el mismo sentimiento.
5. **Prueba D (Memoria en el mismo `thread_id`):**
   * Entrada en el mismo hilo de C: `"¿cuál fue la confianza de esa reseña?"`
   * Validación: La respuesta contiene la cifra devuelta por la red (ej. 1.0) y responde de memoria (`herramientas_invocadas == []`), sin volver a invocar la herramienta.
6. **Prueba E (Reinicio con nuevo `thread_id`):**
   * Entrada con nuevo `thread_id`: `"¿cuál fue la confianza de esa reseña?"`
   * Validación: El agente desconoce la reseña previa y `agente.get_state(config)` no tiene mensajes previos.
7. **Prueba F (Caso de baja confianza / Fallo real):**
   * Entrada: Texto del Caso B de `assets/apuntes/fallo_real.md` ("Not bad, but not great either...").
   * Validación: El agente reporta la baja certeza (0.5835 < 0.70) y asigna la política `POL_REV` para moderación humana, sin corregir al modelo por su cuenta.

---

## Plan de Verificación

### Pruebas Automatizadas
1. **Verificación previa de inferencia:**
   ```powershell
   python -m src.prueba_inferencia
   ```
2. **Ejecución de la suite de integración de la Tarea 10:**
   ```powershell
   python src/prueba_integracion.py
   ```
3. **Búsqueda de residuos de tool provisional:**
   ```powershell
   git grep -i "provisional" src/
   ```

### Criterios de Éxito
* [x] Inferencia de red local operativa antes de conectar el agente.
* [x] `obtener_tools()` enlaza exclusivamente las 2 herramientas reales.
* [x] `src/tool_provisional.py` eliminado y sin referencias residuales.
* [x] Pruebas A a F aprobadas al 100% con asserts programáticos.
* [x] Coherencia estricta de datos entre Tool #1 y Tool #2 comprobada.
* [x] Memoria persistente en el mismo `thread_id` y borrado total con nuevo `thread_id`.
* [x] Salida real documentada en `assets/apuntes/prueba_integracion.md`.
