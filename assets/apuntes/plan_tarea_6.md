# Plan de Implementación — Tarea 6: Configuración del Agente

## Objetivo
Crear en `src/` el agente de triaje inteligente usando `create_react_agent` de LangGraph con `gemini-2.0-flash` (Google AI Studio), `MemorySaver` como checkpointer y gestión de sesiones por `thread_id`. El código debe ser modular (recibe la lista de herramientas como parámetro) y funcionar con la `@tool #2` ya implementada y una `@tool #1` provisional simulada que el equipo acordó.

## Decisiones Acordadas

| Decisión | Valor |
|---|---|
| Estilo de agente | `create_react_agent` (langgraph.prebuilt) |
| Modelo | `gemini-2.0-flash` (plan gratuito Google AI Studio) |
| Checkpointer | `MemorySaver` (langgraph.checkpoint.memory) |
| Formato @tool #1 provisional | `SENTIMIENTO: X \| CONFIANZA: Y` donde X ∈ {positivo, negativo, incierto} |
| Umbral incierto | confianza < 0.60 → `incierto` |

## Cambios Propuestos

### Paso 0 — Instalar dependencias declaradas
Ninguna librería nueva: solo instalar las ya declaradas en `requirements.txt` que aún no están en el entorno:
```
langgraph, langchain, langchain-core, langchain-google-genai, google-genai
```

---

### Componente: `src/`

#### [NEW] `src/agent.py`

El módulo central del agente. Contiene:

1. **`@tool` provisional `inferir_sentimiento_resena`** — claramente marcada como simulada, fácil de reemplazar en Tarea 10. Recibe un texto de reseña y devuelve formato acordado:
   ```
   SENTIMIENTO: negativo | CONFIANZA: 0.87
   ```
   El umbral `< 0.60` produce `incierto` en lugar de `positivo`/`negativo`.

2. **`crear_agente(tools)`** — función modular que recibe la lista de herramientas, crea el `ChatGoogleGenerativeAI` con `gemini-2.0-flash`, instancia el `MemorySaver`, y retorna el `create_react_agent` compilado. Esta firma permite:
   - Fase actual (Tarea 6): `crear_agente([tool_1_provisional, consultar_politicas_triaje])`
   - Tarea 10: `crear_agente([tool_1_real_de_steedwar, consultar_politicas_triaje])`

3. **`nuevo_thread_id()`** — genera un UUID4 único para cada sesión nueva.

4. **`obtener_config(thread_id)`** — retorna el dict `{"configurable": {"thread_id": thread_id}}` que LangGraph usa para aislar la memoria por sesión.

5. **`invocar_agente(agente, mensaje, thread_id)`** — función de invocación que retorna la respuesta final y los `tool_calls` detectados (para transparencia en la app y evaluación).

```python
# API verificada (LangGraph 1.2.x, langchain-google-genai 4.5.x):
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
```

> [!IMPORTANT]
> La `GOOGLE_API_KEY` se lee exclusivamente desde `.env` vía `python-dotenv`. **Nunca** se imprime, escribe en archivos versionados ni se pide en el chat.

---

#### [NEW] `src/test_agent.py`

Script de prueba reproducible que ejecuta las **4 demos obligatorias** con salidas reales:

| # | Demostración | Qué verifica |
|---|---|---|
| 1 | Saludo al agente | No invoca ninguna herramienta |
| 2 | Reseña negativa con alta certeza | Invoca `@tool #1` y `@tool #2` |
| 3 | Dos turnos con mismo `thread_id` | Recuerda información del turno anterior |
| 4 | Nuevo `thread_id` | La memoria no persiste (responde que no recuerda) |

---

## Archivos que se tocan

| Archivo | Acción |
|---|---|
| `src/agent.py` | **[NEW]** Módulo del agente |
| `src/test_agent.py` | **[NEW]** Script de pruebas con 4 demos |
| `src/__init__.py` | Sin cambios |
| `src/tools.py` | Sin cambios |
| `src/policies.py` | Sin cambios |
| `requirements.txt` | Sin cambios (paquetes ya declarados) |
| `.env` | Juan Diego configura manualmente su `GOOGLE_API_KEY` antes de ejecutar |

## Configuración Previa Requerida de Juan Diego

> [!IMPORTANT]
> Antes de ejecutar `src/test_agent.py`, Juan Diego debe asegurarse de que su archivo `.env` local (en la raíz del proyecto) contenga:
> ```
> GOOGLE_API_KEY=su_clave_real_de_google_ai_studio
> ```
> Este archivo está en `.gitignore` y **nunca** se sube al repositorio.

## Plan de Verificación

### Dependencias
```powershell
& "C:\Users\USUARIO\anaconda3\python.exe" -m pip install langgraph langchain langchain-google-genai
```

### Prueba del agente
```powershell
cd "c:\Users\USUARIO\Documents\Inteligencia Artificial\TallerIA"
& "C:\Users\USUARIO\anaconda3\python.exe" src/test_agent.py
```

### Resultado esperado
- Demo 1: Respuesta sin `tool_calls`
- Demo 2: `tool_calls` incluyen `inferir_sentimiento_resena` y `consultar_politicas_triaje`
- Demo 3: El agente recuerda el nombre/dato del turno 1 en el turno 2
- Demo 4: El agente responde que no tiene contexto previo
