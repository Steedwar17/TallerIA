# Medición de Tokens — Antes vs. Después (Tarea 8)

- **Fecha:** 2026-11-17
- **Modelo evaluado:** `gemini-3.5-flash-lite`
- **Checkpointer:** `MemorySaver` con `thread_id`
- **Ahorro global medido:** **40.1%** de tokens totales en los 3 casos

## 1. Componentes Estáticos (Tokens por Invocación)

| Componente | Tokens Antes | Tokens Después | Reducción | % Ahorro |
|---|---|---|---|---|
| **System Prompt (6 reglas)** | 395 | 132 | 263 | 66.6% |
| **Docstring Tool #1 (`clasificar_sentimiento_resena`)** | 118 | 80 | 38 | 32.2% |
| **Docstring Tool #2 (`consultar_politica_triaje`)** | 199 | 181 | 18 | 9.0% |

## 2. Invocaciones Dinámicas (3 Casos Reales)

| Caso | Métricas Antes (In / Out / Tot / LLM Calls) | Métricas Después (In / Out / Tot / LLM Calls) | Ahorro Total (%) | Tools Invocadas |
|---|---|---|---|---|
| **Caso 1: Saludo** | 828 / 106 / 934 / 1 | 513 / 48 / 561 / 1 | **39.9%** | `[]` |
| **Caso 2: Pregunta de política** | 2009 / 147 / 2156 / 2 | 1255 / 87 / 1342 / 2 | **37.8%** | `['consultar_politica_triaje']` |
| **Caso 3: Reseña completa** | 3833 / 193 / 4026 / 3 | 2231 / 129 / 2360 / 3 | **41.4%** | `['clasificar_sentimiento_resena', 'consultar_politica_triaje']` |
| **TOTALES** | **6670 / 446 / 7116 / 6** | **3999 / 264 / 4263 / 6** | **40.1%** | — |

## 3. Conclusiones Empíricas
1. **Reducción del System Prompt:** Al eliminar prosa redundante manteniendo intactas las 6 reglas, el tamaño del prompt de sistema se redujo sustancialmente.
2. **Docstrings de Herramientas:** La sintaxis compacta con viñetas 'Usala:' y 'NO usar:' preserva la precisión del enrutamiento agéntico y ahorra tokens en cada turno.
3. **Recorte de Historial:** Garantiza que en turnos avanzados el número de mensajes inyectados no crezca indefinidamente, iniciando siempre en un `HumanMessage` y sin separar pares `AIMessage` - `ToolMessage`.
4. **Límite de Ritmo:** `InMemoryRateLimiter(requests_per_second=0.25)` previene errores de cuota HTTP 429 en el plan gratuito de 15 RPM.