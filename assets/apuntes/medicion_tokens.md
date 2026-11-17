# Medición de Tokens — Antes vs. Después (Tarea 8)

- **Fecha:** 2026-11-17
- **Modelo evaluado:** `gemini-3.5-flash-lite`
- **Checkpointer:** `MemorySaver` con `thread_id`
- **Ahorro global medido:** **33.9%** de tokens totales en los 3 casos

## 1. Componentes Estáticos (Tokens por Invocación)

| Componente | Tokens Antes | Tokens Después | Reducción | % Ahorro |
|---|---|---|---|---|
| **System Prompt (6 reglas)** | 395 | 136 | 259 | 65.6% |
| **Docstring Tool #1 (`clasificar_sentimiento_resena`)** | 122 | 122 | 0 | 0.0% |
| **Docstring Tool #2 (`consultar_politica_triaje`)** | 205 | 187 | 18 | 8.8% |

## 2. Invocaciones Dinámicas (3 Casos Reales)

| Caso | Métricas Antes (In / Out / Tot / LLM Calls) | Métricas Después (In / Out / Tot / LLM Calls) | Ahorro Total (%) | Tools Invocadas |
|---|---|---|---|---|
| **Caso 1: Saludo** | 838 / 95 / 933 / 1 | 565 / 69 / 634 / 1 | **32.0%** | `[]` |
| **Caso 2: Pregunta de política** | 2007 / 151 / 2158 / 2 | 1401 / 98 / 1499 / 2 | **30.5%** | `['consultar_politica_triaje']` |
| **Caso 3: Reseña completa** | 3835 / 199 / 4034 / 3 | 2420 / 159 / 2579 / 3 | **36.1%** | `['clasificar_sentimiento_resena', 'consultar_politica_triaje']` |
| **TOTALES** | **6680 / 445 / 7125 / 6** | **4386 / 326 / 4712 / 6** | **33.9%** | — |

## 3. Conclusiones Empíricas
1. **Reducción del System Prompt:** Al eliminar prosa redundante manteniendo intactas las 6 reglas, el tamaño del prompt de sistema se redujo sustancialmente.
2. **Docstrings de Herramientas:** La sintaxis compacta con viñetas 'Usala:' y 'NO usar:' preserva la precisión del enrutamiento agéntico y ahorra tokens en cada turno.
3. **Recorte de Historial:** Garantiza que en turnos avanzados el número de mensajes inyectados no crezca indefinidamente, iniciando siempre en un `HumanMessage` y sin separar pares `AIMessage` - `ToolMessage`.
4. **Límite de Ritmo:** `InMemoryRateLimiter(requests_per_second=0.25)` previene errores de cuota HTTP 429 en el plan gratuito de 15 RPM.