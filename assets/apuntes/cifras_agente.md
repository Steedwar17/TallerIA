# Cifras Verificadas del Agente (Insumo para Secciones 2, 4 y 5 del README)

Documento autogenerado a partir de las salidas reales guardadas en `notebooks/evaluacion_agente_u5.ipynb` y `assets/apuntes/medicion_tokens.md`.

## 1. Casos de Evaluación del Agente (notebooks/evaluacion_agente_u5.ipynb)

| Caso | Celda Notebook | Herramientas Esperadas | Herramientas Invocadas | Trayectoria | Resultado | Tokens Medidos |
|---|---|---|---|:---:|:---:|---:|
| **Caso 1** | Celda 3 (In[2]) | `[]` | `[]` | **CUMPLE** | **CUMPLE** | 609 |
| **Caso 2** | Celda 5 (In[3]) | `consultar_politica_triaje` | `consultar_politica_triaje` | **CUMPLE** | **CUMPLE** | 1,365 |
| **Caso 3** | Celda 7 (In[4]) | `clasificar_sentimiento_resena, consultar_politica_triaje` | `clasificar_sentimiento_resena, consultar_politica_triaje` | **CUMPLE** | **CUMPLE** | 2,230 |
| **Caso 4** | Celda 9 (In[5]) | `clasificar_sentimiento_resena, consultar_politica_triaje` | `clasificar_sentimiento_resena, consultar_politica_triaje` | **CUMPLE** | **CUMPLE** | 2,130 |

- **Efectividad en Trayectoria:** 4 de 4 casos (100.0%)
- **Efectividad en Resultado:** 4 de 4 casos (100.0%)
- **Inspección de trayectoria:** Obtenida exclusivamente de `tool_calls` (`AIMessage.tool_calls`) en `src/agente.py` vía `herramientas_invocadas`, consumida en `src/evaluacion.py`.
- **Total llamadas al LLM en evaluación:** 9 llamadas (1 en Caso 1, 2 en Caso 2, 3 en Caso 3, 3 en Caso 4).

---

## 2. Métricas de Ahorro de Tokens (assets/apuntes/medicion_tokens.md)

### Ahorro Estático (por cada invocación)
- **System Prompt (6 reglas):** Reducción de 395 a 136 tokens (**65.6% de ahorro**).
- **Docstring Tool #1 (`clasificar_sentimiento_resena`):** Reducción de 122 a 122 tokens (**0.0% de ahorro**).
- **Docstring Tool #2 (`consultar_politica_triaje`):** Reducción de 205 a 187 tokens (**8.8% de ahorro**).

### Ahorro Dinámico en Ejecución (Suite de 3 turnos reales)
- **Tokens totales antes de optimización:** 7,125 tokens (6,680 entrada + 445 salida) en 6 llamadas LLM.
- **Tokens totales después de optimización:** 4,712 tokens (4,386 entrada + 326 salida) en 6 llamadas LLM.
- **Ahorro global neto:** **33.9% de reducción** de consumo de cuota.

### Medidas de Ahorro Implementadas con Ubicación en Código
1. `src/agente.py:48-57`: System prompt compacto con 6 directrices concisas (136 tokens).
2. `src/tool_sentimiento.py:16-29`: Docstring con cláusulas operativas directas (122 tokens).
3. `src/tool_politicas.py:38-51`: Docstring de políticas sin narrativa superflua (187 tokens).
4. `src/tool_sentimiento.py:44`: Assert duro de salida máxima en Tool #1 `<= 120` caracteres.
5. `src/tool_politicas.py:84`: Assert duro de salida máxima en Tool #2 `<= 600` caracteres.
6. `src/agente.py:192`: `max_output_tokens=512` y `temperature=0` para respuestas compactas y deterministas.
7. `src/agente.py:157`: Recorte dinámico de historial (`MAX_MENSAJES_HISTORIAL = 8`) preservando pares `AIMessage`-`ToolMessage` y comenzando en `HumanMessage`.
8. `src/agente.py:126`: Control de ritmo con `InMemoryRateLimiter(requests_per_second=0.25)` para respetar 15 RPM.
9. `src/config.py:14`, `src/agente.py:233`: Truncado defensivo de entrada a `MAX_CARACTERES_RESENA = 4000`.
