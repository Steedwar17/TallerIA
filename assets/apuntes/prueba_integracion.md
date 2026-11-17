# Reporte de Pruebas de Integración — Tarea 10

- **Herramienta #1 conectada:** `clasificar_sentimiento_resena` (modelo real Keras U4).
- **Herramienta #2 conectada:** `consultar_politica_triaje` (políticas de triaje U5).
- **Checkpointer:** `MemorySaver` con `thread_id`.

## Resultados por Prueba

| Prueba | Entrada | Esperadas | Invocadas | Cumple | Tokens (In/Out/Tot) | Llamadas |
|---|---|---|---|---|---|---|
| **Prueba A** | `Hola, ¿en qué me puedes ayudar?...` | `[]` | `[]` | **CUMPLE ✓** | 608 | 1 |
| **Prueba B** | `¿Qué política aplica si un cliente amena...` | `['consultar_politica_triaje']` | `['consultar_politica_triaje']` | **CUMPLE ✓** | 1325 | 2 |
| **Prueba C** | `Please classify the sentiment and determ...` | `['clasificar_sentimiento_resena', 'consultar_politica_triaje']` | `['clasificar_sentimiento_resena', 'consultar_politica_triaje']` | **CUMPLE ✓** | 2136 | 3 |
| **Prueba D** | `¿cuál fue la confianza de esa reseña?...` | `['(memoria o tool)']` | `[]` | **CUMPLE ✓** | 853 | 1 |
| **Prueba E** | `¿cuál fue la confianza de esa reseña?...` | `[]` | `[]` | **CUMPLE ✓** | 610 | 1 |
| **Prueba F** | `Customer review: 'Not bad, but not great...` | `['clasificar_sentimiento_resena', 'consultar_politica_triaje']` | `['clasificar_sentimiento_resena', 'consultar_politica_triaje']` | **CUMPLE ✓** | 2145 | 3 |

## Detalle de Respuestas Finales Obtenidas

### Prueba A
- **Entrada:** Hola, ¿en qué me puedes ayudar?
- **Herramientas invocadas:** `[]`
- **Respuesta final:**
> ¡Hola! Puedo ayudarte a clasificar reseñas de clientes (positivas o negativas), analizar su nivel de confianza y consultar las políticas internas de triaje y respuesta correspondientes para cada caso. 

¿Tienes alguna reseña que necesites que procese?

### Prueba B
- **Entrada:** ¿Qué política aplica si un cliente amenaza con demandar por una pésima experiencia?
- **Herramientas invocadas:** `['consultar_politica_triaje']`
- **Respuesta final:**
> **Política aplicada:** POL_NEG  
**Prioridad:** ALTA  
**Acción:** Ticket de insatisfacción  
**Pauta:** Empatizar con el malestar, agradecer la franqueza y ofrecer soporte directo.

### Prueba C
- **Entrada:** Please classify the sentiment and determine the action plan under corporate prot...
- **Herramientas invocadas:** `['clasificar_sentimiento_resena', 'consultar_politica_triaje']`
- **Respuesta final:**
> - **Sentimiento:** Negativo
- **Confianza:** 1.0
- **Prioridad:** ALTA
- **Acción:** Ticket de insatisfacción
- **Pauta:** Empatizar con malestar, agradecer franqueza y ofrecer soporte directo.

### Prueba D
- **Entrada:** ¿cuál fue la confianza de esa reseña?
- **Herramientas invocadas:** `[]`
- **Respuesta final:**
> La confianza de la clasificación de la última reseña fue de **1.0** (100%).

### Prueba E
- **Entrada:** ¿cuál fue la confianza de esa reseña?
- **Herramientas invocadas:** `[]`
- **Respuesta final:**
> No tengo el contexto de cuál reseña estás preguntando, ya que no se ha analizado ninguna en este turno. 

Por favor, proporcione el texto de la reseña que desea consultar para que podamos clasificarla y verificar su nivel de confianza.

### Prueba F
- **Entrada:** Customer review: 'Not bad, but not great either. It sits right in the middle.' Please analyze and apply our protocol.
- **Herramientas invocadas:** `['clasificar_sentimiento_resena', 'consultar_politica_triaje']`
- **Respuesta final:**
> * **Sentimiento detectado:** Negativo (con baja confianza: 0.58)
* **Política aplicada:** POL_REV
* **Prioridad:** MEDIA
* **Acción:** Derivar a moderación
* **Pauta:** Solicitar cortésmente mayor detalle o clarificación al usuario.
