# Guion de video — Tarea 17: Demostración de la interfaz

**Duración objetivo:** 2:50 (límite de entrega: 3:00).  
**Proyecto:** Sistemas Neuro-Agénticos Interactivos — Ruta 3: PLN con Embeddings + Agente de Triaje Inteligente.  
**Integrantes:** Steedwar y Juan Diego.

## Lista previa a grabar

- [ ] La app está iniciada y el modelo neuronal ya terminó de cargar.
- [ ] La conversación está reiniciada y se muestra un `thread_id` nuevo.
- [ ] No se hicieron ensayos adicionales inmediatamente antes de grabar, para cuidar la cuota de Gemini.
- [ ] Notificaciones del sistema y del navegador están silenciadas.
- [ ] El zoom permite leer la categoría, la confianza y el detalle de herramientas.
- [ ] La terminal, el archivo `.env`, claves API, rutas personales y pestañas ajenas quedan fuera del cuadro.
- [ ] Se abre el expander **Detalle técnico de herramientas** cuando se indique abajo.

## Secuencia de grabación

| Tiempo | Qué se muestra en pantalla | Qué se dice en una frase | Texto exacto a pegar |
|---|---|---|---|
| 0:00–0:15 | Título de la app, Ruta 3 y nombres de integrantes; mostrar las dos pestañas. | “Esta aplicación integra la red neuronal de sentimiento y el agente de triaje con memoria.” | *(No pegar texto)* |
| 0:15–0:32 | Pestaña **Clasificación Neuronal Directa**. Pegar una reseña positiva, clasificar y dejar visibles categoría, confianza, barra y probabilidades. | “Primero vemos la salida local de la red, sin usar el agente ni la API.” | `This movie was absolutely wonderful. The acting was superb, the story kept me engaged from start to finish, and the ending left me smiling. Highly recommended!` |
| 0:32–0:45 | Pestaña del agente. Caso 1; dejar visible el indicador “Sin herramientas invocadas”. | “Un saludo se responde directamente: no requiere clasificar ni consultar políticas.” | `Hola, ¿qué puedes hacer por mí?` |
| 0:45–1:05 | Caso 2; dejar visible `@tool #2 consultar_politica_triaje` y su detalle técnico. | “Para una consulta de protocolo, el agente invoca solo la herramienta de política.” | `¿Qué política debo aplicar si un cliente amenaza con acciones legales en su reseña?` |
| 1:05–1:32 | Caso 3; expandir detalle técnico y mostrar primero clasificación y después política. | “Con una reseña real, el agente clasifica el sentimiento y luego determina la acción de triaje.” | `Haz el triaje de esta reseña de un cliente: "Widow hires a psychopath as a handyman. Sloppy film noir thriller which doesn't make much of its tension promising set-up. (3/10)"` |
| 1:32–1:52 | Caso 4 (Caso B); herramientas y detalle visibles. Volver brevemente al panel neuronal si hace falta para mostrar la confianza baja y la advertencia. | “Este caso ambiguo queda bajo el umbral de confianza y se deriva a revisión humana.” | `Haz el triaje de esta reseña de un cliente: "Not bad, but not great either. It sits right in the middle."` |
| 1:52–2:12 | Pestaña neuronal: pegar el Caso A. Dejar visibles la predicción positiva y probabilidad 1.0000; mencionar que el valor real documentado es negativo. | “Este es el fallo real de test: el modelo predice positivo con confianza máxima para una reseña realmente negativa.” | `Primary plot!Primary direction!Poor interpretation.` |
| 2:12–2:35 | En el mismo `thread_id`, preguntar por el sentimiento de la reseña anterior; mostrar que usa el contexto. Pulsar **Reiniciar conversación** y repetir la misma pregunta, mostrando que ya no conserva el contexto. | “La memoria existe dentro del hilo, pero al reiniciarlo se aísla y no arrastra la conversación previa.” | `¿Cuál fue el sentimiento de la reseña anterior?` |
| 2:35–2:50 | Vista final con las pestañas y un resultado/herramientas legibles. | “Con esto se evidencia la inferencia neuronal, la trazabilidad de tools y el control de memoria del sistema.” | *(No pegar texto)* |

## Referencia de resultados esperados durante la toma

| Caso | Herramientas esperadas, en orden | Resultado que debe quedar visible |
|---|---|---|
| Caso 1 | Ninguna | Indicador sin herramientas. |
| Caso 2 | `consultar_politica_triaje` | Política de triaje, normalmente `POL_NEG`. |
| Caso 3 | `clasificar_sentimiento_resena` → `consultar_politica_triaje` | Sentimiento negativo y acción/política. |
| Caso 4 / Caso B | `clasificar_sentimiento_resena` → `consultar_politica_triaje` | Confianza 0.5835 (tolerancia 1e-4), baja confianza y revisión humana / `POL_REV`. |
| Fallo real / Caso A | Panel neuronal; opcionalmente las dos tools si se envía al agente | Real: negativo; predicción: positivo; probabilidad positiva y confianza: 1.000000. |

## Publicación del video

1. Publicar en YouTube como **No listado**, o en Google Drive con acceso **Cualquier persona con el enlace**.
2. Abrir el enlace en una ventana de incógnito sin iniciar sesión y comprobar que reproduce el video.
3. Confirmar que la duración final es de **3:00 o menos**.
4. Entregar el enlace para registrarlo en `assets/apuntes/enlace_video.md` antes del correo de entrega: los commits posteriores no cuentan.
