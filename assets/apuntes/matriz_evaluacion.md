# Matriz de Evaluación del Agente de Triaje

| caso   | entrada resumida                                      | herramientas esperadas                                   | herramientas invocadas                                   | trayectoria   | resultado   |   tokens |
|:-------|:------------------------------------------------------|:---------------------------------------------------------|:---------------------------------------------------------|:--------------|:------------|---------:|
| Caso 1 | Hola, ¿qué puedes hacer por mí?                       | []                                                       | []                                                       | CUMPLE        | CUMPLE      |      609 |
| Caso 2 | ¿Qué política debo aplicar si un cliente amenaza c... | consultar_politica_triaje                                | consultar_politica_triaje                                | CUMPLE        | CUMPLE      |     1365 |
| Caso 3 | Haz el triaje de esta reseña de un cliente: "Widow... | clasificar_sentimiento_resena, consultar_politica_triaje | clasificar_sentimiento_resena, consultar_politica_triaje | CUMPLE        | CUMPLE      |     2230 |
| Caso 4 | Haz el triaje de esta reseña de un cliente: "Not b... | clasificar_sentimiento_resena, consultar_politica_triaje | clasificar_sentimiento_resena, consultar_politica_triaje | CUMPLE        | CUMPLE      |     2130 |
