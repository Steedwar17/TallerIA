# Firma de la @tool #1 (para Juan Diego)

- **Import exacto:** `from src.tool_sentimiento import clasificar_sentimiento_resena`
- **Nombre:** `clasificar_sentimiento_resena`
- **Firma:** `clasificar_sentimiento_resena(texto: str) -> str`
- **Entrada:** el texto literal y completo de la reseña (inglés).
- **Salida:** JSON de una línea (máx. 120 caracteres), p. ej. entrada `"This movie was absolutely wonderful, a brilliant story with superb acting. I loved every minute."` -> `{"sentimiento":"positivo","confianza":1.0,"baja_confianza":false}` (65 caracteres).
- **Error (nunca lanza):** entrada `""` -> `{"error":"El texto está vacío o solo contiene espacios."}`
- **Plan usado: A** (modelo extremo a extremo `Input(string) -> TextVectorization(standardize=None) -> modelo_mejorado`). Diferencia máxima en proceso nuevo: 0.0 (< 1e-5). La limpieza está en `src/preprocesamiento.py::limpiar_texto`.
- **Tamaño de `models/sentimiento_imdb.keras`:** 7.58 MB.
- **Latencias medidas (`python -m src.prueba_inferencia`):** primera llamada 57.9 ms; segunda 1.8 ms. El modelo se carga una sola vez (`lru_cache`, hits 0 -> 1).
- **Funciones de apoyo:** `src.inferencia.predecir_sentimiento(texto) -> {"sentimiento","confianza","p_positivo","baja_confianza"}`.
- **Avisos para integrar:** (1) `src/config.py` no existía; creé uno provisional con `SEQ_LEN = 585` (no 300): es el MAX_LEN con que se entrenó el modelo. (2) El nombre real de la tool #2 en el repo es `consultar_politicas_triaje(categoria, consulta)` y el contrato dice `consultar_politica_triaje(consulta, sentimiento, confianza)`; hay que alinearlos en la tarea 10.
