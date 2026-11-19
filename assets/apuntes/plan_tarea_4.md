# Plan de Implementación — Tarea 4: Diseño de la @tool #2 (RAG Ligero de Políticas de Triaje)

## Descripción del Objetivo
Implementar en `src/tools.py` la herramienta `@tool #2` para el agente de triaje inteligente de la Ruta 3 (dataset IMDB). La herramienta implementará un **RAG ligero** que consulta las políticas internas de triaje aprobadas (`POLITICA_NEGATIVO`, `POLITICA_POSITIVO`, `POLITICA_INCIERTO`), devolviendo una respuesta compacta con la directriz de acción y pauta de respuesta según la categoría y consulta, optimizando el consumo de tokens.

Adicionalmente, se implementará un script de prueba con entradas simuladas (`tests/test_tool_rag.py` o bloque de ejecución directa en `src/tools.py`) para verificar el comportamiento de la tool de forma 100% independiente sin depender del modelo de la Unidad 4 de Steedwar.

---

## Decisiones Acordadas con Juan Diego
1. **Tipo de herramienta:** RAG ligero sobre políticas de triaje.
2. **Categorías reconocidas:** `positivo`, `negativo`, y `neutral` / `incierto` (para casos de baja confianza o probabilidad cercana a 0.50).
3. **Contenido de las políticas:**
   - `POLITICA_NEGATIVO`: Prioridad Alta. Registrar ticket de insatisfacción, agradecer franqueza, empatizar y ofrecer canal de contacto.
   - `POLITICA_POSITIVO`: Prioridad Estándar. Clasificar como reseña destacada / promotor, agradecer y recomendar títulos afines.
   - `POLITICA_INCIERTO`: Prioridad Media / Revisión Humana. No automatizar categorización final, solicitar clarificación o derivar a moderador.
4. **Motor RAG:** RAG semántico con `langchain_google_genai.GoogleGenerativeAIEmbeddings` e `InMemoryVectorStore` de `langchain_core.vectorstores`. Con fallback determinista en memoria en caso de no tener API key configurada o cuota disponible durante pruebas unitarias.
5. **Librerías:** 0 dependencias adicionales (usa paquetes ya declarados en `requirements.txt`).

---

## Cambios Propuestos

### Componente: `src/`

#### [NEW] `src/policies.py`
Contendrá el texto estructurado de los documentos de políticas de triaje en formato de documentos `Document` de LangChain:
- Metadatos por política (`category`: `negativo`, `positivo`, `incierto`/`neutral`).
- Contenido sintético, claro y libre de relleno para minimizar tokens al indexar y recuperar.

#### [NEW] `src/tools.py`
Contendrá la definición de la `@tool #2`:
- Decorador `@tool` de `langchain_core.tools`.
- Firma tipada: `consultar_politicas_triaje(categoria: str, consulta: str = "") -> str`
- **Docstring exhaustivo y optimizado para el LLM:** Explica cuándo usarla (después de tener el sentimiento/categoría o ante duda de protocolo), qué argumentos recibe y qué formato compacto retorna.
- Lógica de recuperación compacta: indexa los documentos de políticas en memoria (`InMemoryVectorStore` con embeddings o filtro de metadatos/similitud) y formatea el resultado en un resumen de pocas líneas (Acción recomendada, Pauta de respuesta, Prioridad).

#### [NEW] `src/test_tool2.py`
Script de prueba reproducible e independiente que ejecuta 3 casos simulados:
1. Categoría `"negativo"` con certeza alta.
2. Categoría `"incierto"` / `"neutral"` (baja certeza del modelo de Steedwar).
3. Categoría `"positivo"` con certeza alta.
Imprime en consola los resultados reales retornados por la herramienta.

---

## Plan de Verificación

### Pruebas Automatizadas
Ejecución del script `src/test_tool2.py` mediante el entorno Python:
```powershell
python src/test_tool2.py
```
Se inspeccionarán las salidas reales en consola para verificar:
- Que no arroje excepciones ni advertencias de deprecación de LangChain.
- Que las salidas sean compactas (< 100 tokens por respuesta).
- Que oriente adecuadamente la acción de triaje para cada categoría.
