# AGENTS.md — Reglas para el agente de código (Antigravity)

Este archivo define cómo debe trabajar el agente en el repositorio del **Taller Evaluativo Unidades 4 y 5: Sistemas Neuro-Agénticos Interactivos** (Universidad de Boyacá).
Usuario: **Juan Diego**. Compañero: **Steedwar**. Ruta 3 (PLN con embeddings + agente de triaje), dataset `stanfordnlp/imdb`.

**Antes de hacer cualquier cosa, lee `spec.md` completo. Es la única fuente de verdad.**

---

## 1. Reglas de oro (no negociables)

1. **No inventes nada.** Si algo no está en `spec.md`, no lo supongas: está marcado como **[ABIERTO]** o no existe. **Pregunta a Juan Diego antes de decidir.**
2. **Pregunta antes de suponer.** Ante cualquier ambigüedad (librería, versión, nombre de archivo, modelo de Gemini, formato de salida, estructura de datos), detente y haz una pregunta concreta. Una pregunta a la vez, con opciones si es posible.
3. **No toques lo que no es de Juan Diego.** Solo trabaja en las tareas **2, 4, 6, 8, 10, 12, 14, 16, 18** y la **19** (conjunta). No edites archivos de las tareas de Steedwar (1, 3, 5, 7, 9, 11, 13, 15, 17) salvo para leerlos o fusionarlos cuando la tarea lo indique (14 y 18).
4. **Nunca inventes cifras, métricas, probabilidades ni resultados.** Todo número que aparezca en el README debe salir de las salidas guardadas de los notebooks o de las gráficas de `assets/`. Si no existe, dilo y pide ejecutar el notebook.
5. **Nunca escribas el README por Juan Diego como texto genérico.** El taller prohíbe copiar/parafrasear texto de IA en el README (tope 2.0 por criterio). Tu rol en el README es: dar estructura, listar qué cifras/capturas debe referenciar y revisar. **El texto lo redacta Juan Diego** con sus propias palabras.
6. **Seguridad de secretos:** jamás pidas, imprimas, comitees ni escribas en archivos versionados la `GOOGLE_API_KEY`. Solo `.env` (ignorado por git) y `.env.example` con `GOOGLE_API_KEY=tu_clave_aqui`.
7. **No hagas commits ni push sin confirmación explícita.** Propón el mensaje de commit y espera aprobación.

---

## 2. Flujo de trabajo obligatorio

Para cada tarea:

1. Indica qué tarea vas a trabajar (número y nombre) y cita los requisitos de `spec.md` §7.2.
2. Lista las **dudas [ABIERTO]** que afectan esa tarea y pregunta.
3. Propón un plan corto (archivos que crearás/editarás) y **espera aprobación**.
4. Implementa en pequeños pasos, en la **rama propia de Juan Diego**.
5. Ejecuta/prueba lo que hiciste y muestra la salida real.
6. Recuerda a Juan Diego el **registro de Vibe Coding** (sección 5 de este archivo).
7. Marca la tarea como terminada solo cuando cumpla todos los puntos del spec para esa tarea.

---

## 3. Reglas técnicas del repositorio

### 3.1 Estructura (exacta, según el taller)
```
README.md · requirements.txt · .env.example · .gitignore · app.py
src/ · notebooks/ (entrenamiento_u4.ipynb, evaluacion_agente_u5.ipynb) · models/ · assets/
```
No agregues carpetas ni archivos fuera de esta estructura sin preguntar (excepto las copias de trabajo permitidas: `app_agent.py`, `evaluacion_casos_3_4.ipynb`, `README_agente.md`, que luego se fusionan).

### 3.2 Colaboración
- Una **rama propia** por integrante.
- Archivos compartidos (`app.py`, `evaluacion_agente_u5.ipynb`, `README.md`, `requirements.txt`) se editan **en copias separadas** y se fusionan (tareas 14 y 18).
- Nunca sobrescribas trabajo de Steedwar al fusionar: muestra el diff y pide confirmación.

### 3.3 Agente (LangChain/LangGraph + Google AI Studio)
- Usar `create_agent` **o** `StateGraph` (pregunta cuál). Verifica la API **actual** de LangChain/LangGraph en la documentación oficial antes de escribir código: el taller advierte que la IA suele proponer **APIs desactualizadas**. Si no puedes verificar, dilo.
- Clave: `GOOGLE_API_KEY` leída desde `.env`.
- Memoria: `MemorySaver` (o equivalente) como checkpointer; `thread_id` por sesión; función para generar/reiniciar el `thread_id`; al reiniciar, la memoria debe borrarse.
- Código modular: una función que reciba la **lista de herramientas**, para poder usar una @tool #1 provisional (simulada) y luego la real.
- **Dos herramientas exactamente:**
  - **@tool #1** (de Steedwar): inferencia del modelo local → resumen compacto (predicción + probabilidad/confianza).
  - **@tool #2**: RAG ligero **o** reglas/cálculo de escalamiento.
- **Docstrings** de las tools: deben indicar *cuándo usarla* y *qué recibe/retorna*, porque guían al agente. Se documentan para la Sección 4 del README.
- **Ahorro de tokens:** salidas compactas de las tools, respuestas breves del agente. Documentar cada medida realmente implementada (no listar medidas que no existan).

### 3.4 Evaluación del agente
- 3 a 4 casos: Caso 1 (sin tools), Caso 2 (una sola tool), Caso 3 (y 4 opcional) (ambas tools).
- La trayectoria se verifica **desde los `tool_calls` de los mensajes**, nunca leyendo la respuesta final.
- Por caso imprimir: herramientas esperadas, invocadas y resultado (cumple/no cumple).
- El notebook final se entrega **ejecutado, con todas las salidas guardadas**.

### 3.5 Aplicación (`app.py`)
- Streamlit o Gradio (según decisión pendiente).
- Debe: aceptar texto para la red neuronal, chatear con el agente con memoria, tener **botón de reinicio de `thread_id`**, mostrar **salida directa del modelo** y **qué @tool invocó el agente** (leyendo `tool_calls`), y usar **caché** para modelo y agente (ej. `@st.cache_resource`).

### 3.6 Calidad
- Código claro, comentado lo justo, sin dependencias que no estén en `requirements.txt`.
- No agregues librerías sin preguntar; si hace falta una, explica por qué.
- Evita fugas de datos (el vectorizador/vocabulario se ajusta **solo con train**; es tarea de Steedwar, pero avisa si ves un riesgo al integrar).
- No cambies el modelo ni los pesos de Steedwar.

---

## 4. Reglas para el README (lo que el agente SÍ puede hacer)

El agente puede:
- Proponer el **esqueleto** de cada sección (títulos y viñetas de qué debe cubrir, según `spec.md` §5).
- Listar las **cifras y capturas** que Juan Diego debe citar y de qué celda/archivo salen.
- Revisar que toda cifra citada exista en un notebook o en `assets/`.
- Revisar enlaces, imágenes y formato.
- Generar el **diagrama** de la Sección 2 solo si Juan Diego lo pide, y debe reflejar la arquitectura real implementada (interfaz → agente → @tool #1 / @tool #2 → memoria `thread_id`).

El agente NO puede:
- Redactar párrafos explicativos listos para pegar.
- Citar cifras que no estén en las salidas.

---

## 5. Bitácora de Vibe Coding (Sección 5 del README)

Cada vez que el agente propone código y ocurre **un bug, una API desactualizada, una fuga de datos o una mala práctica**, el agente debe:
1. Avisar a Juan Diego: *"Esto es un episodio de bitácora."*
2. Recordarle **tomar captura de la conversación en el momento y guardarla en `assets/`** (más el enlace compartido si la herramienta lo permite).
3. Pedirle que anote: qué propuso la IA, cuál era el problema y cómo se corrigió.

Declarar uso de IA sin captura descuenta **0.5** de la nota final.

---

## 6. Qué hacer cuando falte información

- Faltan datos de Steedwar (p. ej. firma de la @tool #1): **no la inventes**. Usa la provisional simulada indicada en la tarea 6 y pregunta a Juan Diego cuándo la entregan.
- Falta una decisión [ABIERTO]: pregunta.
- Hay conflicto entre el spec y una sugerencia tuya: gana el spec; señala el conflicto.
- No puedes verificar algo (versión de librería, nombre de modelo): dilo explícitamente en vez de afirmarlo.

---

## 7. Checklist final (tarea 19) que el agente debe ayudar a verificar

- [ ] Repositorio Público
- [ ] Imágenes de `assets/` cargan en el README
- [ ] No hay `.env` real en el repo (revisar también el historial de git)
- [ ] Notebooks guardados con salidas
- [ ] Cada episodio de Vibe Coding tiene captura en `assets/`
- [ ] Toda cifra del README aparece en notebooks o gráficas
- [ ] Correo a `jscely@uniboyaca.edu.co`, asunto `TU4-U5 - Apellidos Integrantes`, con nombres completos, Ruta 3 y enlace al repo
- [ ] Ningún commit después del envío
