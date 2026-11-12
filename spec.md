# SPEC — Taller Evaluativo Unidades 4 y 5: Sistemas Neuro-Agénticos Interactivos

> Fuente única de verdad: el PDF "Taller Unidades 4 y 5" (Universidad de Boyacá) y el PDF "Tareas" (reparto del grupo).
> Regla para el agente: este documento NO debe ampliarse con suposiciones. Todo lo marcado como **[ABIERTO]** no está definido en las fuentes: se debe preguntar a Juan Diego antes de decidir.

---

## 0. Datos generales

| Campo | Valor (según las fuentes) |
|---|---|
| Modalidad | Grupal (2 o 3 integrantes) |
| Grupo | Steedwar + Juan Diego |
| Medio de entrega | Enlace a repositorio público de GitHub enviado a jscely@uniboyaca.edu.co |
| Ruta elegida | **Ruta 3: PLN con Embeddings + Agente de Triaje Inteligente** |
| Dataset | IMDB Reviews: `from datasets import load_dataset; ds = load_dataset("stanfordnlp/imdb")` |
| Escala de calificación | 0.0 a 5.0 |
| Interfaz | Streamlit **o** Gradio **[ABIERTO: el grupo debe elegir una; las tareas 11 y 12 dicen "Streamlit o Gradio" sin decidir]** |
| Quién es el usuario de este spec | Juan Diego (tareas pares + 19 conjunta) |

---

## 1. Objetivo del taller

Diseñar, entrenar, integrar y desplegar una solución interactiva de IA en Streamlit o Gradio que articule dos paradigmas:

1. **Hemisferio Perceptivo / Predictivo (Unidad 4 — IA Moderna):** red neuronal entrenada y validada experimentalmente por el equipo (CNN, RNN/LSTM/GRU o red con capa Embedding/Transformers) que procese datos crudos del problema y entregue predicciones junto con su nivel de confianza.
2. **Hemisferio de Razonamiento y Acción (Unidad 5 — Agentic AI):** agente autónomo construido con LangChain/LangGraph y el API de Google AI Studio que:
   - integre el modelo de red neuronal como herramienta (**@tool #1**),
   - lo complemente con una segunda herramienta (**@tool #2**: RAG ligero o función de cálculo/reglas),
   - mantenga memoria conversacional por sesión (`thread_id`),
   - sea evaluado en su trayectoria y resultado.

### Diagrama de referencia del PDF (flujo)
Usuario → Interfaz web Streamlit/Gradio → Agente inteligente ReAct/LangGraph → { Memoria conversacional `thread_id` | @tool 1 Red neuronal (Modelo Unidad 4 → Inferencia local) | @tool 2 Herramienta adicional RAG/Cálculo/Dominio → Recuperación de información o ejecución de cálculos }

---

## 2. Ruta 3 (la elegida)

| Campo | Contenido según el taller |
|---|---|
| Componente U4 | Red neuronal para clasificar sentimiento, urgencia o categoría en textos |
| Componente U5 | **@tool 1:** ejecuta la red neuronal sobre el ticket/reseña y retorna categoría + certeza. **@tool 2:** consulta políticas internas de respuesta (RAG) o verifica reglas de escalamiento |
| Datasets sugeridos | IMDB Reviews, Amazon/Yelp Reviews, Customer Support Tickets (Kaggle), PQRS/quejas (datos.gov.co / Hugging Face), AG News u otro corpus. **El grupo eligió IMDB (stanfordnlp/imdb).** |

(Las rutas 1, 2 y 4 no aplican a este grupo.)

---

## 3. Requerimientos técnicos obligatorios

### 3.1 Componente de Red Neuronal (Unidad 4)

- **Entrenamiento y comparación experimental** en `notebooks/entrenamiento_u4.ipynb`: preparar el dataset, entrenar y comparar **al menos 2 configuraciones** (ej. Embedding+Dense vs. Embedding+LSTM), y exportar pesos (`.keras`, `.h5`, `.pt` o `.safetensors`) en `models/`.
  - Si el archivo supera 100 MB (límite de GitHub): usar Git LFS o Hugging Face Hub y explicar la descarga en la **Sección 7** del README.
- **Control de sobreajuste/subajuste:** generar y guardar las curvas reales de pérdida (loss) y métrica de evaluación, entrenamiento vs. validación.
- **Análisis de un fallo real (Error Analysis):** identificar al menos un ejemplo del conjunto de prueba donde la red falle o arroje baja confianza. El notebook debe imprimir: identificador (índice o nombre de archivo), valor real, predicción y probabilidad. Esa salida debe quedar guardada en el notebook para analizarla en el README.

### 3.2 Componente Agéntico y Gestión de Tokens (Unidad 5)

- **Dos (2) herramientas @tool** bien definidas, para optimizar tokens y respetar los límites gratuitos de Google AI Studio:
  - **@tool #1 (Inferencia Unidad 4):** envuelve el modelo entrenado localmente y retorna un **resumen compacto** (predicción + probabilidad/confianza).
  - **@tool #2 (Conocimiento/Reglas de Dominio):** RAG sobre documentos del problema **o** función de cálculo/reglas de negocio.
- **Memoria y estado (checkpointer + thread_id):** el agente (creado con `create_agent` o `StateGraph` de LangGraph) debe mantener memoria entre turnos usando `MemorySaver` (o equivalente) y permitir gestionar/reiniciar la sesión mediante `thread_id`.
- **Evaluación del agente** en `notebooks/evaluacion_agente_u5.ipynb`: 3 a 4 casos de prueba midiendo **Trayectoria** (herramientas esperadas vs. invocadas) y **Resultado**:
  - **Caso 1:** saludo o pregunta general; el agente NO debe invocar ninguna herramienta.
  - **Caso 2:** consulta que requiere únicamente una de las herramientas.
  - **Caso 3 (y 4 opcional):** caso completo que requiere invocar @tool #1 y complementar con @tool #2.
  - La trayectoria se verifica **en el código a partir de los mensajes del agente (`tool_calls`)**, no leyendo la respuesta final.

### 3.3 Aplicación interactiva en Streamlit o Gradio (`app.py`)

1. **Interacción de extremo a extremo:** permitir ingresar los datos para la red neuronal y conversar con el agente manteniendo la memoria de la sesión (con **botón para reiniciar el thread_id**).
2. **Transparencia técnica en pantalla:** visualizar la salida directa del modelo de la Unidad 4 **y** el indicador de qué herramientas (@tool) invocó el agente en su respuesta.
3. **Uso de caché:** cargar el modelo entrenado y el agente en caché (ej. `@st.cache_resource` en Streamlit) para no recargar pesos en cada mensaje.

---

## 4. Uso de IA generativa (Vibe Coding) y estructura del repositorio

Se permite y promueve usar asistentes de IA (ChatGPT, Claude, Gemini, Cursor, Copilot) para codificar. **Está prohibido copiar y pegar texto genérico generado por un LLM en las explicaciones del README.md.** Toda explicación debe referenciar directamente las gráficas, métricas numéricas, casos de error reales del modelo y decisiones técnicas del equipo.

Cómo se verificará la autoría:
- **Trazabilidad de cifras:** notebooks entregados ejecutados y con salidas guardadas. Toda métrica, probabilidad o ejemplo citado en el README debe aparecer en esas salidas o en las gráficas de `assets/`.
- **Tope por texto de IA:** si la explicación de un criterio es transcripción o paráfrasis de una respuesta de IA, o cita cifras que no aparecen en los notebooks, ese criterio se califica con **máximo 2.0**.
- **Evidencia de la bitácora:** cada episodio de Vibe Coding incluye captura en `assets/` y, si la herramienta lo permite, el enlace compartido de la conversación. Declarar uso de IA sin captura **descuenta 0.5 de la nota final**.

### Estructura obligatoria del repositorio

```
nombre-del-proyecto/
├── README.md                     # Informe técnico y defensa (7 secciones obligatorias)
├── requirements.txt              # Dependencias necesarias para ejecutar el proyecto
├── .env.example                  # Plantilla: GOOGLE_API_KEY=tu_clave_aqui (¡NUNCA subir el .env real!)
├── .gitignore                    # Excluir .env, .venv, __pycache__, etc.
├── app.py                        # Aplicación interactiva en Streamlit o Gradio
├── src/                          # Módulos del agente, herramientas (@tool) e inferencia del modelo
├── notebooks/
│   ├── entrenamiento_u4.ipynb    # Entrenamiento y comparación experimental de la red neuronal (U4)
│   └── evaluacion_agente_u5.ipynb# Evaluación de trayectoria y resultado del agente en 3-4 casos (U5)
├── models/                       # Pesos exportados del modelo entrenado
└── assets/                       # Gráficas de entrenamiento, capturas de la app y evidencias de Vibe Coding
```

Nombre del repositorio (`nombre-del-proyecto`): **[ABIERTO]**

---

## 5. Especificación del README.md (7 secciones obligatorias)

Debe estar escrito con la voz y análisis propio del equipo.

| # | Sección | Contenido exigido |
|---|---|---|
| 1 | Identificación, Ruta y Propuesta de Valor | Título del proyecto, nombres completos de integrantes, ruta y dataset, justificación de por qué se requiere red neuronal (U4) + agente (U5) en lugar de solo un prompt a un LLM, **tabla breve de qué componente desarrolló cada integrante** |
| 2 | Arquitectura del Sistema (Diagrama Propio) | Diagrama con la conexión entre interfaz Streamlit/Gradio, agente, @tool #1 (Red Neuronal), @tool #2 (Conocimiento/Cálculo) y memoria (thread_id) |
| 3 | Ingeniería del Modelo de Red Neuronal (U4) y Análisis de un Fallo Real | Justificación de arquitectura, hiperparámetros, gráficas reales entrenamiento vs. validación comparando modelo base vs. mejorado |
| 4 | Diseño del Agente y Matriz de Evaluación (U5) | Descripción de las 2 @tool, cómo se redactaron los docstrings para guiar al agente, manejo de memoria (thread_id), medidas para cuidar tokens/cuota de la API |
| 5 | Bitácora de Vibe Coding y Razonamiento Crítico | Dónde la IA propuso código con bugs, APIs desactualizadas de LangChain/Keras, fugas de datos o malas prácticas, y cómo el equipo lo corrigió |
| 6 | Demostración de la Interfaz | Capturas en `assets/` y enlace a video corto (**máx. 3 min**, YouTube no listado o Google Drive) mostrando la app ejecutando los casos de evaluación, incluido el caso de fallo de la Sección 3, con la visualización de la red neuronal y las herramientas invocadas |
| 7 | Guía de Instalación y Ejecución Reproducible | Pasos exactos: clonar repo, instalar `requirements.txt`, configurar `.env` desde `.env.example`, ejecutar la app |

---

## 6. Entrega y calificación

1. **Verificación previa:** repositorio **Público**; todas las imágenes de `assets/` cargan en el README; **NO** subir `.env` con la `GOOGLE_API_KEY`; notebooks guardados con sus salidas.
2. **Envío por correo** a `jscely@uniboyaca.edu.co`:
   - Asunto: `TU4-U5 - Apellidos Integrantes`
   - Cuerpo: nombres completos de los integrantes, ruta seleccionada (Ruta 3) y enlace directo al repositorio.
   - **Los commits posteriores al envío no se tienen en cuenta.**
3. **Distribución (100%):**
   - Componente 1 — Modelo de Red Neuronal y Rigor Experimental (U4): **25%**
   - Componente 2 — Arquitectura Agéntica, Herramientas y Evaluación (U5): **25%**
   - Componente 3 — Aplicación Interactiva y Eficiencia: **20%**
   - Componente 4 — Calidad del README, Razonamiento Propio y Bitácora: **30%**

Fecha límite de entrega: **[ABIERTO: no aparece en los PDF]**

---

## 7. Reparto de tareas del grupo (PDF "Tareas")

Convención de trabajo: cada integrante trabaja en **una rama propia** y solo edita los archivos que le asignan las tareas. Los archivos compartidos (`app.py`, `evaluacion_agente_u5.ipynb`, `README.md`, `requirements.txt`) se editan en **copias separadas por integrante** y se fusionan en la tarea 18/19 (el PDF dice "tarea 19" en la convención; la fusión concreta ocurre en la tarea 14 para el notebook y en la 18 para README, app y requirements).

### 7.1 Vista general

| Tarea | Responsable | Resumen |
|---|---|---|
| 1 | Steedwar | Preparación del dataset IMDB |
| **2** | **Juan Diego** | Estructura del repo y archivos base |
| 3 | Steedwar | Entrenamiento modelo base |
| **4** | **Juan Diego** | Diseño @tool #2 |
| 5 | Steedwar | Modelo mejorado y comparación |
| **6** | **Juan Diego** | Configuración del agente (LangChain/LangGraph + Google AI Studio) |
| 7 | Steedwar | Error Analysis |
| **8** | **Juan Diego** | Ahorro de tokens y apuntes del agente |
| 9 | Steedwar | Exportación del modelo y @tool #1 |
| **10** | **Juan Diego** | Integración de las dos tools en el agente |
| 11 | Steedwar | app.py parte red neuronal |
| **12** | **Juan Diego** | app.py parte agente |
| 13 | Steedwar | Evaluación casos 1 y 2 |
| **14** | **Juan Diego** | Evaluación casos 3 y 4 + fusión + ejecución completa |
| 15 | Steedwar | README secciones 3 y 5 (parte modelo) |
| **16** | **Juan Diego** | README secciones 2, 4 y 5 (parte agente) |
| 17 | Steedwar | Capturas y video |
| **18** | **Juan Diego** | README secciones 1, 6, 7 y fusión |
| **19** | **Steedwar + Juan Diego** | Verificación previa y entrega |

### 7.2 Tareas de Juan Diego (detalle fiel al PDF)

#### Tarea 2 — Estructura del repositorio y archivos base
- Crear el repositorio **público** en GitHub con la estructura exacta de la sección 4: `README.md`, `requirements.txt`, `.env.example` (contenido `GOOGLE_API_KEY=tu_clave_aqui`), `.gitignore` (excluyendo `.env`, `.venv`, `__pycache__`, etc.), `app.py`, `src/`, `notebooks/` (con `entrenamiento_u4.ipynb` y `evaluacion_agente_u5.ipynb`), `models/` y `assets/`.
- Verificar que el `.env` real **NO** se suba.
- Definir el `requirements.txt` inicial: Streamlit o Gradio, LangChain/LangGraph, integración con Google AI Studio, librería de la red neuronal.
- **Dar acceso de escritura a Steedwar.**
- Aplicar la convención de ramas y copias separadas.

#### Tarea 4 — Diseño de la @tool #2 (Conocimiento/Reglas de Dominio)
- Implementar en `src/` un RAG ligero sobre documentos del problema (políticas internas de respuesta) **o** una función de reglas de escalamiento según categoría/sentimiento.
- Redactar el docstring para guiar al agente: cuándo usarla y qué recibe/retorna.
- Salida **compacta** (optimizar tokens).
- Probarla de forma independiente con **entradas simuladas** de categoría/certeza (sin depender del modelo de Steedwar).
- Vibe Coding: si se usa IA, **captura en `assets/` en el momento**, con registro de errores de LangChain detectados y su corrección.
- **[ABIERTO]** Decidir entre RAG ligero o reglas de escalamiento (el PDF permite ambas).

#### Tarea 6 — Configuración del agente con LangChain/LangGraph y Google AI Studio
- Crear en `src/` el agente (con `create_agent` o `StateGraph`) usando el API de Google AI Studio, con `GOOGLE_API_KEY` leída desde `.env`.
- Integrar `MemorySaver` (o equivalente) como checkpointer y gestionar la memoria por sesión mediante `thread_id`, con una función para generar/reiniciar el `thread_id`.
- Código modular: una función que reciba la lista de herramientas, para probarlo con la @tool #2 y una **@tool #1 provisional** (función simulada que retorne predicción + probabilidad) hasta que Steedwar entregue la real.
- Vibe Coding: captura en `assets/` de episodios con IA (APIs desactualizadas de LangChain, etc.) y cómo se corrigieron.
- **[ABIERTO]** Modelo de Gemini concreto a usar (no lo define el PDF).

#### Tarea 8 — Medidas de ahorro de tokens y apuntes del agente
- Implementar y documentar en `src/` medidas para cuidar tokens/cuota de Google AI Studio (salidas compactas de las 2 tools, respuestas breves, etc.).
- Dejar apuntes de: cómo se redactaron los docstrings de las 2 herramientas, cómo se maneja la memoria (`thread_id`), qué medidas de ahorro se tomaron. Alimentan la Sección 4 del README.

#### Tarea 10 — Integración de las dos herramientas en el agente
- Reemplazar la @tool #1 provisional por la real de Steedwar (tarea 9) y conectar ambas tools.
- Verificar que el agente las invoque correctamente y mantenga memoria entre turnos con el mismo `thread_id`, y que **al reiniciar el `thread_id` la memoria se borre**.
- Dependencia: nombre exacto de la función y su firma, documentados por Steedwar en la tarea 9.

#### Tarea 12 — app.py, parte del agente
- Trabajar en una copia (ej. `app_agent.py`).
- Chat con el agente manteniendo memoria de la sesión, con **botón para reiniciar el thread_id**.
- Cargar el agente en caché (ej. `@st.cache_resource`).
- Mostrar en pantalla qué @tool invocó el agente en cada respuesta, **leyéndolas de los `tool_calls`** de los mensajes.

#### Tarea 14 — Notebook de evaluación, casos 3 y 4 (opcional)
- Trabajar en una copia (ej. `evaluacion_casos_3_4.ipynb`).
- **Caso 3:** caso completo que requiere @tool #1 y complementar con @tool #2. **Caso 4 (opcional):** otro caso completo.
- Evaluar Trayectoria (esperadas vs. invocadas, verificadas con `tool_calls`) y Resultado.
- **Fusionar** con la copia de Steedwar en `notebooks/evaluacion_agente_u5.ipynb` y **ejecutar el notebook completo, guardándolo con todas las salidas**.

#### Tarea 16 — README, secciones 2, 4 y 5 (parte agente)
- Hacerla **después** de que las tareas 13 y 14 estén ejecutadas y guardadas.
- Sección 2: diagrama propio (interfaz, agente, @tool #1, @tool #2, memoria `thread_id`).
- Sección 4: descripción de las 2 tools, redacción de docstrings, manejo de memoria, medidas de ahorro de tokens, y **matriz de evaluación con resultados reales** (esperadas vs. invocadas y resultado) del notebook `evaluacion_agente_u5.ipynb`.
- Sección 5 (parte agente): episodios de Vibe Coding con LangChain/LangGraph, con capturas en `assets/` y enlace compartido si es posible.
- Texto propio, **sin copiar ni parafrasear respuestas de IA**. Escribir en una copia (ej. `README_agente.md`).

#### Tarea 18 — README secciones 1, 6 y 7, y fusión
- Sección 1: título, nombres completos de los integrantes, Ruta 3 y dataset IMDB (`stanfordnlp/imdb`), justificación de por qué red neuronal + agente y no solo un prompt a un LLM, tabla breve de qué componente desarrolló cada integrante.
- Sección 6: insertar capturas de `assets/` y el enlace al video (lo entrega Steedwar en la tarea 17).
- Sección 7: pasos exactos para clonar, instalar `requirements.txt`, configurar `.env` desde `.env.example` y ejecutar la app (incluyendo cómo descargar los pesos si se usó Git LFS/Hugging Face; texto lo entrega Steedwar en la tarea 9).
- **Fusionar** en un único `README.md` las secciones 1 a 7 (de las tareas 15, 16 y 18), **fusionar `app_nn.py` y `app_agent.py` en `app.py`**, y finalizar `requirements.txt`.

#### Tarea 19 — Verificación previa y entrega (Steedwar + Juan Diego, al final)
Confirmar que:
- El repositorio es Público.
- Todas las imágenes de `assets/` cargan en el README.
- NO se subió el `.env` con la `GOOGLE_API_KEY`.
- Los notebooks están guardados con sus salidas.
- Cada episodio de Vibe Coding tiene su captura en `assets/`.
- Toda métrica, probabilidad o ejemplo citado en el README aparece en las salidas de los notebooks o en las gráficas de `assets/`.
- Enviar el enlace a `jscely@uniboyaca.edu.co` con asunto `TU4-U5 - Apellidos Integrantes`, con nombres completos, Ruta 3 y enlace directo al repositorio. Commits posteriores al envío no cuentan.

### 7.3 Dependencias entre tareas (derivadas del texto del PDF)

| Mi tarea | Necesita de Steedwar |
|---|---|
| 4, 6 | Nada (usa entradas simuladas y @tool #1 provisional) |
| 10 | Tarea 9: `@tool #1` real, nombre exacto de función y firma |
| 12 | Nada técnico para empezar; luego se fusiona con tarea 11 |
| 14 | Tarea 13 (copia de casos 1 y 2) para fusionar |
| 16 | Tareas 13 y 14 ejecutadas y guardadas |
| 18 | Tareas 15 y 16; tarea 17 (enlace del video); tarea 9 (texto de descarga de pesos si aplica) |
| 19 | Todo subido |

---

## 8. Lista de puntos [ABIERTOS] (preguntar a Juan Diego antes de decidir)

1. ¿Streamlit o Gradio?
2. ¿@tool #2 será RAG ligero (qué documentos de políticas) o reglas de escalamiento?
3. ¿Qué modelo de Gemini del API de Google AI Studio?
4. ¿Se usará `create_agent` o `StateGraph`?
5. Nombre del repositorio y de las ramas.
6. Versión de Python y gestor de entorno (`.venv`, etc.).
7. Categorías/etiquetas exactas que el @tool #2 recibirá (el dataset IMDB es positivo/negativo; ¿"urgencia" o categorías extra?).
8. Fecha límite de entrega.
9. Apellidos completos de ambos integrantes (para asunto del correo y README).
