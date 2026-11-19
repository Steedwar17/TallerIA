# Uso de IA — Gemini integrado en Google Docs (inicio del proyecto)

- Herramienta: Gemini integrado en Google Docs.
- Momento: al inicio del proyecto (Tarea 1, preparación del dataset IMDB).
- Qué generó: el código base de carga del dataset:

```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb")
```

- Contexto: el documento de Google Docs contenía la tabla de reparto de tareas (Steedwar / Juan Diego, tareas 1 a 19). La Tarea 1 de Steedwar pide cargar `stanfordnlp/imdb`, explorar los splits, preprocesar y crear validación sin fugas de datos.
- Captura: pendiente de guardar en `assets/vibe_coding/`.
