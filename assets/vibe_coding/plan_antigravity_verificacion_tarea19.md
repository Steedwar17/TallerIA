# Plan generado por el agente de Antigravity: Verificación de entrega "Cumplimiento Tarea 19" (Fase 1, solo lectura)

> Evidencia de uso de IA (Vibe Coding). Texto copiado tal cual del plan que produjo el agente de Antigravity; no es redacción propia del equipo.

## Context
El usuario pidió un verificador de entrega para el taller U4-U5 (Ruta 3). Se verifica lo que está en GitHub (clon limpio de `main` en carpeta temporal fuera del repo), no la carpeta local. Fase 1 no modifica nada ni hace commits. Fase 2 (correcciones) solo si el usuario dice "corrige", con lista aprobada, un commit, `git push origin main`, sin force. Prohibido: enviar el correo, cambiar visibilidad, ramas/PR/merge. Máx. 4 llamadas a Gemini. Nunca mostrar `.env` ni la clave.

## Hallazgos preliminares (lectura local, aún no verificados en el clon)
- Ramas: existe `origin/juandiego/agente` además de `main` → req. 1 será ⚠️ (listar último commit y `git log main..origin/juandiego/agente`; no borrar).
- Autoría: Steedwar 11 commits, juandiegobb 9 → req. 4 probablemente ✅.
- `gh` NO está instalado: para visibilidad/rama por defecto usar `curl` a `https://api.github.com/repos/Steedwar17/triaje-neuro-agentico` sin autenticación (campos `private`, `default_branch`). El repo se movió de `TallerIA` a `triaje-neuro-agentico`: usar la URL nueva y avisarlo.
- README: quedan muchos "ESCRIBE AQUÍ", "BORRADOR", "pendiente de redactar" y enlace de video vacío → req. 15 ❌ (son textos del equipo; no se tocan).
- Sección 5: 5 episodios apuntan a `assets/vibe_coding/episodioN.png`; la carpeta está vacía → req. 19 ❌ (-0.5 cada uno).
- `src/` tiene archivos sospechosos de restos (`prueba_*.py`, `test_tool2.py`, `tools.py` vs `tool_sentimiento.py`/`tool_politicas.py`/`herramientas.py`): auditar con `git grep` quién los importa (req. 7).
- Pesos: `models/sentimiento_imdb.keras` = 7.9 MB (<100 MB, req. 9 ✅ probable).
- Checkpointer: `MemorySaver` en `src/agente.py:28,188`. Tools: `src/tool_politicas.py:32`, `src/tools.py:33`; localizar la de sentimiento y `@st.cache_resource` en `app.py`.

## Ejecución (orden)
1. Preparar `TMP=$(mktemp -d)` fuera del repo; `git clone https://github.com/Steedwar17/triaje-neuro-agentico.git $TMP/clon`. Scripts temporales en `$TMP/scripts`.
2. **A (1-5)**: `git branch -a`, `git fetch`, `git rev-list --left-right --count main...origin/main`, `git status`, `curl` API pública y `curl -o /dev/null -w %{http_code}` a la URL del repo, `git shortlog -sne`, `git log --author=… --name-only` agrupado → tabla autor/commits/archivos. Anotar hash entregado.
3. **B (6-9)**: `ls -a` del clon vs. estructura exigida; `git ls-files`/`find` de patrones del punto 7 + `git grep` de "provisional|simulad|mock|fake"; para cada hallazgo, `git grep` de importaciones; leer `app.py` completo (imports, NN + chat en un archivo); `ls -l models/`.
4. **C (10-11)**: `git ls-files .env`, `git check-ignore .env`, leer `.env.example`; `git log --all --full-history -- .env`; `git grep -E 'AIza[0-9A-Za-z_-]{35}' $(git rev-list --all)`; grep en salidas de notebooks (JSON) y `assets/**/*.md|json|txt`. Si aparece clave: DETENER y avisar de revocarla.
5. **D (12-14)**: script `nbformat` en `$TMP/scripts` que revise salidas, errores y `execution_count` consecutivo; extraer evidencias de req. 13 por celda; leer `src/evaluacion.py` y celdas del notebook de agente para citar la línea donde la trayectoria sale de `tool_calls`.
6. **E (15-20)**: grep de marcadores en README; leer Secciones 1 (nombres completos, Ruta 3, enlace dataset, justificación, tabla de componentes vs. req. 4) y verificar las siete secciones; script que extrae todas las imágenes `![](…)`, comprueba existencia con mayúsculas exactas en el clon, ruta relativa y `curl` a URL raw (200); tabla de trazabilidad de cifras del README contra salidas de notebooks y `assets/apuntes/cifras_*.md`; verificar episodios de Sección 5; enlace de video queda ⚠️ pendiente del usuario (y hoy es un placeholder vacío).
7. **F (21-23)**: `python -m venv $TMP/venv`; seguir Sección 7 literal; `pip install -r requirements.txt`, `pip check`; `streamlit run app.py --server.headless true` en background, revisar log (errores/deprecaciones) y `curl localhost`; leer `@st.cache_resource` (archivo:línea); prueba funcional del panel y del agente usando `streamlit.testing.v1.AppTest` o scripts (copiar `.env` al clon sin mostrarlo, máx. 4 llamadas: saludo, reseña completa, pregunta de memoria, y reinicio sin llamada). Si no hay navegador, marcar ⚠️ lo que no se pueda ver en UI.
8. **G (24-25)**: buscar apellidos/nombres completos en el README; si no están, preguntar al usuario (AskUserQuestion). Redactar borrador del correo (Para jscely@uniboyaca.edu.co, Asunto `TU4-U5 - Apellido1 Apellido2`, cuerpo con integrantes, Ruta 3 y URL del repo) sin enviar. Veredicto: NO ENVIAR si hay algún ❌, con lista por gravedad.
9. **Reporte final**: tabla "Cumplimiento Tarea 19" (# | Requisito | Estado | Evidencia archivo:línea/celda/comando), totales ✅/⚠️/❌, hash verificado, pendientes del usuario (video, textos "ESCRIBE AQUÍ", capturas Vibe Coding, apellidos), y comandos ejecutados. No marcar ✅ sin ejecución.

## Fase 2 (solo con "corrige")
Proponer lista de correcciones mecánicas (borrar copias sueltas sin referencias, rutas/mayúsculas de imágenes, archivos sobrantes, requirements.txt) y esperar aprobación; aplicar en un único commit en main + `git push origin main` (sin force); repetir Fase 1 sobre un clon nuevo con el nuevo hash. No tocar texto del equipo, notebooks ni pesos.

## Verificación del plan
Se considera completa cuando cada uno de los 25 requisitos tiene fila con evidencia ejecutada sobre el clon, y el reporte incluye el hash `origin/main` verificado.
