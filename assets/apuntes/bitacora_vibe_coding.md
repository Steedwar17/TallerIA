# Bitácora de Vibe Coding — Tarea 9

## Episodio 1 — 2026-11-17 — Tarea 9 (src/inferencia.py)
- **Qué propuse:** en `predecir_sentimiento`, llamar `modelo.predict(np.array([[limpio]], dtype=object), verbose=0)`.
- **Por qué estaba mal:** Keras 3 rechaza arreglos NumPy `dtype=object` (`ValueError: Invalid dtype: object`); `src.prueba_inferencia` falló en la primera llamada. Error de API/mala práctica con entradas string.
- **Corrección:** `modelo(tf.constant([[limpio]]), training=False)`. Verificado ejecutando `src.prueba_inferencia`.
- **Captura:** pendiente (la toma Steedwar en `assets/vibe_coding/`).

## Decisión de plan (no es error): Plan A vs B
- Plan A (modelo extremo a extremo con `TextVectorization(standardize=None)`) quedó vigente: diferencia máxima 0.0 en proceso nuevo (notebook sección 5, celda 62). El Plan B no fue necesario.
