"""Prueba del panel app_nn.py con streamlit.testing.v1.AppTest.

Ejecutar desde la raíz del repo:  .venv/bin/python -m src.prueba_app_nn
"""
import logging
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.inferencia import cargar_modelo

RAIZ = Path(__file__).resolve().parent.parent
APP = str(RAIZ / "app_nn.py")


class Contador(logging.Handler):
    def __init__(self):
        super().__init__()
        self.cargas = []

    def emit(self, record):
        if "Modelo cargado de disco" in record.getMessage():
            self.cargas.append(record.getMessage())


def main():
    # Empezar en frío: sin caché de Streamlit ni del modelo de src.inferencia
    st.cache_resource.clear()
    cargar_modelo.cache_clear()
    contador = Contador()
    log = logging.getLogger("nn_app")
    log.setLevel(logging.INFO)
    log.addHandler(contador)

    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    assert not contador.cargas, "el modelo no debe cargarse antes de clasificar"

    # 1) Texto válido: métrica y JSON
    texto = "A wonderful film with brilliant acting and a moving story."
    at.text_area(key="nn_entrada").set_value(texto).run()
    at.button(key="nn_btn_clasificar").click().run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 2, "faltan las métricas"
    assert len(at.json) == 1, "falta el JSON de salida cruda"
    r1 = at.session_state["nn_resultado"]
    assert at.session_state["nn_texto"] == texto
    assert set(r1) == {"sentimiento", "confianza", "p_positivo", "baja_confianza"}
    assert any("ms" in c.value for c in at.caption), "falta el tiempo de inferencia"
    print("1ª clasificación:", r1, "|", [c.value for c in at.caption if "ms" in c.value])

    # 2) Segunda clasificación seguida: sigue habiendo un solo mensaje de carga
    at.text_area(key="nn_entrada").set_value("Boring, awful and far too long.").run()
    at.button(key="nn_btn_clasificar").click().run()
    assert not at.exception, at.exception
    print("2ª clasificación:", at.session_state["nn_resultado"], "|", [c.value for c in at.caption if "ms" in c.value])
    assert len(contador.cargas) == 1, f"mensajes de carga: {len(contador.cargas)}"
    print("Mensajes de carga de disco tras 2 clasificaciones:", len(contador.cargas))

    # 3) Texto vacío: aviso y sin llamada al modelo
    at.text_area(key="nn_entrada").set_value("").run()
    at.button(key="nn_btn_clasificar").click().run()
    assert not at.exception, at.exception
    assert any("antes de clasificar" in w.value for w in at.warning), [w.value for w in at.warning]
    assert at.session_state["nn_resultado"] is None
    assert len(at.metric) == 0 and len(at.json) == 0
    assert len(contador.cargas) == 1
    print("Texto vacío: aviso =", [w.value for w in at.warning])

    # 4) Texto largo: aviso de truncado
    from src.config import SEQ_LEN
    at.text_area(key="nn_entrada").set_value("good " * (SEQ_LEN + 5)).run()
    assert any(f"primeras {SEQ_LEN}" in w.value for w in at.warning), [w.value for w in at.warning]
    print("Texto largo: aviso =", [w.value for w in at.warning])

    # 5) Botones de ejemplo: rellenan el área; Caso B viene de fallo_real.md y dispara baja confianza
    at.button(key="nn_btn_pos").click().run()
    assert "wonderful" in at.text_area(key="nn_entrada").value
    at.button(key="nn_btn_neg").click().run()
    assert "terrible" in at.text_area(key="nn_entrada").value
    at.button(key="nn_btn_casob").click().run()
    assert at.text_area(key="nn_entrada").value.startswith("Not bad, but not great either")
    at.button(key="nn_btn_clasificar").click().run()
    assert not at.exception, at.exception
    rb = at.session_state["nn_resultado"]
    print("Caso B:", rb)
    assert rb["baja_confianza"] is True
    assert any("umbral" in w.value and "0.70" in w.value for w in at.warning)
    assert len(contador.cargas) == 1
    print("OK: todas las verificaciones pasaron; at.exception vacío.")


if __name__ == "__main__":
    main()
