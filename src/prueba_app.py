"""src/prueba_app.py — Pruebas unitarias completas de la aplicación unificada app.py.

Valida mediante streamlit.testing.v1.AppTest:
1. Renderizado general sin excepciones y presencia de ambas pestañas.
2. Inferencia directa de la red neuronal (Unidad 4) con cálculo de certeza y métricas.
3. Chat conversacional del agente con doble de prueba (Unidad 5) e indicador de tools.
4. Reinicio de memoria conversacional (nuevo thread_id y vaciado de historial).
5. Puente funcional: transferencia de reseña desde el panel del modelo hacia el agente.
"""

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

# Asegurar raíz del proyecto en sys.path
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from streamlit.testing.v1 import AppTest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import src.agente

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "test_key_mock_123"


class TestAppUnificada(unittest.TestCase):
    """Suite de pruebas para app.py unificada."""

    def test_1_render_inicial(self):
        """Verifica que app.py inicie limpiamente sin excepciones."""
        at = AppTest.from_file(str(RAIZ / "app.py"))
        at.run(timeout=35)

        self.assertEqual(len(at.exception), 0, f"Excepciones encontradas: {at.exception}")
        self.assertIn("ag_thread_id", at.session_state)
        self.assertIn("ag_historial", at.session_state)
        self.assertEqual(len(at.session_state["ag_historial"]), 0)

    def test_2_panel_modelo_inferencia(self):
        """Verifica la clasificación directa en el panel neuronal (Unidad 4)."""
        at = AppTest.from_file(str(RAIZ / "app.py"))
        at.run(timeout=35)

        # Buscar textarea y botón de clasificar
        self.assertGreaterEqual(len(at.text_area), 1)
        at.text_area[0].set_value("This film was fantastic, brilliantly executed and deeply moving.").run(timeout=35)

        # Buscar botón Clasificar
        btn_clasificar = None
        for b in at.button:
            if "Clasificar" in b.label:
                btn_clasificar = b
                break
        self.assertIsNotNone(btn_clasificar, "Botón Clasificar no encontrado")
        btn_clasificar.click().run(timeout=35)

        self.assertEqual(len(at.exception), 0)
        self.assertIn("nn_resultado", at.session_state)
        res = at.session_state["nn_resultado"]
        self.assertIsNotNone(res)
        self.assertIn("sentimiento", res)
        self.assertIn("confianza", res)
        self.assertEqual(res["sentimiento"], "positivo")
        self.assertGreater(res["confianza"], 0.70)
        self.assertEqual(at.session_state["nn_texto"], "This film was fantastic, brilliantly executed and deeply moving.")

    def test_3_panel_agente_chat_mock(self):
        """Verifica el chat del agente con doble de prueba sin consumir cuota LLM."""
        retorno_mock = {
            "respuesta": "Triaje completado con éxito: prioridad ALTA.",
            "tools_invocadas": ["clasificar_sentimiento_resena", "consultar_politica_triaje"],
            "herramientas": ["clasificar_sentimiento_resena", "consultar_politica_triaje"],
            "input_tokens": 150,
            "output_tokens": 50,
            "total_tokens": 200,
            "num_llamadas_modelo": 3,
            "mensajes": [
                HumanMessage(content="Terrible movie"),
                AIMessage(content="", tool_calls=[{"name": "clasificar_sentimiento_resena", "args": {"texto": "bad"}, "id": "c1"}]),
                ToolMessage(content='{"sentimiento": "negativo", "confianza": 0.99, "baja_confianza": false}', tool_call_id="c1"),
                AIMessage(content="", tool_calls=[{"name": "consultar_politica_triaje", "args": {"sentimiento": "negativo"}, "id": "c2"}]),
                ToolMessage(content='{"pol_id": "POL_NEG", "prioridad": "ALTA", "accion": "Ticket"}', tool_call_id="c2"),
                AIMessage(content="Triaje completado con éxito: prioridad ALTA.")
            ],
        }

        with patch.object(src.agente, "responder", return_value=retorno_mock):
            at = AppTest.from_file(str(RAIZ / "app.py"))
            at.run(timeout=35)

            self.assertEqual(len(at.chat_input), 1)
            at.chat_input[0].set_value("The movie was completely boring and horrible.").run(timeout=35)

            self.assertEqual(len(at.exception), 0)
            historial = at.session_state["ag_historial"]
            self.assertEqual(len(historial), 2)
            asistente = historial[1]
            self.assertEqual(asistente["role"], "assistant")
            self.assertEqual(asistente["herramientas"], ["clasificar_sentimiento_resena", "consultar_politica_triaje"])
            self.assertEqual(len(asistente["detalle_tools"]), 2)
            self.assertEqual(asistente["tokens"]["total"], 200)

    def test_4_reinicio_conversacion(self):
        """Verifica que el botón de reinicio regenere thread_id y limpie historial."""
        at = AppTest.from_file(str(RAIZ / "app.py"))
        at.run(timeout=35)

        tid_inicial = at.session_state["ag_thread_id"]
        at.session_state["ag_historial"].append({"role": "user", "contenido": "Texto previo"})
        at.session_state["ag_historial"].append({"role": "assistant", "contenido": "Respuesta previa", "herramientas": [], "detalle_tools": [], "tokens": {}})

        btn_reinicio = None
        for b in at.button:
            if "Reiniciar" in b.label:
                btn_reinicio = b
                break
        self.assertIsNotNone(btn_reinicio, "Botón de reinicio no encontrado")
        btn_reinicio.click().run(timeout=35)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["ag_historial"]), 0)
        self.assertNotEqual(at.session_state["ag_thread_id"], tid_inicial)

    def test_5_puente_modelo_a_agente(self):
        """Verifica que una reseña en nn_texto active el botón puente y despache al agente."""
        retorno_mock = {
            "respuesta": "Triaje recibido del puente.",
            "tools_invocadas": ["clasificar_sentimiento_resena"],
            "herramientas": ["clasificar_sentimiento_resena"],
            "input_tokens": 100,
            "output_tokens": 30,
            "total_tokens": 130,
            "num_llamadas_modelo": 1,
            "mensajes": [],
        }

        with patch.object(src.agente, "responder", return_value=retorno_mock):
            at = AppTest.from_file(str(RAIZ / "app.py"))
            at.run(timeout=35)

            # Inyectar texto proveniente del panel del modelo
            at.session_state["nn_texto"] = "A masterpiece of cinematic storytelling."
            at.run(timeout=35)

            btn_puente = None
            for b in at.button:
                if "Enviar al agente" in b.label:
                    btn_puente = b
                    break
            self.assertIsNotNone(btn_puente, "Botón puente no encontrado cuando existe nn_texto")

            btn_puente.click().run(timeout=35)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.session_state["ag_historial"]), 2)


if __name__ == "__main__":
    unittest.main()
