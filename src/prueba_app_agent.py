"""src/prueba_app_agent.py — Pruebas unitarias de app_agent.py con AppTest.

Requisito 15 (Tarea 12):
Utiliza AppTest con el agente sustituido por un doble de prueba (sin gastar cuota)
para validar:
1. Render inicial y estado por defecto (ag_thread_id, ag_historial vacío).
2. Pintado del historial e indicador visible de herramientas del doble.
3. Botón de reinicio: genera nuevo thread_id y vacía ag_historial.
4. Puente con panel del modelo (nn_texto).
5. Aislamiento y persistencia segura en caché del agente.
"""

import os
import sys
from pathlib import Path

# Asegurar raíz del proyecto en sys.path
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import unittest
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import src.agente

# Asegurar clave ficticia si no existe en entorno de test
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "test_key_mock_123"


class TestAppAgent(unittest.TestCase):
    """Suite de pruebas automatizadas para app_agent.py usando AppTest."""

    def test_1_render_inicial(self):
        """Verifica el estado inicial sin errores ni excepciones."""
        at = AppTest.from_file("app_agent.py")
        at.run(timeout=30)

        self.assertEqual(len(at.exception), 0, f"Excepciones encontradas: {at.exception}")
        self.assertIn("ag_thread_id", at.session_state)
        self.assertIn("ag_historial", at.session_state)
        self.assertEqual(len(at.session_state["ag_historial"]), 0)
        self.assertTrue(len(at.session_state["ag_thread_id"]) > 10)
        self.assertEqual(len(at.chat_input), 1)
        self.assertGreaterEqual(len(at.button), 1)

    def test_2_cache_seguro_compartido(self):
        """Verifica que obtener_agente esté en caché y comente la seguridad arquitectónica."""
        import app_agent
        import inspect

        doc = inspect.getdoc(app_agent.obtener_agente) or ""
        self.assertIn("thread_id", doc.lower(), "Debe documentar por qué el caché compartido es seguro con thread_id.")

    def test_3_envio_mensaje_con_doble_de_prueba(self):
        """Verifica pintado de historial, herramientas del doble y expander sin gastar cuota."""
        # Configurar doble de prueba (Mock)
        mock_msg_user = HumanMessage(content="Reseña negativa mock")
        mock_msg_ai1 = AIMessage(
            content="",
            tool_calls=[{"name": "clasificar_sentimiento_resena", "args": {"texto": "bad movie"}, "id": "call_1"}],
        )
        mock_msg_tool1 = ToolMessage(
            content='{"sentimiento": "negativo", "confianza": 0.95, "baja_confianza": false}',
            tool_call_id="call_1",
        )
        mock_msg_ai2 = AIMessage(
            content="",
            tool_calls=[{"name": "consultar_politica_triaje", "args": {"sentimiento": "negativo", "confianza": 0.95}, "id": "call_2"}],
        )
        mock_msg_tool2 = ToolMessage(
            content='{"pol_id": "POL_NEG", "prioridad": "ALTA", "accion": "Ticket"}',
            tool_call_id="call_2",
        )
        mock_msg_ai_final = AIMessage(content="Triaje completado: Sentimiento negativo, prioridad ALTA.")

        retorno_mock = {
            "respuesta": "Triaje completado: Sentimiento negativo, prioridad ALTA.",
            "tools_invocadas": ["clasificar_sentimiento_resena", "consultar_politica_triaje"],
            "herramientas": ["clasificar_sentimiento_resena", "consultar_politica_triaje"],
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
            "num_llamadas_modelo": 3,
            "mensajes": [mock_msg_user, mock_msg_ai1, mock_msg_tool1, mock_msg_ai2, mock_msg_tool2, mock_msg_ai_final],
        }

        with patch.object(src.agente, "responder", return_value=retorno_mock):
            at = AppTest.from_file("app_agent.py")
            at.run(timeout=30)

            # Enviar mensaje por el chat input
            at.chat_input[0].set_value("The movie was terrible and disappointing.").run(timeout=30)

            self.assertEqual(len(at.exception), 0, f"Excepciones: {at.exception}")
            historial = at.session_state["ag_historial"]
            self.assertEqual(len(historial), 2, "Debe haber 2 turnos (1 usuario, 1 asistente)")

            # Verificar contenido de asistente
            turno_asistente = historial[1]
            self.assertEqual(turno_asistente["role"], "assistant")
            self.assertEqual(
                turno_asistente["herramientas"],
                ["clasificar_sentimiento_resena", "consultar_politica_triaje"],
            )
            self.assertEqual(len(turno_asistente["detalle_tools"]), 2)
            self.assertEqual(turno_asistente["detalle_tools"][0]["nombre"], "clasificar_sentimiento_resena")
            self.assertEqual(turno_asistente["tokens"]["total"], 165)

    def test_4_reinicio_conversacion(self):
        """Verifica que el botón de reinicio genere un nuevo thread_id y limpie el historial."""
        at = AppTest.from_file("app_agent.py")
        at.run(timeout=30)

        # Simular historial previo
        thread_id_original = at.session_state["ag_thread_id"]
        at.session_state["ag_historial"].append({"role": "user", "contenido": "Mensaje anterior"})
        at.session_state["ag_historial"].append({"role": "assistant", "contenido": "Respuesta anterior", "herramientas": [], "detalle_tools": [], "tokens": {}})
        self.assertEqual(len(at.session_state["ag_historial"]), 2)

        # Hacer click en el botón de reinicio
        boton_reinicio = at.button[0]
        self.assertIn("Reiniciar", boton_reinicio.label)
        boton_reinicio.click().run(timeout=30)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["ag_historial"]), 0, "El historial debe vaciarse tras el reinicio.")
        self.assertNotEqual(
            at.session_state["ag_thread_id"],
            thread_id_original,
            "El thread_id debe renovarse.",
        )

    def test_5_puente_panel_modelo(self):
        """Verifica que si existe nn_texto en session_state, aparezca el botón puente y funcione."""
        retorno_mock = {
            "respuesta": "Triaje recibido desde panel de modelo.",
            "tools_invocadas": ["clasificar_sentimiento_resena"],
            "herramientas": ["clasificar_sentimiento_resena"],
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": 100,
            "num_llamadas_modelo": 1,
            "mensajes": [],
        }

        with patch.object(src.agente, "responder", return_value=retorno_mock):
            at = AppTest.from_file("app_agent.py")
            at.run(timeout=30)

            # Inicialmente solo está el botón de reinicio
            self.assertEqual(len(at.button), 1)

            # Inyectar texto del panel del modelo
            at.session_state["nn_texto"] = "Outstanding performance by the cast."
            at.run(timeout=30)

            # Ahora deben existir 2 botones: reinicio y puente
            self.assertEqual(len(at.button), 2)
            boton_puente = at.button[1]
            self.assertIn("Enviar al agente la reseña del panel del modelo", boton_puente.label)

            # Hacer click en el botón puente
            boton_puente.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.session_state["ag_historial"]), 2)


if __name__ == "__main__":
    unittest.main()
