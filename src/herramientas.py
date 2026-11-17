"""
src/herramientas.py — Proveedor oficial de herramientas del agente de triaje (Unidad 5).
Contrato del taller: obtener_tools() -> [clasificar_sentimiento_resena, consultar_politica_triaje]
Es la única fuente de herramientas para la aplicación interactiva (app.py) y los notebooks.
"""
from typing import List
from langchain_core.tools import BaseTool

from src.tool_sentimiento import clasificar_sentimiento_resena
from src.tool_politicas import consultar_politica_triaje


def obtener_tools() -> List[BaseTool]:
    """Retorna la lista oficial de herramientas para el agente de triaje.

    - @tool #1: clasificar_sentimiento_resena (inferencia real con red neuronal U4).
    - @tool #2: consultar_politica_triaje (RAG de políticas internas de triaje).
    """
    return [clasificar_sentimiento_resena, consultar_politica_triaje]
