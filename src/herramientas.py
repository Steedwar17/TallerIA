"""
src/herramientas.py — Proveedor de herramientas del agente de triaje (Unidad 5).
Contrato: obtener_tools() -> list
"""
from typing import List
from langchain_core.tools import BaseTool

from src.tool_politicas import consultar_politica_triaje
from src.agente import clasificar_sentimiento_resena


def obtener_tools() -> List[BaseTool]:
    """Retorna la lista oficial de herramientas para el agente de triaje.

    - @tool #1: clasificar_sentimiento_resena (actualmente provisional Tarea 8;
      se reemplaza por el modelo real en Tarea 10).
    - @tool #2: consultar_politica_triaje (RAG de políticas de triaje).
    """
    return [clasificar_sentimiento_resena, consultar_politica_triaje]
