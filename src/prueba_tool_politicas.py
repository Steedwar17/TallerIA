"""
src/prueba_tool_politicas.py — Pruebas unitarias de la @tool #2 consultar_politica_triaje.
Verifica: asserts de limite duro (600 chars), claves compactas, precision en la respuesta.
"""
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.tool_politicas import consultar_politica_triaje


def test_tool_politicas():
    print("=" * 60)
    print("PRUEBAS DE LA @TOOL #2 (consultar_politica_triaje)")
    print("=" * 60)

    casos = [
        ("Sentimiento negativo explicito", {"consulta": "Protocolo para queja", "sentimiento": "negativo", "confianza": 0.95}, "POL_NEG"),
        ("Sentimiento positivo explicito", {"consulta": "Protocolo para felicitacion", "sentimiento": "positivo", "confianza": 0.90}, "POL_POS"),
        ("Baja confianza (<0.70)", {"consulta": "Resena dudosa", "sentimiento": "positivo", "confianza": 0.55}, "POL_REV"),
        ("Sentimiento inferido por consulta", {"consulta": "¿Cual es la politica para resenas negativas?", "sentimiento": None, "confianza": None}, "POL_NEG"),
    ]

    for nombre, kwargs, pol_esperada in casos:
        res_str = consultar_politica_triaje.invoke(kwargs) if hasattr(consultar_politica_triaje, "invoke") else consultar_politica_triaje(**kwargs)
        
        # Validar JSON valido
        data = json.loads(res_str)
        longitud = len(res_str)
        print(f"[{nombre}]")
        print(f"  Salida: {res_str}")
        print(f"  Longitud: {longitud} chars (limite duro <= 600)")
        
        # Asserts requeridos
        assert longitud <= 600, f"Error: salida excede 600 chars ({longitud})"
        assert data["pol_id"] == pol_esperada, f"Error: esperaba {pol_esperada}, obtuve {data['pol_id']}"
        assert set(data.keys()) == {"pol_id", "prioridad", "accion", "pauta"}, f"Claves no coinciden: {data.keys()}"
        print(f"  Validacion: PASS (pol_id={data['pol_id']}, length={longitud})\n")

    print("TODAS LAS PRUEBAS DE TOOL_POLITICAS PASARON EXITOSAMENTE.")


if __name__ == "__main__":
    test_tool_politicas()
