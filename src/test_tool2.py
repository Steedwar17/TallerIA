"""
test_tool2.py — Pruebas independientes para la @tool #2 (consultar_politicas_triaje).
Verifica el comportamiento de la herramienta con entradas simuladas sin depender del modelo U4.
"""
import sys
from pathlib import Path

# Agregar directorio raíz al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.tools import consultar_politicas_triaje


def ejecutar_pruebas_simuladas():
    print("=" * 70)
    print("EJECUTANDO PRUEBAS INDEPENDIENTES: @tool #2 (consultar_politicas_triaje)")
    print("=" * 70)

    # 1. Verificación de metadatos de la tool (nombre y docstring)
    nombre = getattr(consultar_politicas_triaje, "name", consultar_politicas_triaje.__name__)
    doc = getattr(consultar_politicas_triaje, "description", consultar_politicas_triaje.__doc__)
    print(f"\n[METADATOS DE LA TOOL]")
    print(f"- Nombre: {nombre}")
    print(f"- Docstring presente: {'SÍ' if doc else 'NO'}")

    # 2. Casos de prueba simulados (categoría y certeza simulada de la U4)
    casos = [
        {
            "caso": "Caso A: Reseña negativa con certeza alta (Steedwar: negativo, 0.96)",
            "categoria": "negativo",
            "consulta": "El usuario reclama por pésima calidad en el guión y dirección.",
            "esperado_id": "POLITICA_NEGATIVO",
        },
        {
            "caso": "Caso B: Reseña ambigua / baja certeza (Steedwar: incierto, 0.52)",
            "categoria": "incierto",
            "consulta": "Opinión mixta, menciona aspectos buenos y malos con dudas.",
            "esperado_id": "POLITICA_INCIERTO",
        },
        {
            "caso": "Caso C: Reseña positiva con certeza alta (Steedwar: positivo, 0.94)",
            "categoria": "positivo",
            "consulta": "El usuario elogia la actuación y recomienda la película.",
            "esperado_id": "POLITICA_POSITIVO",
        },
    ]

    print("\n[RESULTADOS DE EJECUCIÓN CON ENTRADAS SIMULADAS]\n")

    for i, item in enumerate(casos, 1):
        print(f"--- Prueba {i}: {item['caso']} ---")
        print(f"Entrada simulada -> Categoria: '{item['categoria']}' | Detalle: '{item['consulta']}'")
        
        # Invocar la tool (soporta llamada directa de función o .invoke())
        if hasattr(consultar_politicas_triaje, "invoke"):
            salida = consultar_politicas_triaje.invoke({
                "categoria": item["categoria"],
                "consulta": item["consulta"],
            })
        else:
            salida = consultar_politicas_triaje(
                categoria=item["categoria"],
                consulta=item["consulta"],
            )

        print(f"Salida real retornada por la tool:\n{salida}")
        
        # Validación de salida compacta y política esperada
        contiene_esperado = item["esperado_id"] in salida
        longitud_chars = len(salida)
        cumple_tamano = longitud_chars < 260  # salida ultra compacta (< 260 caracteres / ~50 tokens)
        
        print(f"Validación -> Contiene {item['esperado_id']}: {'PASS' if contiene_esperado else 'FAIL'}")
        print(f"Longitud: {longitud_chars} caracteres (~{longitud_chars // 4} tokens) -> {'COMPACTA' if cumple_tamano else 'EXTENSA'}\n")


if __name__ == "__main__":
    ejecutar_pruebas_simuladas()
