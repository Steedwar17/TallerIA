"""Pruebas de la tarea 9. Ejecutar desde la raíz del repo: .venv/bin/python -m src.prueba_inferencia"""
import re
import time
from pathlib import Path

from src.inferencia import RAIZ_REPO, cargar_modelo, predecir_sentimiento
from src.tool_sentimiento import clasificar_sentimiento_resena


def leer_casos_fallo() -> list:
    """Casos de fallo real (texto + p_positivo) escritos por la tarea 7."""
    for nombre in ("assets/apuntes/fallo_real.md", "assets/apuntes_fallo_tarea7.md"):
        ruta = RAIZ_REPO / nombre
        if ruta.is_file():
            break
    else:
        raise FileNotFoundError("No existe fallo_real.md (tarea 7)")
    md = ruta.read_text(encoding="utf-8")
    p = float(re.search(r"Probabilidad de clase positiva:\s*([0-9.]+)", md).group(1))
    texto = re.search(r"## Texto completo[^\n]*\n(.*?)\n## ", md, re.S).group(1).strip()
    print(f"[fuente de casos] {ruta.relative_to(RAIZ_REPO)} -> {1} caso(s)")
    return [{"texto": texto, "p_positivo": p}]


def main():
    casos = leer_casos_fallo()

    # (d) latencias + (c) caché: primera llamada (carga el modelo) y segunda
    t0 = time.perf_counter()
    r1 = predecir_sentimiento(casos[0]["texto"])
    t1 = time.perf_counter()
    hits_antes = cargar_modelo.cache_info().hits
    r2 = predecir_sentimiento(casos[0]["texto"])
    t2 = time.perf_counter()
    hits_despues = cargar_modelo.cache_info().hits
    print(f"(d) latencia 1a llamada (incluye carga): {(t1 - t0) * 1000:.1f} ms")
    print(f"(d) latencia 2a llamada               : {(t2 - t1) * 1000:.1f} ms")
    print("cache_info:", cargar_modelo.cache_info())
    assert hits_despues > hits_antes, "la segunda llamada recargó el modelo"
    print(f"(c) OK: hits {hits_antes} -> {hits_despues}")

    # (a) claves y tipos
    assert set(r1) == {"sentimiento", "confianza", "p_positivo", "baja_confianza"}
    assert r1["sentimiento"] in ("positivo", "negativo")
    assert isinstance(r1["confianza"], float) and isinstance(r1["p_positivo"], float)
    assert isinstance(r1["baja_confianza"], bool)
    assert r1 == r2
    for v in (r1["confianza"], r1["p_positivo"]):
        assert v == round(v, 4)
    assert r1["confianza"] == round(max(r1["p_positivo"], 1 - r1["p_positivo"]), 4)
    print("(a) OK:", r1)

    # (b) casos de fallo_real.md
    for c in casos:
        r = predecir_sentimiento(c["texto"])
        assert abs(r["p_positivo"] - c["p_positivo"]) < 1e-4, (r, c["p_positivo"])
        print(f"(b) OK: p_positivo={r['p_positivo']} vs esperado {c['p_positivo']}")

    # Entradas inválidas y texto largo
    for malo in ("", "   \n\t", None, 123):
        try:
            predecir_sentimiento(malo)
            raise AssertionError(f"debió lanzar ValueError con {malo!r}")
        except ValueError as e:
            print(f"ValueError OK para {malo!r}: {e}")
    largo = predecir_sentimiento("great movie " * 5000)
    print("Texto largo (10000 palabras) OK:", largo)

    # Requisito 10: la herramienta
    tool = clasificar_sentimiento_resena
    print("tool.name       :", tool.name)
    print("tool.description:", tool.description)
    print("tool.args       :", tool.args)
    print("palabras del docstring:", len(tool.description.split()))
    positiva = "This movie was absolutely wonderful, a brilliant story with superb acting. I loved every minute."
    negativa = "Terrible film. The plot was boring, the acting was awful and I wasted my money."
    for nombre, t in (("positiva", positiva), ("negativa", negativa), ("vacía", "")):
        s = tool.invoke({"texto": t})
        print(f"tool.invoke ({nombre}) -> {s}  [{len(s)} caracteres]")
        assert len(s) <= 120 and "\n" not in s
    print("\nTODAS LAS PRUEBAS PASARON")


if __name__ == "__main__":
    main()
