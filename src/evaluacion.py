"""src/evaluacion.py — Evaluación del agente: Trayectoria y Resultado (Unidad 5).

Función reutilizable por las tareas 13 y 14. La trayectoria se decide solo con
AIMessage.tool_calls (vía herramientas_invocadas); nunca con el texto de la respuesta.
El resultado lo decide una función determinista (sin LLM juez).
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import ToolMessage

from src.agente import (
    herramientas_invocadas,
    mensajes_del_turno,
    nuevo_thread_id,
    responder,
)
from src.config import RPM_GRATUITO

MAX_REINTENTOS_429 = 2
ESPERA_429_POR_DEFECTO = 60


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _texto(contenido: Any) -> str:
    """Normaliza el contenido de un mensaje (str o lista de bloques) a texto."""
    if isinstance(contenido, list):
        partes = []
        for b in contenido:
            if isinstance(b, dict) and "text" in b:
                partes.append(str(b["text"]))
            elif isinstance(b, str):
                partes.append(b)
        return " ".join(partes)
    return str(contenido or "")


def _es_429(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}"
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def _espera_indicada(exc: Exception) -> float:
    """Extrae la espera indicada por la API ('retry in 12.3s' / retryDelay: '12s')."""
    s = str(exc)
    m = re.search(r"retry in ([\d.]+)\s*s", s, re.I) or re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", s, re.I
    )
    if m:
        return float(m.group(1)) + 1
    return ESPERA_429_POR_DEFECTO


def pausa_entre_casos() -> float:
    """Pausa (segundos) según RPM_GRATUITO de src/config.py; la ejecuta y la retorna."""
    segundos = 60.0 / RPM_GRATUITO + 1
    time.sleep(segundos)
    return segundos


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
def evaluar_caso(
    agente,
    nombre: str,
    entrada: str,
    esperadas: List[str],
    verificar_resultado: Callable[[str, List[ToolMessage]], Tuple[bool, str]],
) -> Dict[str, Any]:
    """Ejecuta un caso con thread_id nuevo y evalúa Trayectoria y Resultado."""
    thread_id = nuevo_thread_id()
    intentos = 0
    while True:
        try:
            salida = responder(agente, entrada, thread_id)
            break
        except Exception as exc:  # noqa: BLE001
            if _es_429(exc) and intentos < MAX_REINTENTOS_429:
                intentos += 1
                espera = _espera_indicada(exc)
                print(f"[429] {nombre}: esperando {espera:.0f}s y repitiendo solo este caso "
                      f"(reintento {intentos}/{MAX_REINTENTOS_429})")
                time.sleep(espera)
                thread_id = nuevo_thread_id()
                continue
            raise

    mensajes = mensajes_del_turno(salida["mensajes"])
    invocadas = herramientas_invocadas(mensajes)
    trayectoria_ok = invocadas == list(esperadas)

    tool_msgs = [m for m in mensajes if isinstance(m, ToolMessage)]
    respuesta = salida["respuesta"]
    resultado_ok, detalle = verificar_resultado(respuesta, tool_msgs)

    return {
        "nombre": nombre,
        "entrada": entrada,
        "esperadas": list(esperadas),
        "invocadas": invocadas,
        "trayectoria_ok": bool(trayectoria_ok),
        "resultado_ok": bool(resultado_ok),
        "detalle_resultado": detalle,
        "respuesta": respuesta,
        "tokens": salida.get("total_tokens", 0),
    }


def imprimir_caso(resultado: Dict[str, Any]) -> None:
    """Imprime un caso evaluado con los rótulos del reporte."""
    cumple = lambda ok: "CUMPLE" if ok else "NO CUMPLE"  # noqa: E731
    print(f"Caso: {resultado['nombre']}")
    print(f"Entrada: {resultado['entrada']}")
    print(f"Herramientas esperadas: {resultado['esperadas']}")
    print(f"Herramientas invocadas: {resultado['invocadas']}")
    print(f"Trayectoria: {cumple(resultado['trayectoria_ok'])}")
    print(f"Resultado: {cumple(resultado['resultado_ok'])} ({resultado['detalle_resultado']})")
    print(f"Respuesta final: {resultado['respuesta']}")
    print(f"Tokens: {resultado['tokens']}")


def guardar_resultados(resultados: List[Dict[str, Any]], ruta: str) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Verificadores deterministas
# --------------------------------------------------------------------------
_PORCENTAJE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
_CONFIANZA_CIFRA = re.compile(
    r"confianz\w*[^\d\n]{0,20}\d|\d[^\n]{0,20}confianz", re.IGNORECASE
)


def verificar_caso_1(respuesta: str, tool_msgs: List[ToolMessage]) -> Tuple[bool, str]:
    """Sin herramientas: respuesta no vacía, sin ToolMessage, sin clasificación inventada."""
    if not (respuesta or "").strip():
        return False, "respuesta vacía"
    if tool_msgs:
        return False, f"se encontraron {len(tool_msgs)} ToolMessage"
    if _PORCENTAJE.search(respuesta):
        return False, "la respuesta contiene un porcentaje"
    if _CONFIANZA_CIFRA.search(respuesta):
        return False, "la respuesta menciona 'confianza' junto a una cifra"
    return True, "respuesta no vacía, sin ToolMessage y sin clasificación inventada"


def verificar_caso_2(respuesta: str, tool_msgs: List[ToolMessage]) -> Tuple[bool, str]:
    """Una herramienta: el pol_id o la acción del ToolMessage aparece en la respuesta."""
    if not tool_msgs:
        return False, "no hay ToolMessage del que extraer la política"
    norm = lambda s: re.sub(r"[\s_\-]+", "", s.lower())  # noqa: E731
    resp_n = norm(respuesta or "")
    candidatos = []
    for m in tool_msgs:
        crudo = _texto(m.content)
        try:
            datos = json.loads(crudo)
        except (json.JSONDecodeError, TypeError):
            datos = {}
        if isinstance(datos, dict):
            for clave in ("pol_id", "accion"):
                if datos.get(clave):
                    candidatos.append((clave, str(datos[clave])))
        else:
            for pid in re.findall(r"POL[_-]\w+", crudo):
                candidatos.append(("pol_id", pid))
    if not candidatos:
        return False, "el ToolMessage no contiene pol_id ni accion legibles"
    for clave, valor in candidatos:
        if norm(valor) and norm(valor) in resp_n:
            return True, f"{clave} '{valor}' del ToolMessage aparece en la respuesta"
    return False, f"ni {candidatos} aparecen en la respuesta"
