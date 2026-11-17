"""Limpieza de texto: réplica exacta, en Python puro, de `estandarizar` del notebook (celda 15).

La limpieza va aquí y NO dentro de TextVectorization (que se guarda con standardize=None).
Se imita el comportamiento de TensorFlow: tf.strings.lower solo pasa a minúscula ASCII,
y \\s de RE2 es [\\t\\n\\f\\r ].
"""
import re
import string

_PUNCT_RE = re.compile("[" + re.escape(string.punctuation) + "]")
_SPACES_RE = re.compile(r"[\t\n\f\r ]+")
_MINUS_ASCII = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}


def limpiar_texto(texto: str) -> str:
    """Minúsculas ASCII, '<br />' -> espacio, sin puntuación ASCII, espacios colapsados y recortados."""
    x = texto.translate(_MINUS_ASCII)
    x = x.replace("<br />", " ")
    x = _PUNCT_RE.sub("", x)
    x = _SPACES_RE.sub(" ", x)
    return x.strip(" \t\n\f\r\v")
