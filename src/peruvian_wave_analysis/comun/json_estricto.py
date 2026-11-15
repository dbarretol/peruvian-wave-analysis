"""JSON estándar para todo lo que se publica.

``json.dumps`` escribe los float no finitos como ``NaN`` o ``Infinity``, que no son JSON estándar. Aquí:

- ``finitos(obj)`` copia la estructura con los float no finitos como ``None`` (``null``). Quien publica un
  ``null`` pone su causa al lado.
- ``dumps(obj)`` serializa con ``allow_nan=False`` y sin convertir en texto tipos desconocidos: un tipo no
  serializable es un error.
- ``leer(ruta)`` rechaza ``NaN``, ``Infinity`` y ``-Infinity``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _por_defecto(obj: Any) -> Any:
    """Solo conversiones sin pérdida; cualquier otro tipo es un error."""
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Tipo no serializable en JSON: {type(obj).__name__}")


def finitos(obj: Any) -> Any:
    """Copia con los float no finitos como None; numpy pasa a tipos de Python; tuplas a listas."""
    if isinstance(obj, dict):
        return {k: finitos(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finitos(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [finitos(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dumps(obj: Any, *, indent: int | None = 2, ordenar: bool = True) -> str:
    """JSON estándar, UTF-8 sin escapar, terminado en salto de línea."""
    return json.dumps(finitos(obj), ensure_ascii=False, indent=indent, sort_keys=ordenar,
                      default=_por_defecto, allow_nan=False) + "\n"


def escribir(ruta: Path, obj: Any, **kwargs: Any) -> Path:
    ruta = Path(ruta)
    ruta.write_text(dumps(obj, **kwargs), encoding="utf-8", newline="\n")
    return ruta


def _constante(nombre: str) -> float:
    raise ValueError(f"JSON no estándar: aparece «{nombre}»")


def loads(texto: str) -> Any:
    return json.loads(texto, parse_constant=_constante)


def leer(ruta: Path) -> Any:
    return loads(Path(ruta).read_text(encoding="utf-8"))
