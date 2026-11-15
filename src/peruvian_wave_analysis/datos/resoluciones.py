"""Resoluciones manuales versionadas (contrato de la F3, §3.9 y §3.10).

Archivo YAML con una lista de entradas::

    - tiempo: "2016-07-03 12:00"   # UTC
      regla: maximo                 # o la regla del candidato (detector, dipolo, limite_VHM0, ...)
      decision: mantener            # o descartar
      motivo: "..."
      fecha: 2026-11-16
      autor: "..."
      confirmado_por_usuario: 2026-11-16   # obligatorio para descartar uno de los 20 máximos

El archivo entra en el manifiesto con su hash: si cambia, la etapa ``datos-serie`` queda desactualizada.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

CAMPOS = ("tiempo", "regla", "decision", "motivo", "fecha", "autor")
DECISIONES = ("descartar", "mantener")
REGLA_MAXIMO = "maximo"


class ErrorResoluciones(Exception):
    """El archivo de resoluciones no tiene la forma exigida."""


@dataclass(frozen=True)
class Resolucion:
    tiempo: pd.Timestamp
    regla: str
    decision: str
    motivo: str
    fecha: str
    autor: str
    confirmado_por_usuario: str | None = None


def leer(ruta: Path | None) -> list[Resolucion]:
    """Lee y valida el archivo. Sin archivo, ninguna resolución."""
    if ruta is None or not Path(ruta).exists():
        return []
    datos = yaml.safe_load(Path(ruta).read_text(encoding="utf-8")) or []
    if not isinstance(datos, list):
        raise ErrorResoluciones("El archivo de resoluciones debe ser una lista de entradas")
    out, vistos = [], set()
    for i, e in enumerate(datos):
        if not isinstance(e, dict):
            raise ErrorResoluciones(f"Entrada {i}: no es un diccionario")
        faltan = [c for c in CAMPOS if e.get(c) in (None, "")]
        if faltan:
            raise ErrorResoluciones(f"Entrada {i}: faltan {faltan}")
        if e["decision"] not in DECISIONES:
            raise ErrorResoluciones(f"Entrada {i}: decision {e['decision']!r} no es {DECISIONES}")
        r = Resolucion(
            pd.Timestamp(str(e["tiempo"])),
            str(e["regla"]),
            str(e["decision"]),
            str(e["motivo"]),
            str(e["fecha"]),
            str(e["autor"]),
            None if e.get("confirmado_por_usuario") in (None, "") else str(e["confirmado_por_usuario"]),
        )
        clave = (r.tiempo, r.regla)
        if clave in vistos:
            raise ErrorResoluciones(f"Entrada {i}: resolución repetida para {clave}")
        vistos.add(clave)
        out.append(r)
    return out


def indice(resoluciones: Iterable[Resolucion]) -> dict[tuple[pd.Timestamp, str], Resolucion]:
    return {(r.tiempo, r.regla): r for r in resoluciones}


def faltas(manuales: pd.DataFrame, maximos: Iterable[pd.Timestamp], resoluciones: Iterable[Resolucion]) -> list[str]:
    """Motivos de gate fallido de ``datos-serie`` (G4, G5).

    ``manuales``: candidatos que exigen resolución manual, con columnas ``tiempo`` y ``regla``.
    """
    idx = indice(resoluciones)
    out = []
    for t, regla in zip(pd.to_datetime(manuales["tiempo"]), manuales["regla"], strict=True):
        if (pd.Timestamp(t), str(regla)) not in idx:
            out.append(f"Candidato sin resolución: {t} ({regla})")
    for t in maximos:
        r = idx.get((pd.Timestamp(t), REGLA_MAXIMO))
        if r is None:
            out.append(f"Máximo sin resolución: {t}")
        elif r.decision == "descartar" and not r.confirmado_por_usuario:
            out.append(f"Máximo descartado sin confirmación del usuario: {t}")
    return out
