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


def faltas(
    manuales: pd.DataFrame,
    maximos: Iterable[pd.Timestamp],
    resoluciones: Iterable[Resolucion],
    maximos_dudosos: Iterable[pd.Timestamp] = (),
) -> list[str]:
    """Motivos de gate fallido de ``datos-serie`` (G4, G5; F03-K-02).

    ``manuales``: candidatos que exigen resolución manual, con columnas ``tiempo`` y ``regla``.
    ``maximos_dudosos``: máximos que coinciden con un candidato resuelto como dudoso por clase (§16): su resolución
    debe llevar «dudoso» en el motivo.

    - Todo candidato manual y todo máximo tiene resolución.
    - **Todo** ``descartar`` cuyo tiempo sea uno de los máximos exige ``confirmado_por_usuario``, venga de la entrada
      que venga.
    - Una entrada que no corresponde a un candidato manual ni a un máximo es huérfana y se rechaza.
    - Dos entradas del mismo tiempo con decisiones distintas son una contradicción y se rechazan.
    """
    res = list(resoluciones)
    idx = indice(res)
    maximos = {pd.Timestamp(t) for t in maximos}
    claves_manuales = {
        (pd.Timestamp(t), str(r)) for t, r in zip(pd.to_datetime(manuales["tiempo"]), manuales["regla"], strict=True)
    }
    validas = claves_manuales | {(t, REGLA_MAXIMO) for t in maximos}
    out = []
    for t, regla in sorted(claves_manuales):
        if (t, regla) not in idx:
            out.append(f"Candidato sin resolución: {t} ({regla})")
    for t in sorted(maximos):
        if (t, REGLA_MAXIMO) not in idx:
            out.append(f"Máximo sin resolución: {t}")
    decisiones: dict[pd.Timestamp, set[str]] = {}
    for r in res:
        if (r.tiempo, r.regla) not in validas:
            out.append(f"Resolución huérfana (no es candidato manual ni máximo): {r.tiempo} ({r.regla})")
        if r.decision == "descartar" and r.tiempo in maximos and not r.confirmado_por_usuario:
            out.append(f"Máximo descartado sin confirmación del usuario: {r.tiempo} ({r.regla})")
        decisiones.setdefault(r.tiempo, set()).add(r.decision)
    for t, d in sorted(decisiones.items()):
        if len(d) > 1:
            out.append(f"Resoluciones contradictorias para {t}: {sorted(d)}")
    for t in sorted({pd.Timestamp(x) for x in maximos_dudosos}):
        r = idx.get((t, REGLA_MAXIMO))
        if r is not None and "dudoso" not in r.motivo.lower():
            out.append(f"Máximo que coincide con un candidato dudoso sin «dudoso» en el motivo: {t}")
    return out
