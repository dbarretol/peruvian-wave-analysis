"""Picos provisionales de la F6 (contrato §3.4), solo para contar y describir.

- **u_tent:** P99 de Hs de los estados válidos de cada rama, con peso mensual igual (como la F4,
  ``homogeneidad.cuantiles``). Sensibilidades P98 y P99.5, descriptivas.
- **Separación:** ``decluster_rachas`` sobre el eje completo, con separación mínima = max(τ̂_e, 48 h) redondeada al
  paso de 3 h. τ̂_e es el de E0 sin corregir: el IET es común a todas las ramas (B-S-03).
- **Huecos:** un hueco no parte una tormenta (``decluster_rachas`` cuenta el NaN como tiempo sin excedencia).
- **Estratos:** B por el mes del pico (invierno mayo–octubre, verano noviembre–abril); C por el arco de `VPED` que
  fija la regla de bimodalidad (§6.1).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..extremos.decluster import decluster_rachas
from ..homogeneidad.cuantiles import cuantiles_mensuales
from . import clase_fisica as CF
from . import descriptivos as DS
from .insumos import marcar_dudosos

PASO_H = 3.0
SEPARACION_MIN_H = 48.0
Q_TENT = 0.99
Q_SENSIBILIDAD = (0.98, 0.995)
INVIERNO = (5, 6, 7, 8, 9, 10)
B_INVIERNO, B_VERANO = "invierno", "verano"
C_MAYOR, C_MENOR = "sector_mayor", "sector_menor"
COLUMNAS_PICO = ("VTPK", "VPED", "VTM02", *CF.PARTICIONES.values(), "VTM01_WW", "VTM01_SW1", "VTM01_SW2",
                 "VMDR_WW", "VMDR_SW1", "VMDR_SW2", "marca_tp_saturado",
                 *(f"marca_particion_ausente_{q}" for q in CF.PARTICIONES))  # fmt: skip


def u_tentativo(hs: pd.Series, q: float = Q_TENT) -> float:
    return float(cuantiles_mensuales(hs, q)[0])


def separacion(tau_e_h: float) -> float:
    """max(τ̂_e, 48 h) redondeada al paso de 3 h."""
    base = SEPARACION_MIN_H if not np.isfinite(tau_e_h) else max(float(tau_e_h), SEPARACION_MIN_H)
    return float(np.round(base / PASO_H) * PASO_H)


def picos(hs: pd.Series, serie: pd.DataFrame, u: float, sep_h: float, dudosos=None) -> pd.DataFrame:
    """Picos de la rama ``hs`` con los descriptores de ``serie`` (E0) en el instante del pico."""
    cols = [c for c in COLUMNAS_PICO if c in serie.columns]
    df = serie.loc[hs.index, cols].copy()
    df.insert(0, "Hs", hs.to_numpy(float))
    p = decluster_rachas(df, u, sep_h, PASO_H, columna="Hs")
    p["mes"] = p.index.month
    p["estrato_B"] = np.where(np.isin(p["mes"], INVIERNO), B_INVIERNO, B_VERANO)
    p["sector"] = DS.sector(p["VPED"].to_numpy(float))
    if all(c in p.columns for c in CF.PARTICIONES.values()):
        p["clase"] = CF.clase_fisica(p, ausentes(p))["clase"].to_numpy()
    p["dudoso_f3"] = False if dudosos is None else marcar_dudosos(p, pd.DatetimeIndex(dudosos))
    return p


def ausentes(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Marcas ``marca_particion_ausente_<p>`` de la F3 que estén en ``df``."""
    return {q: df[f"marca_particion_ausente_{q}"] for q in CF.PARTICIONES if f"marca_particion_ausente_{q}" in df}


def estrato_C(vped, arco: tuple[float, float] | None) -> np.ndarray:
    """Sector menor (el arco de la segunda moda) o mayor; todo «mayor» si no hay arco."""
    v = np.asarray(vped, dtype=float)
    if arco is None:
        return np.full(v.size, C_MAYOR, dtype=object)
    return np.where(DS.en_arco(v, *arco), C_MENOR, C_MAYOR).astype(object)


def conteos(etiquetas, nombres) -> dict[str, int]:
    e = pd.Series(np.asarray(etiquetas, dtype=object))
    return {n: int((e == n).sum()) for n in nombres}


def fraccion_chica(c: Mapping[str, int]) -> tuple[str, float]:
    """(estrato chico, su fracción de los picos)."""
    total = sum(c.values())
    chico = min(c, key=lambda k: (c[k], k))
    return chico, (c[chico] / total if total else float("nan"))
