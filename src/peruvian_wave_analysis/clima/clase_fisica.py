"""Clasificación física por particiones de WAVERYS (contrato de la F6, §3.2).

Fracción de energía de cada partición sobre la **suma de las particiones** (no sobre Hs², porque la F3 publicó
discrepancias de energía):

    r_p = Hs_p² / (Hs_WW² + Hs_SW1² + Hs_SW2²)

Una partición ausente (marca ``particion_ausente_<p>`` de la F3) o con altura NaN aporta 0. Un registro sin ninguna
partición con energía queda **sin clase** y se cuenta aparte (G2).

Clases por r_WW, con cortes (1/3, 2/3) principales y (0.2, 0.8) como sensibilidad descriptiva:

- swell: r_WW < corte inferior;
- mixto: corte inferior ≤ r_WW ≤ corte superior;
- mar de viento: r_WW > corte superior.

Solo energía, no período: la clase no usa ``VTM01_WW``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

PARTICIONES = {"WW": "VHM0_WW", "SW1": "VHM0_SW1", "SW2": "VHM0_SW2"}
CORTES = (1 / 3, 2 / 3)
CORTES_SENSIBILIDAD = (0.2, 0.8)
SWELL, MIXTO, VIENTO, SIN_CLASE = "swell", "mixto", "mar de viento", "sin particiones"
CLASES = (SWELL, MIXTO, VIENTO)
TOL_BORDE = 1e-9


def fracciones(df: pd.DataFrame, ausente: Mapping[str, pd.Series] | None = None) -> pd.DataFrame:
    """``r_WW``, ``r_SW1``, ``r_SW2`` (NaN si ninguna partición tiene energía) y ``energia_particiones``."""
    e = {}
    for p, col in PARTICIONES.items():
        h = df[col].astype(float).fillna(0.0)
        if ausente is not None and p in ausente:
            h = h.where(~ausente[p].astype(bool), 0.0)
        e[p] = h**2
    total = sum(e.values())
    out = pd.DataFrame(index=df.index)
    for p in PARTICIONES:
        out[f"r_{p}"] = (e[p] / total).where(total > 0)
    out["energia_particiones"] = total
    return out


def clasificar(r_ww: pd.Series, cortes: tuple[float, float] = CORTES) -> pd.Series:
    """Clase por r_WW; los bordes caen en «mixto» (contrato §3.2: 1/3 ≤ r_WW ≤ 2/3), con tolerancia ``TOL_BORDE``
    para que un r_WW igual al corte salvo redondeo no cambie de clase."""
    lo, hi = cortes[0] - TOL_BORDE, cortes[1] + TOL_BORDE
    r = r_ww.to_numpy(dtype=float)
    clase = np.full(r.size, SIN_CLASE, dtype=object)
    ok = np.isfinite(r)
    clase[ok & (r < lo)] = SWELL
    clase[ok & (r >= lo) & (r <= hi)] = MIXTO
    clase[ok & (r > hi)] = VIENTO
    return pd.Series(clase, index=r_ww.index, name="clase")


def clase_fisica(df: pd.DataFrame, ausente: Mapping[str, pd.Series] | None = None) -> pd.DataFrame:
    """Fracciones y clase con los dos juegos de cortes (``clase`` y ``clase_sens``)."""
    out = fracciones(df, ausente)
    out["clase"] = clasificar(out["r_WW"], CORTES)
    out["clase_sens"] = clasificar(out["r_WW"], CORTES_SENSIBILIDAD)
    return out


def comprobar_g2(cf: pd.DataFrame) -> dict:
    """G2: fracciones en [0, 1] y que suman 1; las clases más «sin particiones» suman la población."""
    r = cf[[f"r_{p}" for p in PARTICIONES]]
    con = r.dropna(how="all")
    en_rango = bool(((con >= -1e-12) & (con <= 1 + 1e-12)).all().all())
    suma_uno = bool(np.allclose(con.sum(axis=1), 1.0)) if len(con) else True
    conteo = cf["clase"].value_counts().to_dict()
    return {
        "fracciones_en_rango": en_rango,
        "fracciones_suman_uno": suma_uno,
        "conteo_clases": {c: int(conteo.get(c, 0)) for c in (*CLASES, SIN_CLASE)},
        "suman_la_poblacion": int(sum(conteo.values())) == len(cf),
        "cumple": en_rango and suma_uno and int(sum(conteo.values())) == len(cf),
    }


def incoherencia_fisica(tp_mediano: Mapping[str, float]) -> dict[str, bool]:
    """Marca descriptiva (§3.2): «mar de viento» con Tp mediano > 10 s o «swell» con Tp mediano < 8 s."""
    return {
        VIENTO: bool(tp_mediano.get(VIENTO, np.nan) > 10.0),
        SWELL: bool(tp_mediano.get(SWELL, np.nan) < 8.0),
    }
