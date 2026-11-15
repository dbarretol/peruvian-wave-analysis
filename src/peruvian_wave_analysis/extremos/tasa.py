"""Tasa de eventos por estrato, en **eventos por año**: la unidad que reciben los niveles de retorno.

- Años con dato del estrato s: E_s = Σ_años f_{s,año}, con f la fracción de horas válidas **dentro de la ventana
  del propio estrato** en ese año de bloque (0 a 1). Un hueco en otro estrato no cambia E_s.
- λ_s = eventos del estrato / E_s, con IC exacto de Poisson (Garwood, 1936):
  [χ²_{α/2}(2n)/2, χ²_{1−α/2}(2n+2)/2] / E_s.

Con todos los años completos, Σ_s λ_s es el número medio de eventos por año.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .estratos import Estratificacion

INDICE = ["anio_bloque", "estrato"]


def anios_con_dato(tiempos, validos, paso_horas: float, est: Estratificacion) -> pd.DataFrame:
    """Horas válidas, horas de la ventana y fracción f por (año de bloque, estrato) presente en la rejilla."""
    t = pd.DatetimeIndex(tiempos)
    df = pd.DataFrame({"anio_bloque": est.anio_bloque(t), "estrato": est.etiqueta(t),
                       "horas_validas": np.asarray(validos, dtype=float) * paso_horas})
    out = df.groupby(INDICE, sort=True)["horas_validas"].sum().to_frame()
    out["horas_ventana"] = [est.horas_ventana(int(a), s) for a, s in out.index]
    out["fraccion"] = out["horas_validas"] / out["horas_ventana"]
    if (out["fraccion"] > 1 + 1e-9).any():
        raise ValueError("Más horas válidas que horas de la ventana: la rejilla tiene duplicados o el paso es incorrecto")
    return out


def contar_eventos(tiempos_eventos, bloques: pd.DataFrame, est: Estratificacion) -> pd.DataFrame:
    """Añade ``n_eventos`` a la tabla de ``anios_con_dato``. Un evento fuera de los bloques es un error."""
    t = pd.DatetimeIndex(tiempos_eventos)
    claves = pd.MultiIndex.from_arrays([est.anio_bloque(t), est.etiqueta(t)], names=INDICE)
    fuera = ~claves.isin(bloques.index)
    if fuera.any():
        raise ValueError(f"{int(fuera.sum())} eventos caen fuera de los bloques con dato")
    conteo = pd.Series(1, index=claves).groupby(level=[0, 1]).sum()
    out = bloques.copy()
    out["n_eventos"] = conteo.reindex(out.index, fill_value=0).astype(int)
    return out


def ic_garwood(n: int, anios: float, nivel: float = 0.95) -> tuple[float, float]:
    a = 1.0 - nivel
    lo = 0.0 if n == 0 else stats.chi2.ppf(a / 2, 2 * n) / 2.0
    hi = stats.chi2.ppf(1 - a / 2, 2 * n + 2) / 2.0
    return float(lo / anios), float(hi / anios)


def tasas(tabla: pd.DataFrame, est: Estratificacion, nivel: float = 0.95) -> pd.DataFrame:
    """Por estrato: ``n_eventos``, ``anios_con_dato`` (E_s), ``tasa_por_anio`` (λ_s) e IC de Garwood."""
    filas = []
    for s in est.nombres:
        sub = tabla.xs(s, level="estrato") if s in tabla.index.get_level_values("estrato") else tabla.iloc[0:0]
        n = int(sub["n_eventos"].sum()) if len(sub) else 0
        E = float(sub["fraccion"].sum()) if len(sub) else 0.0
        lo, hi = ic_garwood(n, E, nivel) if E > 0 else (np.nan, np.nan)
        filas.append({"estrato": s, "n_eventos": n, "anios_con_dato": E,
                      "tasa_por_anio": n / E if E > 0 else np.nan, "lo": lo, "hi": hi})
    return pd.DataFrame(filas).set_index("estrato")
