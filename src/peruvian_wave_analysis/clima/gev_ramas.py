"""Máximos por bloque y GEV por rama (contrato de la F6, §3.9). Contraste para la F9; no decide nada en la F6.

- **Bloque:** año calendario, salvo que más del 10 % de los máximos anuales de E0 sin corregir caiga en diciembre o
  enero; entonces el año de bloque empieza en el mes con menos máximos anuales (el primero, con empate).
- **Solo bloques completos:** el bloque entero dentro de la rama (E0, 1980–2025; E1, 1994–2025, con año calendario).
- **Ajuste:** ``extremos.gev`` (soporte: verosimilitud finita y |ξ̂| ≤ 0.5; G5). RL50 y RL100 de ADR-0022 con IC de
  perfil; al lado, el de Coles (1 − 1/T), como referencia.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..extremos import gev as G

FRACCION_DIC_ENE = 0.10
PERIODOS = (50.0, 100.0)


def mes_inicio(hs_e0: pd.Series) -> tuple[int, dict[int, int]]:
    """(mes de inicio del bloque, conteo por mes de los máximos anuales calendario de E0)."""
    s = hs_e0.dropna()
    anio = s.index.year
    completos = [a for a in np.unique(anio) if (s.index[anio == a].month.nunique() == 12)]
    idx = [s[anio == a].idxmax() for a in completos]
    meses = pd.DatetimeIndex(idx).month
    conteo = {m: int(np.sum(meses == m)) for m in range(1, 13)}
    if (conteo[12] + conteo[1]) / max(len(idx), 1) > FRACCION_DIC_ENE:
        return min(range(1, 13), key=lambda m: (conteo[m], m)), conteo
    return 1, conteo


def maximos(hs: pd.Series, m0: int = 1) -> pd.DataFrame:
    """Máximo de cada bloque completo: (bloque, t_max, hs_max, n_validos)."""
    t = hs.index
    bloque = t.year - (t.month < m0).astype(int)
    filas = []
    for b in np.unique(bloque):
        ini = pd.Timestamp(year=int(b), month=m0, day=1)
        fin = pd.Timestamp(year=int(b) + 1, month=m0, day=1) - pd.Timedelta(hours=3)
        if ini < t[0] or fin > t[-1]:
            continue
        s = hs[bloque == b].dropna()
        if s.empty:
            continue
        filas.append({"bloque": int(b), "t_max": s.idxmax(), "hs_max": float(s.max()), "n_validos": int(s.size)})
    return pd.DataFrame(filas)


def ajuste(mx: pd.DataFrame) -> dict:
    x = mx["hs_max"].to_numpy(float)
    aj = G.ajustar_gev(x)
    fila = {"n": int(x.size), "bloques": f"{mx['bloque'].min()}–{mx['bloque'].max()}" if len(mx) else "",
            "mu": aj.mu, "sigma": aj.sigma, "xi": aj.xi, "loglik": aj.loglik, "valido": aj.valido,
            "soporte_ok": aj.soporte_ok, "causa": aj.causa, "arranque": aj.arranque}  # fmt: skip
    if not aj.valido:
        return fila
    px = G.perfil_xi(x, aj)
    fila |= {"xi_lo": px.lo, "xi_hi": px.hi, "xi_causa_lo": px.causa_lo, "xi_causa_hi": px.causa_hi}
    for T in PERIODOS:
        pr = G.perfil_rl(x, aj, T)
        k = f"rl{T:g}"
        fila |= {k: pr.punto, f"{k}_lo": pr.lo, f"{k}_hi": pr.hi, f"{k}_causa_lo": pr.causa_lo,
                 f"{k}_causa_hi": pr.causa_hi, f"{k}_coles": aj.rl(T, "coles")}  # fmt: skip
    return fila


def por_rama(ramas: Mapping[str, pd.Series], m0: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    filas, mxs = [], {}
    for rama, hs in ramas.items():
        mx = maximos(hs, m0)
        mxs[rama] = mx
        filas.append({"rama": rama, "mes_inicio_bloque": m0} | ajuste(mx))
    return pd.DataFrame(filas), mxs


def g5(tabla: pd.DataFrame) -> list[str]:
    """Ramas cuyo ajuste falla o no pasa la validación de soporte (G5: detiene la etapa)."""
    malas = tabla[~(tabla["valido"].astype(bool) & tabla["soporte_ok"].astype(bool))]
    return [f"{r.rama}: {r.causa or 'ajuste no válido'}" for r in malas.itertuples()]
