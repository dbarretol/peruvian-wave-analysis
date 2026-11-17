"""Cuantiles por tramo, cuerpo frente a cola e indicadores anuales (contrato de la F4, §3.3). Son descriptivos: IC
del 95 % por bootstrap de bloques anuales, sin p-valor.

- Cuantiles con peso mensual igual (``cuantiles.HistAnual``). Estación «todo» con años calendario; «invierno»
  (mayo a octubre) con años que empiezan en mayo; «verano» (noviembre a abril) con años que empiezan en noviembre,
  para que ningún verano quede partido.
- Media de máximos: solo años completos (todos los meses de la estación).
- Tp con Hs ≥ P99 del registro completo: estados (registros), sin peso mensual (son pocos por mes).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .altimetros import cocientes_relativos
from .cuantiles import HistAnual, cuantiles_mensuales

ESTADISTICOS = ("P50", "P90", "P99", "P99.9", "media_maximos")


def _remuestreos(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]) if n else np.zeros((B, 0))


def _ic(rep: np.ndarray) -> tuple[float, float]:
    r = rep[np.isfinite(rep)]
    return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) if r.size else (np.nan, np.nan)


def cuantiles_de_grupo(serie: pd.Series, mes_inicio: int, meses, qs, B: int, rng) -> dict[str, tuple]:
    """estadístico → (valor, lo, hi, n_anios_completos)."""
    h = HistAnual.desde_serie(serie, mes_inicio, meses)
    n = h.anios.size
    if n == 0:
        return {}
    uno = np.ones((1, n))
    P = _remuestreos(n, B, rng)
    q0 = h.cuantiles(uno, qs)[0]
    qb = h.cuantiles(P, qs)
    m0 = float(h.media_maximos(uno)[0])
    mb = h.media_maximos(P)
    out = {}
    for j, nombre in enumerate(ESTADISTICOS[: len(qs)]):
        out[nombre] = (float(q0[j]), *_ic(qb[:, j]), int(h.completo.sum()))
    out["media_maximos"] = (m0, *_ic(mb), int(h.completo.sum()))
    return out


def tp_hs_alto(df: pd.DataFrame, umbral: float, B: int, rng) -> dict[str, tuple]:
    """Mediana, P10 y P90 de Tp en los estados con Hs ≥ umbral, con IC por años."""
    sel = df[(df["VHM0"] >= umbral) & np.isfinite(df["VTPK"])]
    if sel.empty:
        return {}
    anio = sel.index.year.to_numpy()
    anios, pos = np.unique(anio, return_inverse=True)
    tp = sel["VTPK"].to_numpy(float)
    qs = (0.5, 0.1, 0.9)
    base = np.quantile(tp, qs)
    reps = []
    for _ in range(B):
        k = rng.integers(0, anios.size, anios.size)
        idx = np.concatenate([np.flatnonzero(pos == j) for j in k])
        reps.append(np.quantile(tp[idx], qs) if idx.size else [np.nan] * 3)
    reps = np.asarray(reps)
    return {
        nombre: (float(base[j]), *_ic(reps[:, j]), int(anios.size))
        for j, nombre in enumerate(("mediana", "P10", "P90"))
    }


def cociente_p50(s1: pd.Series, s2: pd.Series, B: int, rng) -> tuple[float, float, float]:
    """Cociente de P50 (lado 2 / lado 1) con IC por bootstrap de años, independiente en cada lado."""
    h1, h2 = HistAnual.desde_serie(s1), HistAnual.desde_serie(s2)
    p1 = h1.cuantiles(np.ones((1, h1.anios.size)), (0.5,))[0, 0]
    p2 = h2.cuantiles(np.ones((1, h2.anios.size)), (0.5,))[0, 0]
    r1 = h1.cuantiles(_remuestreos(h1.anios.size, B, rng), (0.5,))[:, 0]
    r2 = h2.cuantiles(_remuestreos(h2.anios.size, B, rng), (0.5,))[:, 0]
    return float(p2 / p1), *_ic(r2 / r1)


def mediana_vecinas(vecinas: pd.DataFrame, nodo: str, paso: float = 0.2) -> pd.Series:
    """Mediana por registro de las vecinas del primer anillo con dato."""
    la0, lo0 = map(float, nodo.split("_"))
    cols = []
    for c in vecinas.columns:
        la, lo = map(float, c.split("_"))
        if c != nodo and abs(la - la0) <= paso + 1e-6 and abs(lo - lo0) <= paso + 1e-6 and vecinas[c].notna().any():
            cols.append(c)
    return vecinas[cols].median(axis=1, skipna=True).rename("Hs_vecinas_mediana")


def anuales_hs(hs: pd.Series, excluir: tuple[str, str] | None = None) -> pd.DataFrame:
    """Por año calendario: P50 y P99 con peso mensual, máximo, meses con dato.

    ``excluir``: intervalo de días [a, b] que no entra en los estadísticos (el mes de transición 1993-01, §3.2;
    §21, K-06). ``meses`` sigue contando los meses con dato del registro, así que el año no deja de ser completo."""
    filas = []
    fuera = np.zeros(len(hs), bool)
    if excluir is not None:
        a0, b0 = pd.Timestamp(excluir[0]), pd.Timestamp(excluir[1]) + pd.Timedelta(days=1)
        fuera = np.asarray((hs.index >= a0) & (hs.index < b0))
    meses_reg = hs.dropna().groupby(hs.dropna().index.year).apply(lambda g: g.index.month.nunique())
    for a, g in hs[~fuera].groupby(hs[~fuera].index.year):
        g = g.dropna()
        if g.empty:
            continue
        q = cuantiles_mensuales(g, (0.5, 0.99))
        filas.append(
            {
                "anio": int(a),
                "hs_p50": float(q[0]),
                "hs_p99": float(q[1]),
                "hs_max": float(g.max()),
                "meses": int(meses_reg.get(a, g.index.month.nunique())),
            }
        )
    return pd.DataFrame(filas).set_index("anio")


def indicadores_anuales(
    serie: pd.DataFrame,
    excursiones: pd.DataFrame,
    candidatos: pd.DataFrame,
    marcas_anio: pd.DataFrame,
    excluir: tuple[str, str] | None = None,
) -> pd.DataFrame:
    hs = serie["VHM0"]
    out = anuales_hs(hs, excluir)
    out["completo"] = out["meses"] == 12
    out["excursiones_k_rev"] = excursiones.groupby("anio").size().reindex(out.index, fill_value=0)
    det = candidatos[candidatos["regla"].isin(["detector", "detector_revision"])]
    out["candidatos_detector"] = (
        det.groupby(pd.to_datetime(det["tiempo"]).dt.year).size().reindex(out.index, fill_value=0)
    )
    if "marca_discrepancia_energia" in serie:
        de = serie["marca_discrepancia_energia"].abs()
        out["discrepancia_energia_mediana"] = de.groupby(de.index.year).median().reindex(out.index)
        out["discrepancia_energia_p99"] = de.groupby(de.index.year).quantile(0.99).reindex(out.index)
    # variabilidad de 3 h: P99 anual de |ΔHs|/Hs(t−1) en pasos con Hs(t−1) ≥ P50 del registro
    r = pd.Series(cocientes_relativos(hs.to_numpy(float)), index=hs.index)
    p50 = float(np.nanquantile(hs.to_numpy(float), 0.5))
    prev = hs.shift(1)
    r = r[(prev >= p50) & r.notna()]
    out["variabilidad_3h_p99"] = r.groupby(r.index.year).quantile(0.99).reindex(out.index)
    m = marcas_anio.copy()
    m.index = m.index.astype(int)
    out = out.join(m.add_prefix("marcas_"), how="left")
    return out


def coincidencias(
    anios_cambio: Sequence[int], anios_cronologia: Sequence[int], anios_registro: Sequence[int], ventana: int
) -> tuple[list[bool], float]:
    """Si cada año de cambio cae a ±ventana de alguno de la cronología, y la probabilidad de coincidir por azar
    (fracción de años del registro a ±ventana de alguno de la cronología)."""
    cr = np.asarray(sorted(set(anios_cronologia)))
    co = [bool(np.any(np.abs(cr - a) <= ventana)) for a in anios_cambio]
    reg = np.asarray(anios_registro)
    azar = float(np.mean([np.any(np.abs(cr - a) <= ventana) for a in reg])) if reg.size else np.nan
    return co, azar
