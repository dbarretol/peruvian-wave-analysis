"""Ruido de 1 Hz de la altimetría y función de estructura (contrato v6, §3.5 y §20.3).

**σ de 1 Hz** por misión y nivel de Hs (< 2, 2–4, ≥ 4 m), con las diferencias entre puntos consecutivos de una pasada:
σ̂ = 1.4826·MAD(Δ)/√2. Si los puntos consecutivos están correlacionados, σ̂ subestima σ (lo cubre la sensibilidad
σ × 2). El nivel de una diferencia es el de la media del par.

**Función de estructura** a lo largo de la traza, D(r) = ½·E[(Hs(x + r) − Hs(x))²], con r entre 20 y 50 km, sobre
los puntos de las pasadas del lado A. Lo que excede al ruido de 1 Hz, D − (σ_i² + σ_j²)/2, es la varianza de pequeña
escala; si supera el 25 % de la del ruido, se activa la regla de §20.3 y la sensibilidad de §3.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sitio.exposicion import distancia_km
from ..validacion import config as CV
from . import config as C


def nivel(hs) -> np.ndarray:
    return np.digitize(np.asarray(hs, float), C.NIVELES_HS)  # 0: < 2 m; 1: 2–4 m; 2: ≥ 4 m


def _diferencias(d: pd.DataFrame) -> pd.DataFrame:
    x = d.sort_values(["pasada", "tiempo"], kind="stable")
    mismo = x["pasada"].to_numpy()[1:] == x["pasada"].to_numpy()[:-1]
    dt = np.diff(x["tiempo"].to_numpy()).astype("timedelta64[ns]").astype(float) / 1e9
    ok = mismo & (dt > 0) & (dt <= C.DT_CONSECUTIVO_S)
    h = x["hs"].to_numpy(float)
    return pd.DataFrame(
        {"mision": x["mision"].to_numpy()[1:][ok], "delta": np.diff(h)[ok], "nivel": nivel((h[1:] + h[:-1]) / 2)[ok]}
    )


def _sigma(delta: np.ndarray) -> float:
    if delta.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(delta - np.median(delta))) / np.sqrt(2))


def sigma_1hz(puntos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por misión y nivel: diferencias, σ̂, de dónde sale el σ que se usa y correlación de rezago 1."""
    dif = _diferencias(puntos)
    filas = []
    for mision, g in dif.groupby("mision", sort=True):
        s_mision = _sigma(g["delta"].to_numpy())
        for k in range(len(C.NIVELES_HS) + 1):
            x = g.loc[g["nivel"] == k, "delta"].to_numpy()
            propio = x.size >= C.MIN_DIFERENCIAS_NIVEL
            filas.append(
                {
                    "mision": mision,
                    "nivel": k,
                    "nivel_m": ["< 2", "2–4", "≥ 4"][k],
                    "diferencias": int(x.size),
                    "sigma_nivel": _sigma(x),
                    "sigma_mision": s_mision,
                    "sigma": _sigma(x) if propio else s_mision,
                    "origen": "nivel" if propio else "misión (pocas diferencias en el nivel)",
                }
            )
    return pd.DataFrame(filas)


def correlacion_rezago1(puntos: pd.DataFrame) -> dict[str, float]:
    """Correlación entre diferencias sucesivas por misión (−0.5 con ruido blanco; descriptiva, §3.5)."""
    out = {}
    dif = _diferencias(puntos)
    for mision, g in dif.groupby("mision", sort=True):
        x = g["delta"].to_numpy()
        out[mision] = float(np.corrcoef(x[1:], x[:-1])[0, 1]) if x.size > 3 else float("nan")
    return out


def tabla_sigma(sig: pd.DataFrame) -> dict[tuple[str, int], float]:
    return {(r.mision, int(r.nivel)): float(r.sigma) for r in sig.itertuples()}


def sigma_superobs(so: pd.DataFrame, tabla: dict, factor: float = 1.0) -> np.ndarray:
    """σ de cada superobservación de altimetría: σ̂(misión, nivel de su Hs)/√n."""
    niv = nivel(so["hs_obs"])
    s = np.array([tabla.get((m, int(k)), np.nan) for m, k in zip(so["mision"], niv, strict=True)], float)
    return factor * s / np.sqrt(so["n"].to_numpy(float))


def sigma_boya(p: pd.DataFrame, rel: float = CV.RUIDO_BOYA_REL) -> np.ndarray:
    return rel * p["hs_obs"].to_numpy(float) / np.sqrt(p["k"].to_numpy(float))


def estructura(puntos: pd.DataFrame, tabla: dict, r_km=C.R_ESTRUCTURA_KM) -> dict:
    """D(r) entre 20 y 50 km sobre los puntos dados (los del lado A), y su exceso sobre el ruido de 1 Hz."""
    sumas = {"pares": 0, "d2": 0.0, "ruido": 0.0}
    s_pt = np.array(
        [tabla.get((m, int(k)), np.nan) for m, k in zip(puntos["mision"], nivel(puntos["hs"]), strict=True)], float
    )
    x = puntos.assign(_s2=s_pt**2).sort_values(["pasada", "tiempo"], kind="stable")
    for _, g in x.groupby("pasada", sort=True):
        if len(g) < 2:
            continue
        la, lo = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        h, s2 = g["hs"].to_numpy(float), g["_s2"].to_numpy(float)
        i, j = np.triu_indices(len(g), 1)
        dist = np.asarray(distancia_km(la[i], lo[i], la[j], lo[j]), float)
        sel = (dist >= r_km[0]) & (dist <= r_km[1])
        if not sel.any():
            continue
        sumas["pares"] += int(sel.sum())
        sumas["d2"] += float(np.sum(0.5 * (h[i][sel] - h[j][sel]) ** 2))
        sumas["ruido"] += float(np.sum(0.5 * (s2[i][sel] + s2[j][sel])))
    n = sumas["pares"]
    D = sumas["d2"] / n if n else float("nan")
    ruido = sumas["ruido"] / n if n else float("nan")
    var_ruido = float(np.nanmean(s_pt**2)) if len(s_pt) else float("nan")
    exceso = D - ruido
    return {
        "r_km": list(r_km),
        "pares": n,
        "D": D,
        "ruido_en_los_pares": ruido,
        "varianza_pequena_escala": exceso,
        "varianza_ruido_1hz": var_ruido,
        "cociente": exceso / var_ruido if var_ruido > 0 else float("nan"),
        "umbral": CV.UMBRAL_REPRESENTATIVIDAD,
        "supera_umbral": bool(np.isfinite(exceso) and exceso > CV.UMBRAL_REPRESENTATIVIDAD * var_ruido),
    }
