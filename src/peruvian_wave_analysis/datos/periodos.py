"""Períodos largos entre los estados extremos y razón de asomeramiento (contrato de la F3, §3.11; ADR-0015).

Dos definiciones de «estado extremo», fijadas antes de ver la serie:
- estados con Hs ≥ P99 de la serie válida;
- el estado del máximo de cada año completo.

Ks es el coeficiente de asomeramiento lineal respecto de aguas profundas: Ks = [tanh(kh)·(1 + 2kh/sinh 2kh)]^(−1/2),
con k de la relación de dispersión ω² = g·k·tanh(kh) y ω = 2π/Tp. Se publica la razón media Ks(h_sitio)/Ks(h_modelo).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config as C

UMBRALES_TP = (16.0, 18.0)
PERCENTIL_EXTREMOS = 99.0


def numero_de_onda(T: np.ndarray | float, h: float) -> np.ndarray:
    """k de ω² = g·k·tanh(kh), por Newton desde la aproximación de aguas profundas."""
    w = 2.0 * np.pi / np.asarray(T, dtype=float)
    k = np.maximum(w**2 / C.G, w / np.sqrt(C.G * h))
    for _ in range(50):
        th = np.tanh(k * h)
        f = C.G * k * th - w**2
        df = C.G * th + C.G * k * h * (1.0 - th**2)
        paso = f / df
        k = k - paso
        if np.all(np.abs(paso) < 1e-12 * k):
            break
    return k


def asomeramiento(T: np.ndarray | float, h: float) -> np.ndarray:
    kh = numero_de_onda(T, h) * h
    dos = 2.0 * kh
    cociente = np.where(dos < 700.0, dos / np.sinh(np.minimum(dos, 700.0)), 0.0)  # → 0 en aguas profundas
    return 1.0 / np.sqrt(np.tanh(kh) * (1.0 + cociente))


def estados_extremos(df: pd.DataFrame, anios_completos: Iterable[int]) -> dict[str, pd.DataFrame]:
    hs = df["VHM0"]
    p99 = float(np.nanpercentile(hs, PERCENTIL_EXTREMOS))
    sobre = df[hs >= p99]
    completos = df[df.index.year.isin(list(anios_completos))]
    maximos = completos.loc[completos.groupby(completos.index.year)["VHM0"].idxmax()]
    return {"sobre_p99": sobre, "maximo_anual": maximos}


def resumen(df: pd.DataFrame, anios_completos, h_sitio: float, h_modelo: float) -> dict:
    """Fracción con Tp ≥ 16 s y ≥ 18 s y razón media Ks(h_sitio)/Ks(h_modelo), por definición de extremo."""
    out = {"h_sitio_m": h_sitio, "h_modelo_m": h_modelo}
    for nombre, e in estados_extremos(df, anios_completos).items():
        tp = e["VTPK"].dropna()
        out[nombre] = {
            "n": int(tp.size),
            **{f"fraccion_tp_ge_{int(u)}s": float((tp >= u).mean()) if tp.size else None for u in UMBRALES_TP},
            "razon_ks_media": float(
                np.mean(asomeramiento(tp.to_numpy(), h_sitio) / asomeramiento(tp.to_numpy(), h_modelo))
            )
            if tp.size
            else None,
        }
    return out
