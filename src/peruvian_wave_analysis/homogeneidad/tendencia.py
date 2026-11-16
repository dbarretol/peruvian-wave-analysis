"""Deriva gradual (familia B) y punto de cambio de fecha desconocida (familia C), contrato de la F4 §3.5.

- **Theil-Sen** (Sen, 1968): mediana de las pendientes entre pares.
- **Mann-Kendall** con la varianza corregida por autocorrelación de Hamed y Ramachandra Rao (1998): la serie se
  quita la tendencia de Sen, se calcula la autocorrelación de los rangos y se suman solo los rezagos
  significativos al 5 % (|ρ_k| > 1.96/√n). Factor n/n* = 1 + 2/(n(n−1)(n−2))·Σ (n−k)(n−k−1)(n−k−2)·ρ_k.
- **Pettitt** (1979) y **SNHT** (Alexandersson, 1986), solo la posición del cambio y el estadístico: la familia C
  es exploratoria y no lleva p-valor de confirmación.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def sen(y, x=None) -> float:
    """Pendiente de Theil-Sen de y frente a x (por defecto, 0..n−1)."""
    y = np.asarray(y, float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, float)
    i, j = np.triu_indices(y.size, 1)
    d = x[j] - x[i]
    ok = d != 0
    return float(np.median((y[j] - y[i])[ok] / d[ok]))


def _s_y_varianza(y: np.ndarray) -> tuple[float, float]:
    n = y.size
    i, j = np.triu_indices(n, 1)
    s = float(np.sign(y[j] - y[i]).sum())
    _, t = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(t * (t - 1) * (2 * t + 5))) / 18.0
    return s, float(var)


def _z(s: float, var: float) -> float:
    if var <= 0:
        return 0.0
    return (s - np.sign(s)) / np.sqrt(var)


@dataclass(frozen=True)
class ResultadoMK:
    pendiente: float
    s: float
    z: float
    p: float
    factor: float  # n/n*; 1 sin corrección


def mann_kendall(y, corregir: bool = True) -> ResultadoMK:
    y = np.asarray(y, float)
    n = y.size
    pend = sen(y)
    s, var = _s_y_varianza(y)
    factor = 1.0
    if corregir and n > 3:
        r = stats.rankdata(y - pend * np.arange(n))
        r = r - r.mean()
        den = np.sum(r * r)
        suma = 0.0
        lim = 1.96 / np.sqrt(n)
        for k in range(1, n - 1):
            rho = np.sum(r[:-k] * r[k:]) / den if den > 0 else 0.0
            if abs(rho) > lim:
                suma += (n - k) * (n - k - 1) * (n - k - 2) * rho
        factor = 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * suma
        factor = max(factor, 1e-12)
    z = _z(s, var * factor)
    return ResultadoMK(pend, s, float(z), float(2 * stats.norm.sf(abs(z))), float(factor))


@dataclass(frozen=True)
class PuntoCambio:
    posicion: int  # índice del primer valor después del cambio
    estadistico: float


def pettitt(y) -> PuntoCambio:
    y = np.asarray(y, float)
    n = y.size
    sgn = np.sign(y[None, :] - y[:, None])  # sgn[i, j] = sign(y_j − y_i)
    U = np.array([sgn[: t + 1, t + 1 :].sum() for t in range(n - 1)])
    t = int(np.argmax(np.abs(U)))
    return PuntoCambio(t + 1, float(abs(U[t])))


def snht(y) -> PuntoCambio:
    y = np.asarray(y, float)
    n = y.size
    z = (y - y.mean()) / y.std(ddof=0)
    T = np.array([k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2 for k in range(1, n)])
    k = int(np.argmax(T))
    return PuntoCambio(k + 1, float(T[k]))
