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


# --- Mann-Kendall calibrado por simulación (familia B; contrato §19, D1) -------------------------


def ar1_lote(R: int, n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """R series AR(1) estacionarias de varianza 1 (R × n)."""
    e = rng.standard_normal((R, n))
    y = np.empty((R, n))
    y[:, 0] = e[:, 0]
    s = np.sqrt(1 - rho**2)
    for k in range(1, n):
        y[:, k] = rho * y[:, k - 1] + s * e[:, k]
    return y


def z_mk_lote(Y: np.ndarray) -> np.ndarray:
    """z de Mann-Kendall sin corrección, por fila (sin empates: series continuas)."""
    Y = np.atleast_2d(Y)
    n = Y.shape[1]
    i, j = np.triu_indices(n, 1)
    S = np.sign(Y[:, j] - Y[:, i]).sum(axis=1)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    return (S - np.sign(S)) / np.sqrt(var)


def nula_mk(n: int, rho: float, R: int, rng: np.random.Generator, lote: int = 5000) -> np.ndarray:
    """|z| de Mann-Kendall sin corrección en R series AR(1) sin tendencia. Como la prueba usa rangos, el resultado no
    depende de la distribución marginal: solo de n y de ρ (supuesto)."""
    out = [np.abs(z_mk_lote(ar1_lote(min(lote, R - k), n, rho, rng))) for k in range(0, R, lote)]
    return np.sort(np.concatenate(out))


def p_monte_carlo(z: float, nula_abs: np.ndarray) -> float:
    """p bilateral = (1 + #{|z_nula| ≥ |z|}) / (N + 1), con ``nula_abs`` ordenada."""
    k = nula_abs.size - np.searchsorted(nula_abs, abs(z) - 1e-12, side="left")
    return float((1 + k) / (nula_abs.size + 1))


def mk_calibrado(y, nula_abs: np.ndarray) -> dict:
    """Mann-Kendall sin corrección (con la varianza corregida por empates) y su p de Monte Carlo frente a la nula
    AR(1). Incluye Hamed-Rao con su p nominal como sensibilidad rotulada."""
    y = np.asarray(y, float)
    sin = mann_kendall(y, corregir=False)
    hr = mann_kendall(y, corregir=True)
    return {
        "pendiente": sin.pendiente,
        "s": sin.s,
        "z": sin.z,
        "p": p_monte_carlo(sin.z, nula_abs),
        "sensibilidad_hamed_rao": {
            "z": hr.z,
            "p_nominal": hr.p,
            "factor": hr.factor,
            "rotulo": "no mantiene el nivel con n ≈ 20 (contrato §19, D1)",
        },
    }
