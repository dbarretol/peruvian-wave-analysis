"""Procesos sintéticos de la etapa 0 y de las pruebas de la F6 (contrato §8). Sin datos del sitio.

- **AR(1)** gaussiano de varianza 1 con τ_e conocido: φ = exp(−PASO_H / τ_e), así ρ(τ_e) = 1/e.
- **Proceso max-autorregresivo** de Ferro y Segers (2003, ec. 6): X₁ = W₁/θ, X_n = max((1 − θ) X_{n−1}, W_n), con W
  Fréchet unitaria (como en el artículo). Marginal estacionaria Fréchet de escala 1/θ, P(X ≤ z) = exp(−1/(θz)), e
  índice extremal θ exacto. Con θ = 1 son excedencias independientes.
- **Ciclo anual** sinusoidal con máximo a mitad del invierno austral (mayo–octubre), con amplitud calibrada para que
  la fracción esperada de excedencias del P99 en mayo–octubre sea 0.86 (como en la simulación del revisor):
  - AR(1): X_t + A cos(fase_t);
  - Ferro y Segers: X_t · exp(A cos(fase_t)) (la tasa de excedencias de un umbral alto es ∝ a la escala).

El eje de tiempo es el de la serie real: ``LARGO`` pasos de 3 h desde 1980-01-01.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import lfilter
from scipy.stats import norm

from ..extremos.dependencia import PASO_H

INICIO = "1980-01-01"
LARGO = 136_360  # 1980-01-01 a 2026-08-31 21:00, el largo de la serie del nodo
INVIERNO = (5, 6, 7, 8, 9, 10)
FRACCION_INVIERNO = 0.86
Q_UMBRAL = 0.99


def eje(n: int = LARGO) -> pd.DatetimeIndex:
    return pd.date_range(INICIO, periods=n, freq=f"{int(PASO_H)}h")


def fase(t: pd.DatetimeIndex) -> np.ndarray:
    """Fase del ciclo anual con máximo en el centro de mayo–octubre (1 de agosto, 00 h, ≈ día 212.5)."""
    dias = (t - pd.Timestamp("1980-08-01")) / pd.Timedelta(days=1)
    return 2 * np.pi * np.asarray(dias, dtype=float) / 365.2425


def ar1(n: int, tau_e_h: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.exp(-PASO_H / tau_e_h)
    e = rng.standard_normal(n) * np.sqrt(1 - phi**2)
    e[0] = rng.standard_normal()  # arranque estacionario
    return lfilter([1.0], [1.0, -phi], e)


def ferro_segers(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Proceso de la ec. 6, vectorizado: log X_n = n c + máx_{j ≤ n} (log W_j − j c), con c = log(1 − θ)."""
    w = -1.0 / np.log(rng.random(n))
    if theta >= 1.0:
        return w
    c = np.log1p(-theta)
    j = np.arange(n, dtype=float)
    v = np.log(w) - j * c
    v[0] = np.log(w[0] / theta)
    return np.exp(j * c + np.maximum.accumulate(v))


def _fraccion_invierno(prob: np.ndarray, invierno: np.ndarray) -> float:
    return float(prob[invierno].sum() / prob.sum())


def amplitud(proceso: str, n: int = LARGO) -> float:
    """Amplitud del ciclo tal que la fracción esperada de excedencias del P99 en mayo–octubre sea 0.86."""
    return _amplitud(proceso, int(n))


@cache
def _amplitud(proceso: str, n: int) -> float:
    t = eje(n)
    cos = np.cos(fase(t))
    inv = np.isin(t.month, INVIERNO)

    def prob(A: float) -> np.ndarray:
        if proceso == "ar1":
            f = lambda u: norm.sf(u - A * cos).mean() - (1 - Q_UMBRAL)  # noqa: E731
            u = optimize.brentq(f, -10, 20)
            return norm.sf(u - A * cos)
        s = np.exp(A * cos)
        f = lambda lu: (-np.expm1(-s / np.exp(lu))).mean() - (1 - Q_UMBRAL)  # noqa: E731
        lu = optimize.brentq(f, -5, 30)
        return -np.expm1(-s / np.exp(lu))

    return float(optimize.brentq(lambda A: _fraccion_invierno(prob(A), inv) - FRACCION_INVIERNO, 0.0, 10.0))


def generar(proceso: str, parametro: float, ciclo: bool, rng: np.random.Generator, n: int = LARGO) -> pd.Series:
    """Serie sintética en el eje de la serie real. ``proceso``: «ar1» (parámetro τ_e en h) o «fs» (θ)."""
    t = eje(n)
    if proceso == "ar1":
        x = ar1(n, parametro, rng)
        if ciclo:
            x = x + amplitud("ar1", n) * np.cos(fase(t))
    elif proceso == "fs":
        x = ferro_segers(n, parametro, rng)
        if ciclo:
            x = x * np.exp(amplitud("fs", n) * np.cos(fase(t)))
    else:
        raise ValueError(f"Proceso desconocido: {proceso}")
    return pd.Series(x, index=t)
