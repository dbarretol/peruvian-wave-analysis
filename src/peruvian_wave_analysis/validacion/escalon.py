"""Estadísticos por lado con peso mensual y bootstrap por bloques de un mes (contrato v6, §3.4, §3.7 y §4.4).

Aquí se usan solo sobre el **modelo**: el escalón del modelo en las colocaciones de cada fuente, S_mod, frente al
del nodo en los mismos pasos, para la regla del ⅓ (§4.4) y el traslado por equivalencia (§3.7).

**Pesos.** Dentro de un lado, cada mes del calendario pesa lo mismo (``homogeneidad.cuantiles``, la definición de
la F4): w_i = 1/(M·N_m). El cuantil es el menor x con F_w(x) ≥ q.

**Bootstrap.** Bloques = meses (año y mes) dentro de cada lado, remuestreados con reposición por separado en cada
lado. Una réplica es un vector de multiplicidades por bloque; los pesos y los cuantiles se recalculan con esas
multiplicidades sobre los valores ya ordenados (sin volver a ordenar). Las dos series del mismo lado (modelo en la
colocación y nodo en los mismos pasos) comparten las réplicas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

ESTADISTICOS = ("P50", "P90", "P99", "P99.9", "max")
MIN_MESES_MAX = 12  # §3.4: la media de máximos mensuales exige al menos 12 meses


@dataclass
class LadoBoot:
    """Valores de un lado preparados para el bootstrap: varias series sobre las mismas colocaciones."""

    series: np.ndarray  # (s, n)
    mes: np.ndarray  # (n,) mes del calendario, 1–12
    bloque: np.ndarray  # (n,) índice del bloque (año-mes), 0..K−1
    K: int

    @classmethod
    def desde(cls, tiempos, *series) -> LadoBoot:
        t = pd.DatetimeIndex(tiempos)
        ym = (t.year * 12 + t.month - 1).to_numpy()
        _, bloque = np.unique(ym, return_inverse=True)
        return cls(
            np.vstack([np.asarray(s, float) for s in series]),
            t.month.to_numpy(),
            bloque,
            int(bloque.max()) + 1 if len(t) else 0,
        )


def _estadisticos(lado: LadoBoot, cuentas: np.ndarray) -> np.ndarray:
    """(r, s, 5) para r réplicas dadas por ``cuentas`` (r, K)."""
    r = cuentas.shape[0]
    s, n = lado.series.shape
    out = np.full((r, s, len(ESTADISTICOS)), np.nan)
    if n == 0:
        return out
    mult = cuentas[:, lado.bloque].astype(float)  # (r, n)
    por_mes = np.zeros((lado.K, 13))
    np.add.at(por_mes, (lado.bloque, lado.mes), 1.0)
    N = cuentas @ por_mes  # (r, 13)
    M = (N[:, 1:] > 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = mult / (M[:, None] * N[:, lado.mes])
    w = np.nan_to_num(w)
    qs = np.array([C.CUANTILES[k] for k in ESTADISTICOS[:4]])
    for j in range(s):
        x = lado.series[j]
        o = np.argsort(x, kind="stable")
        xs, ws = x[o], w[:, o]
        F = np.cumsum(ws, axis=1)
        F /= F[:, -1:]
        for k, q in enumerate(qs):
            i = (F < q - 1e-12).sum(axis=1)
            out[:, j, k] = xs[np.minimum(i, n - 1)]
        # media de los máximos mensuales: máximo de cada bloque, ponderado por su multiplicidad
        maxb = np.full(lado.K, -np.inf)
        np.maximum.at(maxb, lado.bloque, x)
        meses = (cuentas > 0).sum(axis=1)
        media = (cuentas @ maxb) / cuentas.sum(axis=1)
        out[:, j, 4] = np.where(meses >= MIN_MESES_MAX, media, np.nan)
    return out


def puntual(lado: LadoBoot) -> np.ndarray:
    """(s, 5): estadísticos de cada serie con todas las colocaciones."""
    return _estadisticos(lado, np.ones((1, lado.K)))[0]


def replicas(lado: LadoBoot, B: int, rng: np.random.Generator, trozo: int = 500) -> np.ndarray:
    """(B, s, 5): estadísticos de B réplicas por bloques de un mes."""
    partes = []
    hechas = 0
    while hechas < B:
        r = min(trozo, B - hechas)
        cuentas = rng.multinomial(lado.K, np.full(lado.K, 1.0 / lado.K), size=r) if lado.K else np.zeros((r, 0))
        partes.append(_estadisticos(lado, cuentas))
        hechas += r
    return np.concatenate(partes, axis=0)


@dataclass
class Escalon:
    """S(estadístico) = log(lado2 / lado1) de cada serie, puntual y por réplica."""

    puntual: np.ndarray  # (s, 5)
    replicas: np.ndarray  # (B, s, 5)


def escalon(l1: LadoBoot, l2: LadoBoot, B: int, rng: np.random.Generator) -> Escalon:
    p = np.log(puntual(l2) / puntual(l1))
    r = np.log(replicas(l2, B, rng) / replicas(l1, B, rng))
    return Escalon(p, r)


def ic(replicas_1d: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    r = replicas_1d[np.isfinite(replicas_1d)]
    if r.size == 0:
        return float("nan"), float("nan")
    a = (1 - nivel) / 2
    lo, hi = np.quantile(r, [a, 1 - a])
    return float(lo), float(hi)
