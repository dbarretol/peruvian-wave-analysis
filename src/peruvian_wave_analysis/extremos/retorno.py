"""Niveles de retorno de picos sobre umbral, de un estrato y de varios.

Definición del proyecto: RL_T es el nivel con **una excedencia esperada cada T años** (intervalo medio de
recurrencia)::

    Σ_s λ_s · S_s(RL_T − u) = 1/T,   λ_s en eventos por año.

La probabilidad de excederlo al menos una vez en N años es 1 − exp(−N/T).

Cuando no hay nivel, el resultado es nulo **con su causa** (``NivelRetorno.causa``), nunca un número inventado.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .gpd import cuantil_supervivencia, supervivencia

# Causas de un nivel de retorno nulo
BAJO_EL_UMBRAL = "bajo_el_umbral"  # la tasa total por T años no llega a 1: el RL_T queda bajo u
SIN_ESTRATOS = "sin_estratos"
SIN_COTA = "sin_cota"  # la búsqueda de la raíz superó u + 1e6 sin cruzar 1/T

_COTA_BUSQUEDA = 1e6


@dataclass(frozen=True)
class EstratoRetorno:
    tasa_por_anio: float
    xi: float
    sigma: float


@dataclass(frozen=True)
class NivelRetorno:
    valor: float  # NaN si no hay nivel
    causa: str | None = None  # None si hay nivel

    @property
    def existe(self) -> bool:
        return self.causa is None


def nivel_retorno(u: float, xi: float, sigma: float, tasa_por_anio: float, T: float) -> NivelRetorno:
    """RL_T de un estrato en forma cerrada: u + σ/ξ·[(λT)^ξ − 1]. Nulo con causa ``bajo_el_umbral`` si λT ≤ 1."""
    p = 1.0 / (tasa_por_anio * T)
    if p >= 1.0:
        return NivelRetorno(np.nan, BAJO_EL_UMBRAL)
    return NivelRetorno(float(u + cuantil_supervivencia(p, xi, sigma)))


def tasa_de_excedencia(x: float, u: float, estratos: Sequence[EstratoRetorno]) -> float:
    """Σ_s λ_s · S_s(x − u), en excedencias por año."""
    return float(sum(e.tasa_por_anio * supervivencia(np.array([x - u]), e.xi, e.sigma)[0] for e in estratos))


def nivel_retorno_multiestrato(u: float, estratos: Sequence[EstratoRetorno], T: float) -> NivelRetorno:
    """RL_T con varios estratos: raíz de Σ_s λ_s·S_s(RL − u) = 1/T. Nulo con su causa si no hay solución sobre u."""
    if not estratos:
        return NivelRetorno(np.nan, SIN_ESTRATOS)
    objetivo = 1.0 / T

    def g(x: float) -> float:
        return tasa_de_excedencia(x, u, estratos) - objetivo

    if g(u) <= 0:
        return NivelRetorno(np.nan, BAJO_EL_UMBRAL)
    extremos = [(-e.sigma / e.xi) if e.xi < 0 else np.inf for e in estratos]
    hi = u + max(extremos) if np.all(np.isfinite(extremos)) else u + 1.0
    while g(hi) > 0:
        hi = u + 2.0 * (hi - u)
        if hi - u > _COTA_BUSQUEDA:
            return NivelRetorno(np.nan, SIN_COTA)
    return NivelRetorno(float(brentq(g, u, hi, xtol=1e-12, rtol=1e-12, maxiter=500)))


def probabilidad_en_n_anios(T: float, N: float) -> float:
    """Probabilidad de al menos una excedencia del RL_T en N años: 1 − exp(−N/T)."""
    return float(-np.expm1(-N / T))
