"""Casos sintéticos de la simulación previa (contrato de la F7, §5.1).

Los picos se generan directamente: con dos umbrales, los picos de la F7 son los máximos de eventos ya identificados, y
subir u solo descarta picos. En cada réplica:

- número de picos ~ Poisson(N0) (llegadas de Poisson en los años del caso, con tasa N0 / años);
- exceso sobre u_id según el caso; pico = u_id + exceso, redondeado a 0.01 m (C-09); un pico que queda ≤ u_id toma
  u_id + 0.01 m;
- parámetros absolutos (C-01): u_id = ``p95_hs_m`` de la F3; σ = 0.35 m en A, B+, B− y el control (C2-06).

- **A:** excesos GPD(ξ, σ), ξ ∈ {−0.1, 0, +0.1}.
- **B+ y B−:** abajo, GPD(ξ_abajo, σ) truncada en u0; arriba, GPD(ξ0, σ + ξ_abajo·(u0 − u_id)), con densidad continua
  (E-20); ξ0 ∈ {−0.1, 0} y ξ_abajo = ξ0 ± 0.15.
- **Control:** como B−, con ξ_abajo = ξ0 − 0.30 (ADR-0033); no entra en la métrica.
- **C:** Weibull de Hs con k ∈ {1.5, 2} y P95 = ``p95_hs_m``, truncada en u_id.

**u0 (E-25, C-05):** en cada réplica, k₀ ~ U[4, 14] continuo y u0 es el nivel con número esperado de picos
n(k₀) = N0·r^(−k₀/20) por encima, r = min(20, N0/30): la masa de los excesos bajo u0 es 1 − n(k₀)/N0.

**RL100 verdadero:** de la distribución y la tasa verdaderas (N0 / años), con ADR-0022: λ·S(RL − u_id) = 1/T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..extremos import gpd
from .umbral import N_MIN_ARRIBA, R_MAX, RESOLUCION_M

SIGMA = 0.35
DELTA_XI = 0.15
DELTA_XI_CONTROL = 0.30
K0_RANGO = (4.0, 14.0)
P_WEIBULL = 0.95

CASOS = ("A", "B+", "B-", "control", "C")


@dataclass(frozen=True)
class Celda:
    caso: str
    forma: float  # ξ en A; ξ0 en B+, B− y control; k de Weibull en C
    N0: int
    anios: float
    u_id: float
    sigma: float = SIGMA

    @property
    def nombre(self) -> str:
        return f"{self.caso}_{self.forma:g}_N{self.N0}"

    @property
    def tasa(self) -> float:
        return self.N0 / self.anios

    @property
    def xi_abajo(self) -> float:
        d = {"B+": DELTA_XI, "B-": -DELTA_XI, "control": -DELTA_XI_CONTROL}
        return self.forma + d[self.caso]

    def escala_weibull(self) -> float:
        return self.u_id / (-np.log(1.0 - P_WEIBULL)) ** (1.0 / self.forma)


def celdas(n0: tuple[int, int], anios: tuple[float, float], u_id: float, control: bool = False) -> list[Celda]:
    """Las 18 celdas de la métrica (o las 4 del control): (caso, forma) × N0, con los años de cada N0."""
    formas = {"A": (-0.1, 0.0, 0.1), "B+": (-0.1, 0.0), "B-": (-0.1, 0.0), "C": (1.5, 2.0)}
    if control:
        formas = {"control": (-0.1, 0.0)}
    return [
        Celda(c, f, int(n), float(a), u_id)
        for c, fs in formas.items()
        for f in fs
        for n, a in zip(n0, anios, strict=True)
    ]


def r_de(N0: int) -> float:
    return min(R_MAX, N0 / N_MIN_ARRIBA)


def masa_bajo_u0(k0: float, N0: int) -> float:
    """1 − n(k₀)/N0, con n(k₀) = N0·r^(−k₀/20)."""
    return 1.0 - r_de(N0) ** (-k0 / 20.0)


def d0_de(celda: Celda, k0: float) -> float:
    """u0 − u_id: el cuantil de la parte de abajo con masa 1 − n(k₀)/N0."""
    q0 = masa_bajo_u0(k0, celda.N0)
    return float(gpd.cuantil_supervivencia(1.0 - q0, celda.xi_abajo, celda.sigma))


def sigma_arriba(celda: Celda, d0: float) -> float:
    return celda.sigma + celda.xi_abajo * d0


def supervivencia(celda: Celda, x, k0: float | None = None) -> np.ndarray:
    """S(x) de los excesos sobre u_id de la celda."""
    x = np.asarray(x, dtype=float)
    if celda.caso == "A":
        return gpd.supervivencia(x, celda.forma, celda.sigma)
    if celda.caso == "C":
        lam, k = celda.escala_weibull(), celda.forma
        h = celda.u_id + np.maximum(x, 0.0)
        return np.exp(-((h / lam) ** k) + (celda.u_id / lam) ** k)
    d0 = d0_de(celda, k0)
    s0 = 1.0 - masa_bajo_u0(k0, celda.N0)
    abajo = gpd.supervivencia(x, celda.xi_abajo, celda.sigma)
    arriba = s0 * gpd.supervivencia(x - d0, celda.forma, sigma_arriba(celda, d0))
    return np.where(x <= d0, abajo, arriba)


def excesos(celda: Celda, n: int, rng: np.random.Generator, k0: float | None = None) -> np.ndarray:
    """n excesos sobre u_id, sin redondear."""
    U = rng.random(n)
    if celda.caso == "A":
        return gpd.cuantil_supervivencia(U, celda.forma, celda.sigma)
    if celda.caso == "C":
        lam, k = celda.escala_weibull(), celda.forma
        h = lam * ((celda.u_id / lam) ** k - np.log(U)) ** (1.0 / k)
        return h - celda.u_id
    d0 = d0_de(celda, k0)
    q0 = masa_bajo_u0(k0, celda.N0)
    out = np.empty(n)
    abajo = U < q0
    out[abajo] = gpd.cuantil_supervivencia(1.0 - U[abajo], celda.xi_abajo, celda.sigma)
    sup = (1.0 - U[~abajo]) / (1.0 - q0)  # supervivencia condicional sobre u0, en (0, 1]
    out[~abajo] = d0 + gpd.cuantil_supervivencia(sup, celda.forma, sigma_arriba(celda, d0))
    return out


def redondear_picos(u_id: float, x: np.ndarray, resolucion: float = RESOLUCION_M) -> np.ndarray:
    h = np.round((u_id + np.asarray(x, dtype=float)) / resolucion) * resolucion
    return np.where(h <= u_id + 1e-9, np.round((u_id + resolucion) / resolucion) * resolucion, h)


@dataclass(frozen=True)
class Replica:
    picos: np.ndarray
    k0: float | None
    u0: float | None  # nivel absoluto de u0 (B+, B−, control)
    rl100_verdadero: float


def rl_verdadero(celda: Celda, T: float = 100.0, k0: float | None = None) -> float:
    """RL_T con λ·S(RL − u_id) = 1/T (ADR-0022)."""
    objetivo = 1.0 / (celda.tasa * T)

    def g(x):
        return float(supervivencia(celda, np.array([x]), k0)[0]) - objetivo

    hi = 1.0
    while g(hi) > 0:
        hi *= 2.0
    return celda.u_id + float(brentq(g, 0.0, hi, xtol=1e-12, rtol=1e-12))


def replica(celda: Celda, rng: np.random.Generator) -> Replica:
    n = int(rng.poisson(celda.N0))
    k0 = float(rng.uniform(*K0_RANGO)) if celda.caso in ("B+", "B-", "control") else None
    x = excesos(celda, n, rng, k0)
    u0 = celda.u_id + d0_de(celda, k0) if k0 is not None else None
    return Replica(redondear_picos(celda.u_id, x), k0, u0, rl_verdadero(celda, 100.0, k0))
