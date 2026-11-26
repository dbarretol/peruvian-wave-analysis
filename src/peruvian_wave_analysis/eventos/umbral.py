"""Rejilla de umbrales sobre los picos, ajuste en cada umbral y reglas R1 y R3 (contrato de la F7, §3.4).

**Rejilla (E-01, C-09), la misma función en la simulación y en los datos.** Con N0 picos y r = min(20, N0/30):
n_k = round(N0 · r^(−k/20)), k = 1, …, 20. Sobre los valores **distintos** de los picos, de mayor a menor, u_k es el
punto medio entre dos valores distintos consecutivos, elegido para que el número de picos **estrictamente mayores**
que u_k sea el más cercano a n_k (con empate, el mayor número). Se publica el n real. Si dos k dan el mismo u, se
conserva el primero y se marca ``rejilla_con_repetidos``.

**Ajuste en u_k:** GPD de los excesos (``extremos.gpd.ajustar_gpd``), λ = n / años, σ* = σ̂ − ξ̂·u y RL_T con
ADR-0022 (``extremos.retorno.nivel_retorno``). Un RL_1 nulo con causa ``bajo_el_umbral`` no es un fallo.

**R1 (estabilidad del RL100).** Tramo estable: ≥ 3 u_k consecutivos con (máx − mín) / mediana del RL100 < 2 %, que
no se puede extender en ningún sentido sin romper el 2 %. Un u_k sin RL100 corta el tramo (C-08).

- R1a: el tramo estable más largo (empate: el más bajo); u* = su centro (con un número par, el más bajo de los dos
  centrales);
- R1b: el tramo estable más alto, el que llega al u_k mayor (empate: el más largo); u* = su umbral más bajo.
- Sin tramo estable: «sin resultado». Si el tramo abarca toda la rejilla: «no discriminó».

**R3 (Solari et al., 2017, §3.1, con las cinco diferencias declaradas en §3.4).** A²_R ponderado a la cola derecha
(Sinclair et al., 1990; ec. 5): A²_R = n/2 − 2·Σ z_i − Σ (2 − (2i − 1)/n)·log(1 − z_i), con z_i = F(x_(i)). El p se
calcula por bootstrap paramétrico con reajuste, B = 99 en la simulación y en los datos; las réplicas se redondean a
0.01 m como los datos (C2-05). p = (1 + m)/(B + 1), con m las réplicas con A²_R ≥ el observado. u* = el menor u_k con
p ≥ p_máx − 2·√(max{p_máx(1 − p_máx), 0.0475}/B) (E-31, C2-03). «Sin resultado» si p_máx < 0.10.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..extremos import gpd, gpd_perfil, retorno

K = 20
N_MIN_ARRIBA = 30
R_MAX = 20.0
PERIODOS = (1.0, 10.0, 50.0, 100.0)
RESOLUCION_M = 0.01
TOL_R1 = 0.02
LARGO_MIN_R1 = 3
B_R3 = 99
P_MIN_R3 = 0.10
PISO_VAR_R3 = 0.0475  # p(1 − p) con p = 0.95
FACTOR_MAX_REPLICAS_R3 = 2.0

SIN_RESULTADO = "sin_resultado"
NO_DISCRIMINO = "no_discrimino"
CON_RESULTADO = "con_resultado"


# --- Rejilla -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rejilla:
    k: np.ndarray
    n_objetivo: np.ndarray
    u: np.ndarray
    n_real: np.ndarray
    con_repetidos: bool
    k_repetidos: tuple[int, ...]

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame({"k": self.k, "n_objetivo": self.n_objetivo, "u": self.u, "n_real": self.n_real})


def objetivos(N0: int, K_: int = K) -> np.ndarray:
    r = min(R_MAX, N0 / N_MIN_ARRIBA)
    if r <= 1:
        raise ValueError(f"Con {N0} picos no hay rejilla: hacen falta más de {N_MIN_ARRIBA}")
    k = np.arange(1, K_ + 1)
    return np.round(N0 * r ** (-k / K_)).astype(int)


def valores_distintos(picos) -> np.ndarray:
    """Valores distintos de los picos, de mayor a menor (redondeados a 1e-6 para que 0.01 m no dé falsos distintos)."""
    return np.unique(np.round(np.asarray(picos, dtype=float), 6))[::-1]


def rejilla(picos, K_: int = K) -> Rejilla:
    p = np.round(np.asarray(picos, dtype=float), 6)
    N0 = p.size
    nk = objetivos(N0, K_)
    d = valores_distintos(p)
    if d.size < 2:
        raise ValueError("Hacen falta al menos dos valores distintos de los picos")
    medios = (d[:-1] + d[1:]) / 2.0  # medios[j] entre d[j] y d[j + 1]
    cuenta = np.array([int(np.sum(p > m)) for m in medios])  # crece con j
    us, ns = [], []
    for n in nk:
        dif = np.abs(cuenta - n)
        mejor = np.flatnonzero(dif == dif.min())
        j = int(mejor[np.argmax(cuenta[mejor])])  # empate: el mayor número de picos
        us.append(float(medios[j]))
        ns.append(int(cuenta[j]))
    us_a, ns_a = np.array(us), np.array(ns)
    k = np.arange(1, K_ + 1)
    _, primero = np.unique(us_a, return_index=True)
    conservar = np.zeros(K_, bool)
    conservar[primero] = True
    repetidos = tuple(int(x) for x in k[~conservar])
    return Rejilla(k[conservar], nk[conservar], us_a[conservar], ns_a[conservar], bool(repetidos), repetidos)


# --- Ajuste en cada umbral -------------------------------------------------------------------------


@dataclass(frozen=True)
class AjusteUmbral:
    u: float
    n: int
    tasa_por_anio: float
    ajuste: gpd.AjusteGPD
    rl: dict[float, retorno.NivelRetorno]

    @property
    def valido(self) -> bool:
        return self.ajuste.valido

    @property
    def sigma_mod(self) -> float:
        return self.ajuste.sigma - self.ajuste.xi * self.u if self.valido else np.nan

    def rl_valor(self, T: float = 100.0) -> float:
        if not self.valido:
            return np.nan
        r = self.rl[T]
        return r.valor if r.existe else np.nan


def ajustar_en(picos, u: float, anios: float, periodos: Sequence[float] = PERIODOS) -> AjusteUmbral:
    p = np.asarray(picos, dtype=float)
    exc = p[p > u] - u
    a = gpd.ajustar_gpd(exc)
    lam = exc.size / anios
    rl = (
        {T: retorno.nivel_retorno(u, a.xi, a.sigma, lam, T) for T in periodos}
        if a.valido
        else {T: retorno.NivelRetorno(np.nan, "ajuste_fallido") for T in periodos}
    )
    return AjusteUmbral(float(u), int(exc.size), float(lam), a, rl)


def barrido(picos, rej: Rejilla, anios: float, periodos: Sequence[float] = PERIODOS) -> list[AjusteUmbral]:
    return [ajustar_en(picos, float(u), anios, periodos) for u in rej.u]


# --- R1 ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoRegla:
    estado: str  # CON_RESULTADO, NO_DISCRIMINO o SIN_RESULTADO
    indice: int | None  # posición de u* en la rejilla (sin repetidos)
    ventana: tuple[int, int] | None  # posiciones [i, j] aceptadas por la regla
    detalle: dict

    @property
    def con_u(self) -> bool:
        return self.indice is not None


def _estable(x: np.ndarray, tol: float = TOL_R1) -> bool:
    if np.any(~np.isfinite(x)):
        return False
    med = float(np.median(x))
    return med > 0 and (float(x.max()) - float(x.min())) / med < tol


def tramos_estables(rl100, tol: float = TOL_R1, largo_min: int = LARGO_MIN_R1) -> list[tuple[int, int]]:
    """Tramos [i, j] (posiciones, inclusivas) estables y que no se pueden extender sin romper la tolerancia."""
    x = np.asarray(rl100, dtype=float)
    n = x.size
    est = {(i, j): _estable(x[i : j + 1], tol) for i in range(n) for j in range(i + largo_min - 1, n)}

    def es(i, j):
        if i < 0 or j >= n:
            return False
        return est.get((i, j), False)

    return [(i, j) for (i, j), ok in sorted(est.items()) if ok and not es(i - 1, j) and not es(i, j + 1)]


def regla_r1(rl100, variante: str) -> ResultadoRegla:
    if variante not in ("R1a", "R1b"):
        raise ValueError(f"Variante de R1 desconocida: {variante}")
    x = np.asarray(rl100, dtype=float)
    tramos = tramos_estables(x)
    if not tramos:
        return ResultadoRegla(SIN_RESULTADO, None, None, {"tramos": []})
    if variante == "R1a":
        i, j = min(tramos, key=lambda t: (-(t[1] - t[0]), t[0]))
        largo = j - i + 1
        idx = i + (largo - 1) // 2  # con largo par, el más bajo de los dos centrales
    else:
        i, j = min(tramos, key=lambda t: (-t[1], -(t[1] - t[0]), t[0]))
        idx = i
    estado = NO_DISCRIMINO if (i == 0 and j == x.size - 1) else CON_RESULTADO
    return ResultadoRegla(estado, int(idx), (int(i), int(j)), {"tramos": [list(t) for t in tramos]})


# --- R3 ------------------------------------------------------------------------------------------


def a2_cola_derecha(excesos, xi: float, sigma: float) -> float:
    """A²_R de Sinclair et al. (1990), ec. 5 de Solari et al. (2017)."""
    x = np.sort(np.asarray(excesos, dtype=float))
    n = x.size
    z = np.clip(1.0 - gpd.supervivencia(x, xi, sigma), 0.0, 1.0 - 1e-16)
    i = np.arange(1, n + 1)
    return float(n / 2.0 - 2.0 * z.sum() - np.sum((2.0 - (2.0 * i - 1.0) / n) * np.log1p(-z)))


def redondear_replica(u: float, excesos_sim: np.ndarray, resolucion: float = RESOLUCION_M) -> np.ndarray:
    """Excesos de una réplica llevados a la retícula de los datos (C2-05): u + exceso se redondea a 0.01 m; un valor
    que queda ≤ u toma el primer valor de la retícula > u. Conserva el n."""
    h = np.round((u + np.asarray(excesos_sim, dtype=float)) / resolucion) * resolucion
    primero = (np.floor(u / resolucion + 1e-9) + 1.0) * resolucion
    h = np.where(h <= u + 1e-9, primero, h)
    return h - u


@dataclass(frozen=True)
class PValorR3:
    a2r: float
    p: float
    m: int
    B: int
    generadas: int


def p_valor_r3(
    excesos,
    ajuste: gpd.AjusteGPD,
    u: float,
    rng: np.random.Generator,
    B: int = B_R3,
    reajustar: Callable[[np.ndarray], gpd.AjusteGPD] = gpd_perfil.ajustar_rapido,
) -> PValorR3:
    """p de A²_R por bootstrap paramétrico con reajuste y réplicas redondeadas. Una réplica con ajuste fallido se
    descarta y se genera otra (como mucho 2·B generaciones; si no alcanza, p = NaN).

    ``ajuste`` es el de los datos (``ajustar_gpd``). Los reajustes usan ``gpd_perfil.ajustar_rapido`` (ADR-0034: el
    mismo estimador, más rápido); ``reajustar=gpd.ajustar_gpd`` da la versión de referencia para el piloto."""
    x = np.asarray(excesos, dtype=float)
    a0 = a2_cola_derecha(x, ajuste.xi, ajuste.sigma)
    m = validas = generadas = 0
    tope = int(FACTOR_MAX_REPLICAS_R3 * B)
    while validas < B and generadas < tope:
        generadas += 1
        e = gpd.cuantil_supervivencia(rng.random(x.size), ajuste.xi, ajuste.sigma)
        e = redondear_replica(u, e)
        ab = reajustar(e)
        if not ab.valido:
            continue
        validas += 1
        m += a2_cola_derecha(e, ab.xi, ab.sigma) >= a0
    if validas < B:
        return PValorR3(a0, np.nan, m, B, generadas)
    return PValorR3(a0, (1.0 + m) / (B + 1.0), int(m), B, generadas)


def tolerancia_r3(p_max: float, B: int = B_R3) -> float:
    return 2.0 * float(np.sqrt(max(p_max * (1.0 - p_max), PISO_VAR_R3) / B))


def regla_r3(p, B: int = B_R3) -> ResultadoRegla:
    """u* = el menor u_k con p ≥ p_máx − tolerancia; «sin resultado» si p_máx < 0.10 (o no hay ningún p)."""
    p = np.asarray(p, dtype=float)
    if not np.any(np.isfinite(p)):
        return ResultadoRegla(SIN_RESULTADO, None, None, {"p_max": np.nan})
    pmax = float(np.nanmax(p))
    tol = tolerancia_r3(pmax, B)
    det = {"p_max": pmax, "tolerancia": tol}
    if pmax < P_MIN_R3:
        return ResultadoRegla(SIN_RESULTADO, None, None, det)
    idx = int(np.flatnonzero(np.nan_to_num(p, nan=-1.0) >= pmax - tol - 1e-12)[0])
    return ResultadoRegla(CON_RESULTADO, idx, (idx, int(p.size - 1)), det)
