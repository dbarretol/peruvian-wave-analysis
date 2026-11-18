"""Motor de la prueba y de la potencia de la F5 (contrato v6, §3.4, §3.5, §4 y §5). Sin valores observados.

**Lado.** Las colocaciones de un lado de una comparación (``LadoSim``): el modelo en cada una, su mes, su bloque
(año y mes), su unidad de muestreo (pasada o día), su número de puntos o registros, su tramo de calibración, su
época y su año.

**Generador (§5, simétrico desde §20).** Cópula que conserva la marginal del modelo en el lado (con peso mensual):
m* y o* = F_m⁻¹(Φ(a·z_L + √(1 − a²)·z)), con z independiente para cada columna, z = √ρ·η_unidad + √(1 − ρ)·ε, y z_L
las puntuaciones normales del modelo real. Después, en o*, el sesgo inyectado, la calibración
(un error por tramo, por época y por año de boya, con su forma lineal a·Hs + b) y el ruido de muestreo:
0.10 m + 5 % de Hs por punto de 1 Hz (σ/√n en la superobservación) o 6 % por registro de boya (σ/√k).
``a`` se calibra para que el SI de o* frente a m* sea el supuesto.

**Estadístico (§3.4).** B(q) = log(Q_modelo/Q_obs) con cuantiles con peso mensual (P50, P90, P99, P99.9) y la media
de máximos mensuales; descontado con SIMEX (λ ∈ {0.5, 1, 1.5, 2}, extrapolación cuadrática a λ = −1).

**IC y prueba (§3.4, §4).** Bootstrap por bloques de un mes dentro de cada lado; la corrección de SIMEX se suma a
cada réplica como un desplazamiento fijo, y los errores de calibración entran en cada réplica **linealizados** sobre
el estadístico (log Q* ≈ log Q + a + b/Q + …), con un error por tramo común a los lados que lo comparten. max-T con t
studentizado sobre los estadísticos estimables.

**Potencia por desplazamiento.** Las pruebas de V1, V2 y V3a son de desplazamiento: un sesgo b en un lado desplaza
todos sus B(q) en b. Cada réplica se simula con b = 0 y se guarda (Δ̂, EE, valor crítico); la potencia para
cualquier b es la fracción de réplicas con max|Δ̂ + b|/EE > crítico. En V3b, B(P99) − B(P50) se desplaza en el
efecto. La aproximación (el ruido no se reescala con b) se comprueba con inyección explícita en la etapa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from ..homogeneidad import cuantiles as Q
from . import config as C
from . import escalon as E

IDX = {k: i for i, k in enumerate(E.ESTADISTICOS)}  # P50, P90, P99, P99.9, max
MIN_UNIDADES_SOBRE = 10  # §3.4: 10 pasadas o días distintos por encima del cuantil


@dataclass
class LadoSim:
    nombre: str
    fuente: str  # «altimetría» o la estación de la boya
    t: pd.DatetimeIndex
    m: np.ndarray
    unidad: np.ndarray  # códigos 0..U−1
    npts: np.ndarray  # puntos de la superobservación o registros del paso
    tramo: np.ndarray  # clave de calibración (altimetría) o "" (boyas)
    epoca: np.ndarray  # clave de la red de referencia (altimetría) o ""
    anio: np.ndarray  # clave «boya_año» (boyas) o ""
    boot: E.LadoBoot = field(repr=False, default=None)
    w: np.ndarray = field(repr=False, default=None)
    _orden_m: np.ndarray = field(repr=False, default=None)
    _F_m: np.ndarray = field(repr=False, default=None)

    @classmethod
    def desde(cls, nombre: str, df: pd.DataFrame) -> LadoSim:
        df = df.sort_values("tiempo", kind="stable").reset_index(drop=True)
        t = pd.DatetimeIndex(df["tiempo"])
        alt = "mision" in df and df["mision"].notna().all()
        fuente = "altimetría" if alt else str(df["boya"].iloc[0])
        _, unidad = np.unique(df["unidad"].astype(str).to_numpy(), return_inverse=True)
        npts = (df["n"] if alt else df["k"]).to_numpy(float)
        tramo = df["tramo_cal"].to_numpy(object) if alt else np.full(len(df), "", object)
        epoca = (
            np.where(t.year < 2000, "red_1985_1992", "red_2011_2026").astype(object)
            if alt
            else np.full(len(df), "", object)
        )
        anio = np.full(len(df), "", object) if alt else np.array([f"{fuente}_{a}" for a in t.year], object)
        m = df["hs_mod"].to_numpy(float)
        L = cls(nombre, fuente, t, m, unidad, npts, tramo, epoca, anio)
        L.boot = E.LadoBoot.desde(t, m)
        L.w = Q.pesos_mensuales(t)
        L._orden_m = np.argsort(m, kind="stable")
        Fw = np.cumsum(L.w[L._orden_m])
        L._F_m = Fw / Fw[-1]
        return L

    @property
    def n(self) -> int:
        return len(self.m)

    def u_modelo(self) -> np.ndarray:
        """F_m(m) con el punto medio de cada salto (sin 0 ni 1)."""
        u = np.empty(self.n)
        Fw = self._F_m
        pw = self.w[self._orden_m] / self.w.sum()
        u[self._orden_m] = Fw - pw / 2
        return np.clip(u, 1e-9, 1 - 1e-9)

    def F_inv(self, u: np.ndarray) -> np.ndarray:
        xs = self.m[self._orden_m]
        pw = self.w[self._orden_m] / self.w.sum()
        return np.interp(u, self._F_m - pw / 2, xs)


# --- Generador -------------------------------------------------------------------------------------------


def _z(L: LadoSim, rho: float, rng: np.random.Generator) -> np.ndarray:
    eta = rng.standard_normal(int(L.unidad.max()) + 1)[L.unidad]
    return np.sqrt(rho) * eta + np.sqrt(1 - rho) * rng.standard_normal(L.n)


def par(L: LadoSim, a: float, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(m*, o*): **generador simétrico** (contrato §20). Las dos columnas son perturbaciones independientes de la
    misma variable latente, las puntuaciones normales del modelo real z_L, con la misma estructura de pasadas:
    m* = F_m⁻¹(Φ(a·z_L + √(1 − a²)·z₁)), o* = F_m⁻¹(Φ(a·z_L + √(1 − a²)·z₂)). Con b = 0 y antes de la calibración y
    el ruido son intercambiables, así que E[B] = 0 para cualquier estadístico. La correlación latente entre ellas es a².
    """
    zl = st.norm.ppf(L.u_modelo())
    c = np.sqrt(1 - a * a)
    m = L.F_inv(st.norm.cdf(a * zl + c * _z(L, rho, rng)))
    o = L.F_inv(st.norm.cdf(a * zl + c * _z(L, rho, rng)))
    return m, o


def si(L: LadoSim, o: np.ndarray, m: np.ndarray | None = None) -> float:
    """SI = RMS(o − m) / media(m), con peso mensual."""
    m = L.m if m is None else m
    w = L.w / L.w.sum()
    return float(np.sqrt(np.sum(w * (o - m) ** 2)) / np.sum(w * m))


def calibrar_r(L: LadoSim, si_objetivo: float, rho: float, semilla: int, n_rep: int = 20) -> float:
    """a tal que el SI medio entre o* y m* de ``par`` sea el supuesto (bisección con números aleatorios fijos)."""

    def si_de(a):
        rng = np.random.default_rng(semilla)
        return np.mean([si(L, *par(L, a, rho, rng)[::-1]) for _ in range(n_rep)])

    lo, hi = 0.0, 1.0
    if si_de(hi - 1e-9) > si_objetivo:
        return 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if si_de(mid) > si_objetivo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sigma_ruido(L: LadoSim, o: np.ndarray, factor: float) -> np.ndarray:
    if L.fuente == "altimetría":
        a, b = C.RUIDO_1HZ
        return factor * (a + b * np.abs(o)) / np.sqrt(L.npts)
    return factor * C.RUIDO_BOYA_REL * np.abs(o) / np.sqrt(L.npts)


# --- Calibración ---------------------------------------------------------------------------------------


def _claves(L: LadoSim, lado_boya: bool) -> dict[str, np.ndarray]:
    """Fracción (con peso mensual) de cada clave de error sistemático en el lado."""
    w = L.w / L.w.sum()
    out = {}
    for arr in (L.tramo, L.epoca, L.anio):
        for k in np.unique(arr):
            if k != "":
                out[k] = float(w[arr == k].sum())
    if lado_boya and L.fuente != "altimetría":
        out[f"lado_{L.nombre}"] = 1.0
    return out


def _sigma_clave(k: str, factor: float) -> tuple[float, float, float]:
    """(σ_a, σ_b en m, σ relativa adicional) de una clave de error sistemático."""
    if k in C.CALIBRACION:
        a, b = C.CALIBRACION[k]
        return factor * a, factor * b, factor * C.SIGMA_ESTADO_MAR
    if k.startswith("red_"):
        return 0.0, 0.0, factor * C.SIGMA_RED_EPOCA
    if k.startswith("lado_"):
        return 0.0, 0.0, factor * C.SIGMA_BOYA_LADO
    return 0.0, 0.0, factor * C.SIGMA_BOYA_ANIO  # año de boya


def aplicar_calibracion(L: LadoSim, o: np.ndarray, errores: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """o·(1 + a)·(1 + e) + b por colocación, con los errores de su tramo, su época y su año (los «verdaderos»)."""
    out = o.copy()
    for arr in (L.tramo, L.epoca, L.anio):
        for k in np.unique(arr):
            if k == "" or k not in errores:
                continue
            a, b, e = errores[k]
            s = arr == k
            out[s] = out[s] * (1 + a) * (1 + e) + b
    k = f"lado_{L.nombre}"
    if k in errores:
        out *= 1 + errores[k][2]
    return out


def sortear_errores(claves, factor: float, rng: np.random.Generator, size: int | None = None) -> dict:
    """Un error (a, b, e) por clave; con ``size``, arreglos de ``size`` réplicas."""
    out = {}
    for k in sorted(claves):
        sa, sb, se = _sigma_clave(k, factor)
        out[k] = tuple(rng.normal(0.0, s, size) if s > 0 else (np.zeros(size) if size else 0.0) for s in (sa, sb, se))
    return out


def termino_calibracion(L: LadoSim, Qobs: np.ndarray, errores: dict, lado_boya: bool) -> np.ndarray:
    """(B, 5): log Q* − log Q por réplica, linealizado: Σ_k f_k·(a_k + b_k/Q + e_k)."""
    fr = _claves(L, lado_boya)
    B = len(next(iter(errores.values()))[0]) if errores else 0
    out = np.zeros((B, len(E.ESTADISTICOS)))
    for k, f in fr.items():
        a, b, e = errores[k]
        out += f * (np.asarray(a)[:, None] + np.asarray(b)[:, None] / Qobs[None, :] + np.asarray(e)[:, None])
    return out


# --- Estadísticos del lado -----------------------------------------------------------------------------


def estadisticos_filas(L: LadoSim, X: np.ndarray) -> np.ndarray:
    """(R, 5) estadísticos con peso mensual de cada fila de X (R, n)."""
    R = X.shape[0]
    o = np.argsort(X, axis=1, kind="stable")
    xs = np.take_along_axis(X, o, axis=1)
    ws = L.w[o]
    F = np.cumsum(ws, axis=1)
    F /= F[:, -1:]
    out = np.empty((R, 5))
    for j, q in enumerate([C.CUANTILES[k] for k in E.ESTADISTICOS[:4]]):
        i = np.minimum((F < q - 1e-12).sum(axis=1), L.n - 1)
        out[:, j] = xs[np.arange(R), i]
    b = L.boot.bloque
    ob = np.argsort(b, kind="stable")
    inicios = np.r_[0, np.flatnonzero(np.diff(b[ob])) + 1]
    maxb = np.maximum.reduceat(X[:, ob], inicios, axis=1)
    out[:, 4] = maxb.mean(axis=1) if L.boot.K >= E.MIN_MESES_MAX else np.nan
    return out


def simex(
    L: LadoSim, o: np.ndarray, sigma: np.ndarray, rng: np.random.Generator, R: int = C.SIMEX_REPLICAS
) -> np.ndarray:
    """(5,) estadísticos de o descontado el ruido (SIMEX, extrapolación cuadrática a λ = −1)."""
    lam = np.r_[0.0, C.SIMEX_LAMBDAS]
    vals = [estadisticos_filas(L, o[None, :])[0]]
    for la in C.SIMEX_LAMBDAS:
        X = o[None, :] + np.sqrt(la) * sigma[None, :] * rng.standard_normal((R, L.n))
        vals.append(estadisticos_filas(L, X).mean(axis=0))
    V = np.vstack(vals)  # (5 λ, 5 estadísticos)
    A = np.vander(lam, 3)  # λ², λ, 1
    coef, *_ = np.linalg.lstsq(A, V, rcond=None)
    return np.array([1.0, -1.0, 1.0]) @ coef


def estimables(L: LadoSim, o: np.ndarray, Qobs: np.ndarray) -> np.ndarray:
    """(5,) bool: ≥ 10 unidades distintas por encima de cada cuantil; máximos con ≥ 12 meses."""
    ok = np.zeros(5, bool)
    for j in range(4):
        ok[j] = np.unique(L.unidad[o > Qobs[j]]).size >= MIN_UNIDADES_SOBRE
    ok[4] = L.boot.K >= E.MIN_MESES_MAX
    return ok


@dataclass
class Medida:
    """B(q) de un lado: puntual descontado, réplicas bootstrap con SIMEX (sin calibración) y estimabilidad."""

    B: np.ndarray  # (5,)
    reps: np.ndarray  # (Bboot, 5), sin el término de calibración
    estimable: np.ndarray  # (5,)
    Qobs: np.ndarray  # (5,) descontados


def medir(
    L: LadoSim, o: np.ndarray, sigma: np.ndarray, Bboot: int, rng: np.random.Generator, m: np.ndarray | None = None
) -> Medida:
    """B(q) = log(Q_m/Q_o) del lado; ``m`` es la columna del modelo (m* en la simulación; el modelo real si falta)."""
    L.boot.series = np.vstack([L.m if m is None else m, o])
    punt = E.puntual(L.boot)  # (2, 5) bruto
    reps = E.replicas(L.boot, Bboot, rng)  # (Bboot, 2, 5)
    Qs = simex(L, o, sigma, rng)
    desplaz = np.log(punt[1]) - np.log(Qs)  # log Q bruto − log Q descontado
    B = np.log(punt[0]) - np.log(Qs)
    Bb = np.log(reps[:, 0, :]) - np.log(reps[:, 1, :]) + desplaz[None, :]
    return Medida(B, Bb, estimables(L, o, Qs), Qs)


def con_calibracion(L: LadoSim, med: Medida, errores_boot: dict, lado_boya: bool = False) -> np.ndarray:
    """Réplicas de B con el error de calibración linealizado de cada réplica."""
    return med.reps + termino_calibracion(L, med.Qobs, errores_boot, lado_boya)


# --- Pruebas -------------------------------------------------------------------------------------------


@dataclass
class Prueba:
    """Lo que hace falta para la potencia por desplazamiento: Δ̂, EE y críticos de max-T."""

    est: np.ndarray  # (s,) estadístico por cuantil (NaN si no estimable)
    ee: np.ndarray  # (s,)
    critico: dict  # α → valor crítico de max|t|
    p: float


def max_t(est: np.ndarray, reps: np.ndarray, usar: np.ndarray, alfas=(C.ALFA_FUENTE, C.ALFA_FAMILIA)) -> Prueba:
    est = np.where(usar, est, np.nan)
    ee = np.nanstd(reps, axis=0, ddof=1)
    ok = usar & np.isfinite(est) & (ee > 0)
    if not ok.any():
        return Prueba(est, ee, {a: np.nan for a in alfas}, np.nan)
    T = np.max(np.abs(est[ok] / ee[ok]))
    Tstar = np.nanmax(np.abs((reps[:, ok] - est[ok]) / ee[ok]), axis=1)
    p = (1 + np.sum(Tstar >= T)) / (len(Tstar) + 1)
    return Prueba(est, ee, {a: float(np.quantile(Tstar, 1 - a)) for a in alfas}, float(p))


def rechaza(pr: Prueba, desplazamiento, alfa: float, factor: float = 1.0) -> bool:
    """max|est + desplazamiento|/EE > factor·crítico (desplazamiento escalar o por estadístico).

    ``factor`` ≠ 1 solo si el error de tipo I quedó fuera de rango y se recalibró el valor crítico (§5).
    """
    ok = np.isfinite(pr.est) & (pr.ee > 0)
    if not ok.any() or not np.isfinite(pr.critico[alfa]):
        return False
    d = np.broadcast_to(np.asarray(desplazamiento, float), pr.est.shape)
    return bool(np.max(np.abs((pr.est[ok] + d[ok]) / pr.ee[ok])) > factor * pr.critico[alfa])
