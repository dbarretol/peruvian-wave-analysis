"""Efecto relevante: el escalón δ_rel que mueve un nivel de retorno un 2 % (contrato de la F4, §3.5; ADR-0006).

**Modelo provisional**, solo para la potencia: umbral u = P99 del registro completo, eventos por rachas con IET de
48 h (``extremos.decluster``), GPD de los excesos (``extremos.gpd``) y RL_T con ``extremos.retorno`` y la tasa de
eventos por año del registro.

**Umbral y eventos fijos (v6, F04-S-09).** Se fijan con la serie sin escalar. Al escalar un lado por (1 + δ) solo
cambia la magnitud de los picos de ese lado. Un pico que, escalado, queda bajo u se toma con exceso 0: así el
número de eventos no cambia y RL(δ) es continua.

**δ_rel** es el menor |δ| de la rejilla con |RL_T(δ)/RL_T(0) − 1| ≥ 2 % en alguno de los períodos de retorno,
recorriendo los dos lados y los dos signos. Si nada llega al 2 %, el contraste queda **sin raíz**.

El RL del modelo provisional no se publica: solo δ_rel, su lado, su signo y su período de retorno.

**Defecto K-03 y versión corregida (contrato §21, posterior a ver datos).** Con u y eventos fijos, volver a ajustar
la GPD a los picos escalados no responde al escalado: los excesos 0 (lado negativo) y el hueco entre 0 y el menor
exceso (lado positivo) cambian la forma ajustada. Escalar todos los picos un −3 % sube el RL100 un 65 %. El δ_rel
original se publica rotulado «no interpretable». La versión corregida (``niveles_mezcla``) ajusta la GPD **una sola
vez**, a los picos sin escalar, y escala la distribución de cada pico: la tasa de excedencia es
R(x) = (λ/N)·Σᵢ S(x/fᵢ), con S la supervivencia del pico sin escalar (1 bajo u), y RL_T es la raíz de R(x) = 1/T.
Es monótona en δ y, con un mismo factor f en todos los picos, RL_T(f) = f·RL_T(1) exactamente.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..extremos import decluster, gpd, retorno


@dataclass(frozen=True)
class ModeloProvisional:
    u: float
    picos: np.ndarray  # magnitud de cada pico
    t_picos: pd.DatetimeIndex
    anios: float  # años de registro (para la tasa)


def modelo_provisional(hs: pd.Series, cuantil: float, iet_h: float, paso_h: float = 3.0) -> ModeloProvisional:
    u = float(np.nanquantile(hs.to_numpy(float), cuantil))
    p = decluster.decluster_rachas(hs.to_frame("Hs"), u, iet_h, paso_h)
    validos = np.isfinite(hs.to_numpy(float)).sum()
    anios = validos * paso_h / (24.0 * 365.25)
    return ModeloProvisional(u, p["Hs"].to_numpy(float), pd.DatetimeIndex(p.index), float(anios))


def niveles(modelo: ModeloProvisional, factores: np.ndarray, periodos: Sequence[float]) -> np.ndarray:
    """RL_T con cada pico multiplicado por su factor."""
    exc = np.maximum(modelo.picos * factores - modelo.u, 0.0)
    aj = gpd.ajustar_gpd(exc)
    if not aj.valido:
        return np.full(len(periodos), np.nan)
    tasa = exc.size / modelo.anios
    return np.array([retorno.nivel_retorno(modelo.u, aj.xi, aj.sigma, tasa, T).valor for T in periodos])


def niveles_mezcla(
    modelo: ModeloProvisional, factores: np.ndarray, periodos: Sequence[float], ajuste=None
) -> np.ndarray:
    """RL_T de la mezcla de picos escalados, sin reajustar la GPD (K-03, contrato §21). ``ajuste``: la GPD de los
    excesos sin escalar; si falta, se ajusta aquí."""
    exc = modelo.picos - modelo.u
    aj = ajuste if ajuste is not None else gpd.ajustar_gpd(np.maximum(exc, 0.0))
    if not aj.valido:
        return np.full(len(periodos), np.nan)
    f = np.broadcast_to(np.asarray(factores, float), exc.shape).astype(float)
    lam = exc.size / modelo.anios

    def tasa(x: float) -> float:
        return float(lam * np.mean(gpd.supervivencia(x / f - modelo.u, aj.xi, aj.sigma)))

    fin = modelo.u + (-aj.sigma / aj.xi if aj.xi < 0 else np.inf)
    out = []
    for T in periodos:
        objetivo = 1.0 / T
        lo = modelo.u * float(f.min())
        if tasa(lo) <= objetivo:
            out.append(np.nan)
            continue
        if np.isfinite(fin):
            hi = fin * float(f.max())
        else:
            hi = modelo.u * float(f.max()) * 2.0
            while tasa(hi) > objetivo:
                hi *= 2.0
        out.append(float(brentq(lambda x, o=objetivo: tasa(x) - o, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500)))
    return np.array(out)


def ajuste_base(modelo: ModeloProvisional):
    """GPD de los excesos sin escalar, para ``niveles_mezcla``."""
    return gpd.ajustar_gpd(np.maximum(modelo.picos - modelo.u, 0.0))


@dataclass(frozen=True)
class DeltaRel:
    delta: float | None  # None: sin raíz
    lado: str | None
    periodo: float | None
    cambio: float | None  # ΔRL relativo en ese punto
    por_lado: dict  # lado → (menor δ con |ΔRL| ≥ cambio, período) o None

    @property
    def sin_raiz(self) -> bool:
        return self.delta is None


def rejilla(lo: float, hi: float, paso: float) -> np.ndarray:
    n = int(round((hi - lo) / paso))
    return np.round(lo + paso * np.arange(n + 1), 10)


def raiz_en_rejilla(
    rl_de_delta: Callable[[float], np.ndarray], deltas: np.ndarray, cambio: float
) -> tuple[float, int, float] | None:
    """Menor |δ| de la rejilla con |RL(δ)/RL(0) − 1| ≥ cambio en algún período. Devuelve (δ, índice del período,
    cambio relativo) o None. Recorre |δ| creciente; con empate de |δ|, el δ negativo primero."""
    base = np.asarray(rl_de_delta(0.0), float)
    orden = sorted((d for d in deltas if d != 0), key=lambda d: (abs(d), d))
    for d in orden:
        r = np.asarray(rl_de_delta(float(d)), float) / base - 1.0
        ok = np.isfinite(r) & (np.abs(r) >= cambio - 1e-12)
        if ok.any():
            j = int(np.flatnonzero(ok)[np.argmax(np.abs(r[ok]))])
            return float(d), j, float(r[j])
    return None


def delta_relevante(
    modelo: ModeloProvisional,
    lados: dict[str, np.ndarray],
    periodos: Sequence[float],
    cambio: float,
    deltas: np.ndarray,
    niveles_fn: Callable = niveles,
) -> DeltaRel:
    """``lados``: nombre → máscara booleana sobre los picos del modelo (qué picos escala ese lado).
    ``niveles_fn(modelo, factores, periodos)``: ``niveles`` (original) o ``niveles_mezcla`` (corregido, K-03)."""
    por_lado = {}
    mejor = None
    for nombre, mascara in lados.items():
        m = np.asarray(mascara, bool)

        def rl(d: float, m=m) -> np.ndarray:
            return niveles_fn(modelo, np.where(m, 1.0 + d, 1.0), periodos)

        r = raiz_en_rejilla(rl, deltas, cambio) if m.any() else None
        por_lado[nombre] = None if r is None else {"delta": r[0], "periodo": periodos[r[1]], "cambio": r[2]}
        if r is not None and (mejor is None or abs(r[0]) < abs(mejor[1][0])):
            mejor = (nombre, r)
    if mejor is None:
        return DeltaRel(None, None, None, None, por_lado)
    nombre, (d, j, c) = mejor
    return DeltaRel(d, nombre, float(periodos[j]), c, por_lado)
