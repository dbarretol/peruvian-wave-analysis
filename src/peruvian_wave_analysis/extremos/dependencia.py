"""Dependencia temporal de la serie y de su cola (contrato de la F6, §3.5).

Todo trabaja **por bloques** (un bloque = un año calendario de la rejilla completa de 3 h, con NaN en los huecos) y
devuelve piezas que se suman o se concatenan entre bloques. Así el punto y las réplicas del bootstrap por bloques
usan la misma función (contrato §3.5, S-02): el punto es la función aplicada a los años en su orden, y ningún
intervalo, racha ni desfase cruza de un bloque a otro.

Dentro de un bloque, ``cond`` marca los pasos que entran en la estimación (por ejemplo, el invierno austral). Un
intervalo entre excedencias solo se retiene si entre sus dos extremos todos los pasos están en ``cond``: los que
cruzan la pausa de verano o el borde del bloque se descartan (S-03). Un registro NaN cuenta como tiempo sin
excedencia (F03-S-09).

Componentes:

- **Índice extremal** θ con el estimador de intervalos de Ferro y Segers (2003, ec. 5), con n_int el número de
  intervalos retenidos (v5, S-19), y su **declustering automático** (§4 del artículo): los C − 1 = ⌊θ n_int⌋
  mayores intervalos separan tormentas; con empates se baja C hasta que T_(C−1) > T_(C). Equivale a declustering
  por rachas con largo T_(C): tormenta nueva si el intervalo es **mayor** que T_(C) (semántica de
  ``decluster_rachas``).
- **Rachas** de excedencias consecutivas, sin tolerancia, y **huecos entre rachas**; **corte natural** de los huecos.
- **Extremograma** ρ(k) = P(X_{t+k} > u | X_t > u) con t en ``cond`` y su **referencia estacional**; escala.
- **τ** de la anomalía estacional: ACF dentro de cada año, promediada; τ_e y τ_int.

Unidades: los intervalos se cuentan en pasos; las escalas publicadas, en horas (paso × ``PASO_H``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

PASO_H = 3.0


# --- Bloques -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bloque:
    """Un año de la rejilla completa: valores (NaN = hueco), mes y día del año de cada paso."""

    anio: int
    x: np.ndarray
    mes: np.ndarray
    dia: np.ndarray  # día del año en 1..366, con el 1 de marzo siempre en 61 (el 60 solo existe en años bisiestos)


def dia_fijo(t: pd.DatetimeIndex) -> np.ndarray:
    d = t.day_of_year.to_numpy()
    return d + ((~t.is_leap_year) & (t.month >= 3)).astype(int)


def bloques_anuales(serie: pd.Series, paso_horas: float = PASO_H) -> list[Bloque]:
    """Bloques por año calendario, en orden. La serie debe estar en su rejilla completa (huecos = NaN)."""
    t = pd.DatetimeIndex(serie.index)
    if len(t) > 1 and not np.allclose(np.diff(((t - t[0]) / pd.Timedelta(hours=1)).to_numpy(float)), paso_horas):
        raise ValueError(f"La serie no está en su rejilla completa de {paso_horas} h: marque los huecos con NaN")
    x = serie.to_numpy(dtype=float)
    anio = t.year.to_numpy()
    mes, dia = t.month.to_numpy(), dia_fijo(t)
    cortes = np.flatnonzero(np.diff(anio)) + 1
    return [
        Bloque(int(anio[i]), x[i:j], mes[i:j], dia[i:j])
        for i, j in zip(np.r_[0, cortes], np.r_[cortes, len(x)], strict=True)
    ]


def condicion(b: Bloque, meses: Sequence[int] | None) -> np.ndarray:
    """Pasos del bloque en los meses dados (todos si ``meses`` es None)."""
    return np.ones(b.x.size, bool) if meses is None else np.isin(b.mes, list(meses))


def excedencias(b: Bloque, u: float | np.ndarray) -> np.ndarray:
    """``x > u``; ``u`` escalar o por mes calendario (vector de 13, índice = mes). NaN no excede."""
    umbral = np.asarray(u, dtype=float)
    uu = umbral[b.mes] if umbral.ndim else umbral
    return np.nan_to_num(b.x, nan=-np.inf) > uu


# --- Intervalos entre excedencias e índice extremal ----------------------------------------------------


def intervalos(exc: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, int]:
    """Tiempos entre excedencias consecutivas (en pasos) retenidos dentro de ``cond`` y número de excedencias.

    Se retiene un intervalo si sus dos excedencias están en el mismo tramo continuo de ``cond``.
    """
    p = np.flatnonzero(exc & cond)
    if p.size < 2:
        return np.empty(0, dtype=int), int(p.size)
    tramo = np.cumsum(~cond)  # constante dentro de cada tramo continuo de cond
    mismo = tramo[p[1:]] == tramo[p[:-1]]
    return np.diff(p)[mismo].astype(int), int(p.size)


def theta_intervalos(T: np.ndarray) -> float:
    """Estimador de intervalos de Ferro y Segers (2003, ec. 5), con n_int = número de intervalos."""
    T = np.asarray(T, dtype=float)
    n = T.size
    if n == 0:
        return float("nan")
    if T.max() <= 2:
        th = 2.0 * T.sum() ** 2 / (n * np.sum(T**2))
    else:
        th = 2.0 * np.sum(T - 1.0) ** 2 / (n * np.sum((T - 1.0) * (T - 2.0)))
    return float(min(1.0, th))


@dataclass(frozen=True)
class Declustering:
    theta: float
    n_int: int
    n_exc: int
    C: int  # número de tormentas según la regla (C − 1 intervalos entre tormentas)
    T_C: int  # largo de racha equivalente, en pasos (tormenta nueva si el intervalo es > T_C)

    @property
    def T_C_horas(self) -> float:
        return float(self.T_C * PASO_H)


def declustering_automatico(T: np.ndarray, theta: float | None = None, n_exc: int | None = None) -> Declustering:
    """Regla de Ferro y Segers (§4): C − 1 = ⌊θ n_int⌋, bajando C mientras T_(C−1) = T_(C).

    Casos de borde: con C − 1 = 0 no hay intervalos entre tormentas y T_(C) es el mayor intervalo; con
    C − 1 ≥ n_int todos lo son y T_(C) = 0 (cada excedencia es una tormenta).
    """
    T = np.asarray(T, dtype=int)
    th = theta_intervalos(T) if theta is None else float(theta)
    n = T.size
    if n == 0 or not np.isfinite(th):
        return Declustering(th, n, n_exc if n_exc is not None else 0, 0, 0)
    s = np.sort(T)[::-1]  # s[j] = T_(j+1)
    c1 = int(np.floor(th * n + 1e-12))
    if c1 >= n:
        return Declustering(th, n, n_exc if n_exc is not None else n + 1, n + 1, 0)
    while c1 > 0 and s[c1 - 1] == s[c1]:
        c1 -= 1
    return Declustering(th, n, n_exc if n_exc is not None else n + 1, c1 + 1, int(s[c1]))


def declustering_bloques(bloques: Sequence[Bloque], u, meses: Sequence[int] | None) -> Declustering:
    """θ y T_(C) con los intervalos retenidos de todos los bloques."""
    Ts, n = [], 0
    for b in bloques:
        T, k = intervalos(excedencias(b, u), condicion(b, meses))
        Ts.append(T)
        n += k
    T = np.concatenate(Ts) if Ts else np.empty(0, int)
    return declustering_automatico(T, n_exc=n)


# --- Rachas y huecos entre rachas ----------------------------------------------------------------------


def rachas(exc: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rachas de excedencias consecutivas dentro de ``cond`` y huecos entre rachas del mismo tramo.

    Devuelve (duraciones en pasos, si la racha está cortada por el borde del tramo o del bloque, huecos en pasos).
    El hueco es la diferencia entre la última excedencia de una racha y la primera de la siguiente (T_i > 1). Una
    racha está cortada si el paso vecino fuera del tramo es una excedencia o si toca el borde del bloque.
    """
    e = exc & cond
    n = e.size
    if not e.any():
        return np.empty(0, int), np.empty(0, bool), np.empty(0, int)
    d = np.diff(np.r_[0, e.astype(int), 0])
    ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    dur = fin - ini + 1
    antes, despues = ini - 1, fin + 1
    cortada = (antes < 0) | (despues >= n)
    a_ok, d_ok = np.clip(antes, 0, n - 1), np.clip(despues, 0, n - 1)
    cortada |= (antes >= 0) & ~cond[a_ok] & exc[a_ok]
    cortada |= (despues < n) & ~cond[d_ok] & exc[d_ok]
    tramo = np.cumsum(~cond)
    mismo = tramo[ini[1:]] == tramo[fin[:-1]]
    huecos = (ini[1:] - fin[:-1])[mismo]
    return dur, cortada, huecos.astype(int)


def huecos_bloques(bloques: Sequence[Bloque], u, meses: Sequence[int] | None) -> np.ndarray:
    hs = [rachas(excedencias(b, u), condicion(b, meses))[2] for b in bloques]
    return np.concatenate(hs) if hs else np.empty(0, int)


@dataclass(frozen=True)
class Corte:
    corte_h: float | None  # None: sin corte natural
    n_modas: int
    frac_segunda: float
    mas_de_dos_modas: bool
    n_huecos: int


def corte_natural(huecos_h: np.ndarray, frac_min: float = 0.10, n_rejilla: int = 2048) -> Corte:
    """Corte natural de los huecos entre rachas (contrato §3.5, punto 3).

    Densidad de log(hueco) con núcleo gaussiano y ancho de Silverman. Hay corte si hay ≥ 2 modas y la cuenca de la
    segunda (de izquierda a derecha, entre el primer y el segundo mínimo) reúne ≥ ``frac_min`` de los huecos. El
    corte es el primer mínimo a la derecha de la primera moda, en horas.
    """
    h = np.asarray(huecos_h, dtype=float)
    h = h[np.isfinite(h) & (h > 0)]
    if h.size < 3 or np.ptp(np.log(h)) == 0:
        return Corte(None, int(h.size > 0), 0.0, False, int(h.size))
    z = np.log(h)
    kde = gaussian_kde(z, bw_method="silverman")
    bw = float(np.sqrt(kde.covariance[0, 0]))
    g = np.linspace(z.min() - 3 * bw, z.max() + 3 * bw, n_rejilla)
    f = kde(g)
    df = np.diff(f)
    maximos = [i for i in range(1, f.size - 1) if df[i - 1] > 0 and df[i] <= 0]
    minimos = [i for i in range(1, f.size - 1) if df[i - 1] < 0 and df[i] >= 0]
    n_modas = len(maximos)
    if n_modas < 2:
        return Corte(None, n_modas, 0.0, False, int(h.size))
    m1 = maximos[0]
    mins = [i for i in minimos if i > m1]
    c1 = mins[0]
    c2 = next((i for i in mins if i > maximos[1]), None)
    sup = np.inf if c2 is None else g[c2]
    frac = float(np.mean((z > g[c1]) & (z <= sup)))
    corte = float(np.exp(g[c1])) if frac >= frac_min else None
    return Corte(corte, n_modas, frac, n_modas > 2, int(h.size))


# --- Extremograma --------------------------------------------------------------------------------------


@dataclass
class ConteosExtremograma:
    """Piezas sumables entre bloques. ``N[k, m]``: pares (t, t+k) condicionantes con mes(t+k) = m."""

    num: np.ndarray  # (kmax+1,) Σ exc_t cond_t exc_{t+k}
    N: np.ndarray  # (kmax+1, 13)
    exc_mes: np.ndarray  # (13,) excedencias por mes (pasos válidos)
    val_mes: np.ndarray  # (13,) pasos válidos por mes

    def __add__(self, o: ConteosExtremograma) -> ConteosExtremograma:
        return ConteosExtremograma(self.num + o.num, self.N + o.N, self.exc_mes + o.exc_mes, self.val_mes + o.val_mes)


def conteos_extremograma(b: Bloque, u, cond: np.ndarray, kmax: int) -> ConteosExtremograma:
    exc = excedencias(b, u)
    valido = np.isfinite(b.x)
    n = b.x.size
    t = np.flatnonzero(exc & cond)
    k = np.arange(kmax + 1)
    j = t[:, None] + k[None, :]
    dentro = j < n
    jj = np.where(dentro, j, 0)
    ok = dentro & valido[jj]
    num = np.sum(ok & exc[jj], axis=0).astype(float)
    N = np.zeros((kmax + 1, 13))
    meses = np.where(ok, b.mes[jj], 0)
    for m in range(1, 13):
        N[:, m] = np.sum(meses == m, axis=0)
    exc_mes = np.bincount(b.mes[exc & valido], minlength=13).astype(float)
    val_mes = np.bincount(b.mes[valido], minlength=13).astype(float)
    return ConteosExtremograma(num, N, exc_mes, val_mes)


@dataclass(frozen=True)
class Extremograma:
    rho: np.ndarray  # ρ(k), k = 0..kmax pasos
    ref: np.ndarray  # referencia estacional por k
    escala_h: float  # primer cruce (NaN si no cruza hasta kmax)
    ultimo_2x_h: float  # último k con ρ(k) > 2 × ref (descriptivo; NaN si ninguno)
    factor: float


def extremograma(c: ConteosExtremograma, factor: float = 1.5, media_h: float = 24.0) -> Extremograma:
    den = c.N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = c.num / den
        p_mes = np.where(c.val_mes > 0, c.exc_mes / c.val_mes, 0.0)
        ref = (c.N @ p_mes) / den
    return Extremograma(rho, ref, *escala_extremograma(rho, ref, factor, media_h), factor)


def escala_extremograma(rho, ref, factor: float = 1.5, media_h: float = 24.0) -> tuple[float, float]:
    """Primer k ≥ 1 en que la media móvil centrada de ``media_h`` de ρ cumple ρ ≤ factor × ref.

    La ventana se trunca a los desfases 1..kmax: ρ(0) = 1 nunca entra (v5, S-18).
    """
    rho, ref = np.asarray(rho, float), np.asarray(ref, float)
    kmax = rho.size - 1
    w = int(round(media_h / PASO_H / 2))
    escala = float("nan")
    for k in range(1, kmax + 1):
        a, b = max(1, k - w), min(kmax, k + w)
        if np.nanmean(rho[a : b + 1]) <= factor * ref[k]:
            escala = k * PASO_H
            break
    sobre = np.flatnonzero(rho[1:] > 2.0 * ref[1:])
    ultimo = float((sobre[-1] + 1) * PASO_H) if sobre.size else float("nan")
    return escala, ultimo


def extremograma_bloques(
    bloques: Sequence[Bloque], u, meses: Sequence[int] | None, kmax: int = 120, factor: float = 1.5
) -> Extremograma:
    c = None
    for b in bloques:
        ci = conteos_extremograma(b, u, condicion(b, meses), kmax)
        c = ci if c is None else c + ci
    return extremograma(c, factor)


# --- τ de la anomalía estacional -----------------------------------------------------------------------


def _suavizar_circular(v: np.ndarray, ancho: int) -> np.ndarray:
    k = ancho // 2
    ext = np.r_[v[-k:], v, v[:k]]
    return np.convolve(ext, np.ones(ancho), mode="valid")


def climatologia(bloques: Sequence[Bloque], ventana_dias: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación por día del año (índice 1..366), suavizadas con una ventana circular de ``ventana_dias``."""
    s1, s2, n = np.zeros(367), np.zeros(367), np.zeros(367)
    for b in bloques:
        ok = np.isfinite(b.x)
        s1 += np.bincount(b.dia[ok], weights=b.x[ok], minlength=367)
        s2 += np.bincount(b.dia[ok], weights=b.x[ok] ** 2, minlength=367)
        n += np.bincount(b.dia[ok], minlength=367)
    S1, S2, N = (_suavizar_circular(a[1:], ventana_dias) for a in (s1, s2, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        media = S1 / N
        sd = np.sqrt(np.maximum(S2 / N - media**2, 0.0))
    return np.r_[np.nan, media], np.r_[np.nan, sd]


def acf_anual(a: np.ndarray, kmax: int) -> np.ndarray:
    """ACF de una anomalía dentro de un año (NaN = hueco), sin quitar la media del año. Pares con ambos datos."""
    m = np.isfinite(a).astype(float)
    a0 = np.where(m > 0, a, 0.0)
    n = a.size
    L = 1 << int(np.ceil(np.log2(2 * n)))
    fa, fm = np.fft.rfft(a0, L), np.fft.rfft(m, L)
    sxx = np.fft.irfft(fa * np.conj(fa), L)[: kmax + 1]
    nn = np.round(np.fft.irfft(fm * np.conj(fm), L)[: kmax + 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(nn > 0, sxx / nn, np.nan)
    return c / c[0]


@dataclass(frozen=True)
class Tau:
    acf: np.ndarray
    tau_e_h: float
    tau_int_h: float


def tau_bloques(bloques: Sequence[Bloque], kmax: int = 720, ventana_dias: int = 31) -> Tau:
    """τ_e (primer desfase con ACF < 1/e) y τ_int (integral hasta el primer cruce por cero), en horas.

    Anomalía respecto del ciclo anual medio de **los bloques recibidos**; ACF dentro de cada año y promediada.
    """
    media, sd = climatologia(bloques, ventana_dias)
    acfs = []
    for b in bloques:
        if b.x.size <= kmax:
            continue  # un año más corto que el desfase máximo no aporta los desfases largos
        with np.errstate(invalid="ignore", divide="ignore"):
            a = (b.x - media[b.dia]) / sd[b.dia]
        acfs.append(acf_anual(a, kmax))
    r = np.nanmean(np.vstack(acfs), axis=0)
    bajo = np.flatnonzero(r < np.exp(-1.0))
    tau_e = float(bajo[0] * PASO_H) if bajo.size else float("nan")
    cero = np.flatnonzero(r <= 0)
    tau_int = float(np.trapezoid(r[: cero[0] + 1]) * PASO_H) if cero.size else float("nan")
    return Tau(r, tau_e, tau_int)
