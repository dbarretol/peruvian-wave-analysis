"""Detector bilateral de picos y saltos, resolución automática, dipolo y máximos (contrato de la F3, §3.4, §3.5, §3.10).

Detector: un candidato se abre en t si |ΔHs(t)| > k·s y Hs **vuelve** en j ≤ w pasos, es decir, la reversión
acumulada −signo(ΔHs(t))·[Hs(t+j) − Hs(t)] supera k·s. La excursión son los registros t … t+j−1 (el menor j). El
salto que la cierra no abre otra. La marca va al **pico**: el registro de la excursión con mayor |Hs − Hs(t−1)|.
La regla es simétrica (subidas y caídas). s = 1.4826·MAD(ΔHs) de la serie que se analiza.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

W = 2  # pasos máximos de la excursión (§3.4)
FRACCION_VUELTA = 0.5  # vuelve al menos al 50 % del nivel previo
SALTO_TM02_S = 1.0  # salto espectral conjunto: Tm02 salta ≥ 1 s
FRACCION_VECINAS = 0.25  # las vecinas comparten si su salto mediano es ≥ 25 % del del nodo
DIPOLO_FRACCION = 0.25  # dipolo: mediana de ΔHs de las vecinas menor que k·s/4 o de signo opuesto
PERCENTIL_COLA = 95.0  # protección de cola (F03-S-04)
TOPE_TOTAL = 47  # descartes automáticos (§3.4, P12)
TOPE_ANIO = 5
TOPE_REVISION = 100  # candidatos de k_rev con pico ≥ P95 (F03-S-12)
N_MAXIMOS = 20  # §3.10
SEPARACION_MAXIMOS = pd.Timedelta(hours=72)


def escala_s(hs: np.ndarray | pd.Series) -> float:
    """s = 1.4826·MAD(ΔHs) sobre los ΔHs finitos."""
    d = np.diff(np.asarray(hs, dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


@dataclass(frozen=True)
class Excursion:
    inicio: int  # primer registro de la excursión (posición)
    fin: int  # último registro de la excursión
    pico: int
    salto: float  # Hs(pico) − Hs(inicio − 1), con signo

    @property
    def duracion(self) -> int:
        return self.fin - self.inicio + 1


def detectar(hs: np.ndarray | pd.Series, k: float, s: float, w: int = W) -> list[Excursion]:
    """Excursiones del detector bilateral, unidas si se solapan o se tocan."""
    x = np.asarray(hs, dtype=float)
    n = x.size
    umbral = k * s
    d = np.diff(x)
    abiertas = np.flatnonzero(np.abs(d) > umbral) + 1  # t con |Hs(t) − Hs(t−1)| > k·s
    out: list[Excursion] = []
    siguiente = 0  # el salto que cierra una excursión no abre otra
    for t in abiertas.tolist():
        if t < siguiente:
            continue
        signo = np.sign(x[t] - x[t - 1])
        for j in range(1, w + 1):
            if t + j >= n or not np.isfinite(x[t + j]):
                break
            if -signo * (x[t + j] - x[t]) > umbral:
                ini, fin = t, t + j - 1
                seg = x[ini : fin + 1] - x[t - 1]
                pico = ini + int(np.argmax(np.abs(seg)))
                exc = Excursion(ini, fin, pico, float(x[pico] - x[t - 1]))
                if out and exc.inicio <= out[-1].fin + 1:  # solapada o contigua: se unen
                    a = out[-1]
                    ini2 = a.inicio
                    seg2 = x[ini2 : fin + 1] - x[ini2 - 1]
                    p2 = ini2 + int(np.argmax(np.abs(seg2)))
                    out[-1] = Excursion(ini2, fin, p2, float(x[p2] - x[ini2 - 1]))
                else:
                    out.append(exc)
                siguiente = t + j + 1
                break
    return out


def contar(hs: np.ndarray, ks: np.ndarray, s: float, w: int = W) -> np.ndarray:
    """Número de excursiones unidas para cada k (para la calibración)."""
    return np.array([len(detectar(hs, float(k), s, w)) for k in ks])


# --- Vecinas (§3.1) -----------------------------------------------------------------------------


def salto_vecinas(hs_vecinas: np.ndarray, inicio: int, pico: int) -> tuple[float, int]:
    """(mediana de |Hs_v(pico) − Hs_v(inicio − 1)|, número de vecinas con dato). ``hs_vecinas``: (tiempo, vecina),
    solo el primer anillo."""
    v = np.abs(hs_vecinas[pico, :] - hs_vecinas[inicio - 1, :])
    v = v[np.isfinite(v)]
    return (float(np.median(v)) if v.size else float("nan")), int(v.size)


# --- Resolución (§3.4) --------------------------------------------------------------------------

DESCARTAR = "descartar"
MANTENER = "mantener"
MANUAL = "manual"


@dataclass(frozen=True)
class Resolucion:
    decision: str  # descartar | mantener | manual
    motivo: str


def resolver(
    exc: Excursion,
    hs: np.ndarray,
    tm02: np.ndarray,
    hs_vecinas: np.ndarray,
    p95: float,
) -> Resolucion:
    """Resolución automática de §3.4, con la protección de cola primero y el criterio de vecinas de §3.1."""
    if np.asarray(hs, dtype=float)[exc.pico] >= p95:
        return Resolucion(MANUAL, "cola_p95")
    return _resolver_sin_cola(exc, hs, tm02, hs_vecinas)


def _resolver_sin_cola(exc: Excursion, hs: np.ndarray, tm02: np.ndarray, hs_vecinas: np.ndarray) -> Resolucion:
    x = np.asarray(hs, dtype=float)
    previo = x[exc.inicio - 1]
    despues = x[exc.fin + 1] if exc.fin + 1 < x.size else np.nan
    vuelve = np.isfinite(despues) and abs(despues - previo) <= FRACCION_VUELTA * abs(exc.salto)
    if not vuelve:
        return Resolucion(MANTENER, "evento_sostenido")
    med, nv = salto_vecinas(hs_vecinas, exc.inicio, exc.pico)
    if nv < C.VECINAS_MIN:
        return Resolucion(MANUAL, "vecinas_insuficientes")
    if med >= FRACCION_VECINAS * abs(exc.salto):
        return Resolucion(MANUAL, "vecinas_comparten")
    t = np.asarray(tm02, dtype=float)
    salto_t = t[exc.inicio] - t[exc.inicio - 1]
    vuelta_t = t[exc.fin + 1] - t[exc.inicio - 1] if exc.fin + 1 < t.size else np.nan
    if abs(salto_t) >= SALTO_TM02_S and np.isfinite(vuelta_t) and abs(vuelta_t) <= FRACCION_VUELTA * abs(salto_t):
        return Resolucion(DESCARTAR, "salto_espectral_conjunto")
    return Resolucion(DESCARTAR, "reversion_rapida")


DUDOSO = "dudoso_vecinas_comparten"


def resolver_por_clase(exc: Excursion, hs: np.ndarray, tm02: np.ndarray, hs_vecinas: np.ndarray) -> Resolucion:
    """Resolución por clase de los candidatos del detector en la cola (contrato v8, §16), fijada antes de mirarlos.

    Las mismas definiciones que la resolución bajo P95 (vuelta, vecinas, salto espectral), con dos diferencias:
    - un salto que vuelve y que las vecinas comparten se **mantiene** como dudoso (va a la F4 y a la sensibilidad de
      la F9), porque una corrección de asimilación también abarca muchas celdas;
    - con vecinas insuficientes, se mantiene («cualquier otro»).
    Un salto que vuelve y no comparten se descarta; uno que no vuelve se mantiene.
    """
    r = _resolver_sin_cola(exc, hs, tm02, hs_vecinas)
    if r.motivo == "vecinas_comparten":
        return Resolucion(MANTENER, DUDOSO)
    if r.motivo == "vecinas_insuficientes":
        return Resolucion(MANTENER, "vecinas_insuficientes")
    return r


def percentil_cola(hs_decodificada: pd.Series | np.ndarray) -> float:
    """P95 de la Hs decodificada **antes de todo descarte** (F03-S-04)."""
    return float(np.nanpercentile(np.asarray(hs_decodificada, dtype=float), PERCENTIL_COLA))


@dataclass(frozen=True)
class Topes:
    total: int
    max_anio: int
    revision_cola: int

    @property
    def revision_requerida(self) -> list[str]:
        m = []
        if self.total > TOPE_TOTAL:
            m.append(f"{self.total} descartes automáticos > {TOPE_TOTAL}")
        if self.max_anio > TOPE_ANIO:
            m.append(f"{self.max_anio} descartes automáticos en un año > {TOPE_ANIO}")
        if self.revision_cola > TOPE_REVISION:
            m.append(f"{self.revision_cola} candidatos de k_rev con pico ≥ P95 > {TOPE_REVISION}")
        return m


def topes(tiempos_descartes: pd.DatetimeIndex, n_revision_cola: int) -> Topes:
    por_anio = pd.Series(1, index=tiempos_descartes).groupby(tiempos_descartes.year).sum()
    return Topes(len(tiempos_descartes), int(por_anio.max()) if len(por_anio) else 0, int(n_revision_cola))


# --- Dipolo (§3.5) ------------------------------------------------------------------------------


def dipolo(hs: np.ndarray, hs_vecinas: np.ndarray, k: float, s: float) -> np.ndarray:
    """Posiciones t con |ΔHs_nodo(t)| > k·s que se revierten en el paso siguiente y cuyas vecinas (primer anillo, al
    menos VECINAS_MIN con dato) tienen un ΔHs mediano de signo opuesto o menor que k·s/4."""
    x = np.asarray(hs, dtype=float)
    v = np.asarray(hs_vecinas, dtype=float)
    umbral = k * s
    out = []
    for t in (np.flatnonzero(np.abs(np.diff(x)) > umbral) + 1).tolist():
        if t + 1 >= x.size:
            continue
        signo = np.sign(x[t] - x[t - 1])
        if not (-signo * (x[t + 1] - x[t]) > umbral):
            continue
        dv = v[t, :] - v[t - 1, :]
        dv = dv[np.isfinite(dv)]
        if dv.size < C.VECINAS_MIN:
            out.append(t)  # sin vecinas suficientes: candidato que exige resolución manual
            continue
        med = float(np.median(dv))
        if np.sign(med) == -signo or abs(med) < DIPOLO_FRACCION * umbral:
            out.append(t)
    return np.array(out, dtype=int)


# --- Máximos (§3.10) ----------------------------------------------------------------------------


def maximos_independientes(hs: pd.Series, n: int = N_MAXIMOS, separacion: pd.Timedelta = SEPARACION_MAXIMOS):
    """Los n mayores picos de Hs separados al menos ``separacion``, sobre la serie **antes de todo descarte**."""
    x = hs.dropna().sort_values(ascending=False)
    elegidos: list[pd.Timestamp] = []
    for t in x.index:
        if all(abs(t - e) >= separacion for e in elegidos):
            elegidos.append(t)
            if len(elegidos) == n:
                break
    return hs.loc[sorted(elegidos)]
