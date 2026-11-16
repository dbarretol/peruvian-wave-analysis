"""Calibración de k del detector con series sintéticas, fijada antes de abrir la serie (contrato de la F3, §3.4).

- Rejilla de 4 generadores de **fondo**: AR(1) con coeficiente φ e innovaciones t de Student con ν grados de
  libertad, (φ, ν) ∈ {0.90, 0.98} × {3, 10}.
- k_g se calibra con **el fondo solo**: el menor k de la rejilla {4, 4.5, …, 40} cuya cota superior de Poisson al
  95 % de la tasa de falsas alarmas (excursiones unidas) es < 1 cada 100 años.
- k_desc = el mayor k_g (único que permite descarte automático); k_rev = el menor k_g de los generadores con ν = 10
  (solo manda a revisión).
- Las series se redondean **siempre** al cuanto de la serie real, expresado en unidades de su s (F03-S-13).
- Las tormentas (10 por año, amplitud U(10, 60)·s_fondo, subida U(6, 36) h y bajada U(12, 96) h en semicoseno) no
  entran en el criterio: su conteo se publica aparte como carga manual esperada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.signal import lfilter

from ..comun import semillas
from . import detector as D

GENERADORES = tuple((phi, nu) for phi in (0.90, 0.98) for nu in (3, 10))
REJILLA_K = np.round(np.arange(4.0, 40.0 + 1e-9, 0.5), 2)
ANIOS = 10_000
PASOS_ANIO = 2920  # 365 × 8
TASA_MAX = 1.0 / 100.0  # falsas alarmas por año
CUANTOS_MIN = 2.0  # s < 2 cuantos → revisión requerida
TORMENTAS_ANIO = 10.0
AMPLITUD = (10.0, 60.0)  # en unidades de s_fondo
SUBIDA_H = (6.0, 36.0)
BAJADA_H = (12.0, 96.0)
PASO_H = 3.0


class CalibracionImposible(Exception):
    """Ningún k ≤ 40 cumple para algún generador, o s < 2 cuantos: la etapa se detiene con «revisión requerida»."""


def fondo(n: int, phi: float, nu: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) con innovaciones t de Student, estacionario desde el inicio."""
    e = rng.standard_t(nu, size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2)
    # recursión lineal x_t = φ·x_{t−1} + e_t, vectorizada con lfilter
    x[1:] = lfilter([1.0], [1.0, -phi], e[1:], zi=[phi * x[0]])[0]
    return x


def tormentas(n: int, s_fondo: float, rng: np.random.Generator) -> np.ndarray:
    """Suma de tormentas en semicoseno (solo para la carga manual esperada, no para el criterio)."""
    out = np.zeros(n)
    n_t = rng.poisson(TORMENTAS_ANIO * n / PASOS_ANIO)
    inicios = rng.integers(0, n, n_t)
    for i0 in inicios.tolist():
        a = rng.uniform(*AMPLITUD) * s_fondo
        sub = max(1, int(round(rng.uniform(*SUBIDA_H) / PASO_H)))
        baj = max(1, int(round(rng.uniform(*BAJADA_H) / PASO_H)))
        forma = np.r_[
            0.5 * (1 - np.cos(np.pi * np.arange(1, sub + 1) / sub)),
            0.5 * (1 + np.cos(np.pi * np.arange(1, baj + 1) / baj)),
        ]
        fin = min(n, i0 + forma.size)
        out[i0:fin] += a * forma[: fin - i0]
    return out


def cuantizar(x: np.ndarray, cuanto_en_s: float) -> np.ndarray:
    return np.round(x / cuanto_en_s) * cuanto_en_s if cuanto_en_s > 0 else x


def cota_superior(n: int, anios: float, nivel: float = 0.95) -> float:
    """Cota superior unilateral de Poisson al nivel dado de la tasa por año."""
    return float(stats.chi2.ppf(nivel, 2 * (n + 1)) / 2.0 / anios)


def elegir_k(conteos: np.ndarray, anios: float, rejilla: np.ndarray = REJILLA_K) -> float:
    """Menor k de la rejilla con cota superior al 95 % < TASA_MAX. Sin ninguno: CalibracionImposible."""
    for k, c in zip(rejilla, conteos, strict=True):
        if cota_superior(int(c), anios) < TASA_MAX:
            return float(k)
    raise CalibracionImposible("Ningún k ≤ 40 cumple < 1 falsa alarma cada 100 años (cota al 95 %)")


@dataclass
class Calibracion:
    k_por_generador: dict[tuple[float, float], float]
    conteos: dict[tuple[float, float], list[int]]
    carga_tormentas: dict[tuple[float, float], list[int]] = field(default_factory=dict)
    anios: int = ANIOS
    cuanto_en_s: float = 0.0

    @property
    def k_desc(self) -> float:
        return max(self.k_por_generador.values())

    @property
    def k_rev(self) -> float:
        return min(k for (phi, nu), k in self.k_por_generador.items() if nu == 10)


def exigir_escala(s_m: float, cuanto_m: float = 0.01) -> float:
    """s en cuantos; < 2 cuantos → CalibracionImposible (F03-S-13)."""
    if not np.isfinite(s_m) or s_m / cuanto_m < CUANTOS_MIN:
        raise CalibracionImposible(f"s = {s_m:.4f} m = {s_m / cuanto_m:.2f} cuantos < {CUANTOS_MIN}")
    return s_m / cuanto_m


def calibrar(
    cuanto_en_s: float,
    anios: int = ANIOS,
    generadores=GENERADORES,
    con_tormentas: bool = True,
    raiz: str = "f03-calibracion",
) -> Calibracion:
    """Calibra k_g para cada generador. ``cuanto_en_s``: cuanto de la serie real (0.01 m) dividido por su s."""
    n = anios * PASOS_ANIO
    kg, conteos, carga = {}, {}, {}
    for phi, nu in generadores:
        rng = semillas.generador(raiz, f"phi={phi}", f"nu={nu}")
        x = fondo(n, phi, nu, rng)
        s_fondo = D.escala_s(x)
        x = x / s_fondo  # en unidades de s_fondo
        xq = cuantizar(x, cuanto_en_s)
        s = D.escala_s(xq)  # s del detector: la de la serie que se analiza
        c = D.contar(xq, REJILLA_K, s)
        conteos[(phi, nu)] = c.tolist()
        kg[(phi, nu)] = elegir_k(c, anios)
        if con_tormentas:
            xt = cuantizar(x + tormentas(n, 1.0, rng), cuanto_en_s)
            carga[(phi, nu)] = D.contar(xt, REJILLA_K, D.escala_s(xt)).tolist()
    return Calibracion(kg, conteos, carga, anios, cuanto_en_s)


# --- Publicación de §3.4: tasa de falsas alarmas con IC y potencia (F03-K-03) ------------------

AMPLITUDES_POTENCIA = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)  # en unidades de k·s (§3.4)
ANIOS_POTENCIA = 200
SEPARACION_POTENCIA = 500  # pasos entre picos sembrados


def tasa_falsas_alarmas(cal: Calibracion, k: float, nivel: float = 0.95) -> dict[str, dict]:
    """Por generador: excursiones del fondo con ese k, tasa por año e IC bilateral exacto de Poisson (Garwood).
    Vale para la serie sintética, no para la real (§3.4)."""
    i = int(np.flatnonzero(np.isclose(REJILLA_K, k))[0])
    a = (1.0 - nivel) / 2.0
    out = {}
    for (phi, nu), c in cal.conteos.items():
        n = int(c[i])
        lo = float(stats.chi2.ppf(a, 2 * n) / 2.0 / cal.anios) if n > 0 else 0.0
        hi = float(stats.chi2.ppf(1.0 - a, 2 * (n + 1)) / 2.0 / cal.anios)
        out[f"phi={phi},nu={nu}"] = {"n": n, "tasa_anio": n / cal.anios, "ic95_anio": [lo, hi]}
    return out


def potencia(
    cuanto_en_s: float,
    k: float,
    amplitudes=AMPLITUDES_POTENCIA,
    anios: int = ANIOS_POTENCIA,
    generadores=GENERADORES,
    raiz: str = "f03-potencia",
) -> dict[str, dict[str, float]]:
    """Por generador: fracción de picos de un paso de amplitud a·k·s sembrados en el fondo cuantizado que el detector
    marca en el registro sembrado (§3.4, «potencia para picos de 0.3 a 1.5·s·k»)."""
    n = anios * PASOS_ANIO
    out = {}
    for phi, nu in generadores:
        rng = semillas.generador(raiz, f"phi={phi}", f"nu={nu}")
        x = fondo(n, phi, nu, rng)
        x = x / D.escala_s(x)
        xq = cuantizar(x, cuanto_en_s)
        s = D.escala_s(xq)
        pos = np.arange(SEPARACION_POTENCIA, n - 10, SEPARACION_POTENCIA)
        fila = {}
        for a in amplitudes:
            y = xq.copy()
            y[pos] += cuantizar(np.full(pos.size, a * k * s), cuanto_en_s)
            picos = {e.pico for e in D.detectar(y, k, s)}
            fila[f"{a:.1f}"] = float(np.mean([int(q) in picos for q in pos.tolist()]))
        out[f"phi={phi},nu={nu}"] = fila
    return out
