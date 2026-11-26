"""R2: prueba de score de Northrop y Coleman (2014) para la estabilidad de ξ por tramos (contrato de la F7, §3.4).

**Modelo** (§2, ec. 4–5 del manuscrito). Umbrales v_1 < … < v_m; Y = exceso de v_1. En (v_j, v_{j+1}), Y − v_j
sigue una GPD(ξ_j, σ_j) truncada, con σ_{j+1} = σ_j + ξ_j·w_j (densidad continua), w_j = v_{j+1} − v_j, y
p_j = P(Y > v_j) = Π_{i<j} (1 + ξ_i w_i/σ_i)^(−1/ξ_i). La log-densidad de una observación es

    log f(y) = log p_j − log σ_j − (1 + 1/ξ_j)·log(1 + ξ_j (y − v_j)/σ_j),   v_j < y < v_{j+1}.

Parámetros φ = (σ_1, ξ_1, …, ξ_m). H0: ξ_1 = … = ξ_m, ajustada con la GPD sobre v_1 (el estimador restringido).

**Estadístico** (ec. 7): S = U(φ̃)ᵀ i(φ̃)⁻¹ U(φ̃), con la información **esperada** en el estimador restringido,
frente a χ² con m − 1 grados de libertad; requiere ξ > −½. S no depende de la parametrización (U e i se transforman
con el mismo jacobiano), así que se usa φ en lugar de la del apéndice del artículo.

**Cálculo (diferencia de implementación, declarada).** El apéndice da U e i en forma cerrada. Aquí:

- U(φ̃) = gradiente de Σ log f(y_i; φ) por **paso complejo** (Squire y Trapp, 1998): ∂g/∂φ_k = Im g(φ + i·h·e_k)/h
  con h = 1e-100. No resta dos valores parecidos, así que es exacto a precisión de máquina, y el paso no saca a una
  observación del soporte aun junto al extremo superior cuando ξ < 0 (la log-densidad es analítica en φ dentro de
  cada tramo, porque los v_j son fijos);
- i(φ̃) = n·E[u uᵀ], con u el score de una observación (bajo regularidad, igual a −E[∂²ℓ]), por cuadratura en la
  escala t = −log S(y) ~ Exp(1) bajo H0: Gauss–Legendre en cada tramo finito y Gauss–Laguerre en el último.

Se valida (§8) contra la información conocida de la GPD (m = 1; Smith, 1985) y contra Monte Carlo con m = 2 y 3.

**Regla** (segunda regla de §4.2 del artículo; contrato §3.4): con v_1 < … < v_10 (los u_k de posición impar de la
rejilla sin repetidos), para i = 1, …, 9 se prueba H0 con los umbrales v_i, …, v_10 (m = 11 − i, χ² con 10 − i g.l.).
u* = el menor v_i con p > 0.05 en v_i **y** en todos los v superiores hasta v_9. Un v_j con ajuste fallido o con
ξ̃ ≤ −½ es «no evaluable» y rompe la condición (C-08).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import roots_laguerre, roots_legendre

from ..extremos import gpd
from .umbral import CON_RESULTADO, SIN_RESULTADO, ResultadoRegla

ALFA = 0.05
XI_MIN_REGULAR = -0.5
N_LEGENDRE = 48
N_LAGUERRE = 60
Z_MIN_EXTREMO = 1e-8
_XL, _WL = roots_legendre(N_LEGENDRE)
_XG, _WG = roots_laguerre(N_LAGUERRE)


def _log1p_div(xi, a):
    """log(1 + ξ a)/ξ, con el límite a cuando ξ → 0 (serie en ξ·a pequeño). Admite valores complejos."""
    xi = np.asarray(xi)
    za = xi * a
    chico = np.abs(za) < 1e-6
    seguro = np.where(chico, 1.0, xi)
    return np.where(chico, a - xi * a * a / 2.0 + xi * xi * a**3 / 3.0, np.log1p(za) / seguro)


def _escalas(v: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """σ_j y log p_j de cada tramo."""
    m = v.size
    sig, xi = phi[0], phi[1:]
    w = np.diff(v)
    s = np.empty(m, dtype=phi.dtype)
    logp = np.zeros(m, dtype=phi.dtype)
    s[0] = sig
    for j in range(m - 1):
        s[j + 1] = s[j] + xi[j] * w[j]
        logp[j + 1] = logp[j] - _log1p_div(xi[j], w[j] / s[j])
    return s, logp


def log_densidad(y, v, phi) -> np.ndarray:
    """log f(y) del modelo por tramos; −∞ fuera del soporte. ``y`` = excesos de v_1 (v_1 = 0)."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    phi = np.asarray(phi)
    phi = phi.astype(complex if np.iscomplexobj(phi) else float)
    s, logp = _escalas(v, phi)
    if np.any(s.real <= 0) or not np.all(np.isfinite(logp)):
        return np.full(y.shape, -np.inf, dtype=phi.dtype)
    j = np.clip(np.searchsorted(v, y, side="right") - 1, 0, v.size - 1)
    xi = phi[1:][j]
    a = (y - v[j]) / s[j]
    z = 1.0 + xi * a
    out = np.full(y.shape, -np.inf, dtype=phi.dtype)
    ok = z.real > 0
    out[ok] = logp[j][ok] - np.log(s[j][ok]) - np.log(z[ok]) - _log1p_div(xi[ok], a[ok])
    return out


PASO_COMPLEJO = 1e-100


def score_por_observacion(y, v, phi) -> np.ndarray:
    """Matriz n × (m + 1) con el gradiente de log f(y_i; φ) por paso complejo. NaN fuera del soporte."""
    phi = np.asarray(phi, dtype=float)
    out = np.empty((np.size(y), phi.size))
    for k in range(phi.size):
        e = np.zeros(phi.size, dtype=complex)
        e[k] = 1j * PASO_COMPLEJO
        g = log_densidad(y, v, phi + e)
        out[:, k] = np.where(np.isfinite(g.real), g.imag / PASO_COMPLEJO, np.nan)
    return out


def nodos_bajo_h0(v, xi: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodos y pesos de E[g(Y)] con Y ~ GPD(ξ, σ) exceso de v_1, partidos en los v_j (la densidad del score cambia
    de forma en cada v_j). Escala t = −log S(y) ~ Exp(1)."""
    v = np.asarray(v, dtype=float)
    t_v = -np.log(np.clip(gpd.supervivencia(v, xi, sigma), 1e-300, None))
    ys, ws = [], []
    for j in range(v.size - 1):
        a, b = t_v[j], t_v[j + 1]
        if not np.isfinite(b) or b <= a:
            continue
        t = (b - a) / 2.0 * _XL + (a + b) / 2.0
        ys.append(gpd.cuantil_supervivencia(np.exp(-t), xi, sigma))
        ws.append((b - a) / 2.0 * _WL * np.exp(-t))
    a = t_v[-1]
    if np.isfinite(a):
        t = a + _XG
        ys.append(gpd.cuantil_supervivencia(np.exp(-t), xi, sigma))
        ws.append(_WG * np.exp(-a))
    y, w = np.concatenate(ys), np.concatenate(ws)
    if xi < 0:
        # junto al extremo superior, 1 + ξ y/σ = S^(−ξ) se pierde por redondeo: se descartan los nodos con
        # S^(−ξ) < 1e-8, cuyo peso total es < e^(−18/|ξ|) (despreciable; el integrando crece como t²)
        t_nodo = -np.log(np.clip(gpd.supervivencia(y, xi, sigma), 1e-300, None))
        ok = (xi * t_nodo > np.log(Z_MIN_EXTREMO)) & (gpd.supervivencia(y, xi, sigma) > 0)
        y, w = y[ok], w[ok]
    return y, w


def informacion_esperada(v, xi: float, sigma: float) -> np.ndarray:
    """i_1(φ̃) = E[u uᵀ] de una observación bajo H0 (ξ_j = ξ en todos los tramos)."""
    v = np.asarray(v, dtype=float)
    phi = np.concatenate([[sigma], np.full(v.size, xi)])
    y, w = nodos_bajo_h0(v, xi, sigma)
    u = score_por_observacion(y, v, phi)
    return (u * w[:, None]).T @ u


@dataclass(frozen=True)
class PruebaScore:
    S: float
    gl: int
    p: float
    xi: float
    sigma: float
    evaluable: bool
    causa: str | None = None


def prueba_score(excesos, v, ajuste: gpd.AjusteGPD | None = None) -> PruebaScore:
    """Prueba de score de H0: ξ_1 = … = ξ_m con los umbrales v (relativos a v_1 = 0) y los excesos de v_1."""
    y = np.asarray(excesos, dtype=float)
    v = np.asarray(v, dtype=float)
    m = v.size
    if v[0] != 0 or np.any(np.diff(v) <= 0):
        raise ValueError("Los umbrales deben ser relativos a v_1 = 0 y crecientes")
    a = ajuste if ajuste is not None else gpd.ajustar_gpd(y)
    if not a.valido:
        return PruebaScore(np.nan, m - 1, np.nan, np.nan, np.nan, False, a.causa or "ajuste_fallido")
    if a.frontera or a.xi <= XI_MIN_REGULAR:
        return PruebaScore(np.nan, m - 1, np.nan, a.xi, a.sigma, False, "xi_no_regular")
    phi = np.concatenate([[a.sigma], np.full(m, a.xi)])
    U = score_por_observacion(y, v, phi).sum(axis=0)
    info = y.size * informacion_esperada(v, a.xi, a.sigma)
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(info))):
        return PruebaScore(np.nan, m - 1, np.nan, a.xi, a.sigma, False, "score_no_finito")
    try:
        S = float(U @ np.linalg.solve(info, U))
    except np.linalg.LinAlgError:
        return PruebaScore(np.nan, m - 1, np.nan, a.xi, a.sigma, False, "informacion_singular")
    return PruebaScore(S, m - 1, float(stats.chi2.sf(S, m - 1)), a.xi, a.sigma, True)


def umbrales_r2(u_rejilla) -> np.ndarray:
    """v_1 < … < v_10: los u_k de posición impar (k = 1, 3, …) de la rejilla sin repetidos."""
    return np.asarray(u_rejilla, dtype=float)[0::2]


def pruebas_r2(picos, vs: Sequence[float], ajustes: Sequence[gpd.AjusteGPD] | None = None) -> list[PruebaScore]:
    """Prueba en cada v_i, i = 1, …, len(vs) − 1, con los umbrales v_i, …, v_último."""
    p = np.asarray(picos, dtype=float)
    vs = np.asarray(vs, dtype=float)
    out = []
    for i in range(vs.size - 1):
        exc = p[p > vs[i]] - vs[i]
        a = ajustes[i] if ajustes is not None else None
        out.append(prueba_score(exc, vs[i:] - vs[i], a))
    return out


def regla_r2(pruebas: Sequence[PruebaScore], alfa: float = ALFA, paso: int = 2) -> ResultadoRegla:
    """u* = el menor v_i con p > α en v_i y en todos los superiores evaluados (hasta v_9); «no evaluable» rompe.

    ``indice`` y ``ventana`` van en posiciones de la rejilla de u_k: v_i es el u_k de posición ``paso·i``.
    """
    acepta = [t.evaluable and t.p > alfa for t in pruebas]
    det = {
        "p": [t.p for t in pruebas],
        "evaluable": [t.evaluable for t in pruebas],
        "causa": [t.causa for t in pruebas],
    }
    for i in range(len(acepta)):
        if all(acepta[i:]):
            det["i_v"] = i
            return ResultadoRegla(CON_RESULTADO, paso * i, (paso * i, paso * (len(acepta) - 1)), det)
    return ResultadoRegla(SIN_RESULTADO, None, None, det)
