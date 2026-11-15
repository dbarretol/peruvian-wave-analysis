"""La única función de ajuste de la GPD del proyecto.

GPD de los excesos sobre el umbral (localización 0)::

    S(x) = (1 + ξ x / σ)^(-1/ξ),   x ≥ 0,   1 + ξ x / σ > 0.

Reglas:

- La log-verosimilitud vale −∞ fuera del soporte o con σ ≤ 0.
- Espacio de parámetros ξ ≥ −1: con ξ < −1 la verosimilitud no está acotada. Si el máximo restringido está en
  ξ = −1, se devuelve la frontera ξ = −1, σ = exceso máximo (extremo superior = exceso máximo), marcada.
- Un ajuste es válido si su log-verosimilitud es finita y su extremo superior supera al exceso máximo, o lo
  iguala en el caso frontera marcado.
- **No regular** si ξ̂ ≤ −0.5 o en la frontera (Smith, 1985): ninguna fase usa errores estándar del hessiano ni
  IC de Wald en ese caso. La marca por intervalo (el IC de ξ incluye valores ≤ −0.5) la pone el perfil de ξ.
- Arranques deterministas en orden: momentos, PWM y una rejilla de ξ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

XI_CERO = 1e-8
XI_MIN = -1.0
XI_NO_REGULAR = -0.5
N_MIN = 10


@dataclass(frozen=True)
class AjusteGPD:
    xi: float
    sigma: float
    loglik: float
    n: int
    valido: bool
    n_arranques: int
    arranque: str | None
    causa: str | None = None
    frontera: bool = False

    @property
    def no_regular(self) -> bool:
        """ξ̂ ≤ −0.5 o frontera: sin normalidad asintótica del MLE; usar perfil (o bootstrap con cautela), no Wald."""
        return bool(self.valido and (self.frontera or self.xi <= XI_NO_REGULAR))

    @property
    def extremo_superior(self) -> float:
        return -self.sigma / self.xi if self.xi < 0 else np.inf


def _comprobar(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("Los excesos deben ser finitos")
    if np.any(x < 0):
        raise ValueError("Los excesos deben ser ≥ 0")
    return x


def loglik(x, xi: float, sigma: float) -> float:
    """Log-verosimilitud de la GPD; −∞ fuera del soporte o con σ ≤ 0."""
    x = np.asarray(x, dtype=float)
    if not (np.isfinite(xi) and np.isfinite(sigma)) or sigma <= 0 or xi < XI_MIN:
        return -np.inf
    n = x.size
    if abs(xi) < XI_CERO:
        return float(-n * np.log(sigma) - x.sum() / sigma)
    if xi == XI_MIN:
        # Frontera: densidad uniforme 1/σ en [0, σ]; el exceso máximo puede tocar el extremo.
        return float(-n * np.log(sigma)) if np.all(x <= sigma) else -np.inf
    z = 1.0 + xi * x / sigma
    if np.any(z <= 0):
        return -np.inf
    return float(-n * np.log(sigma) - (1.0 / xi + 1.0) * np.log(z).sum())


def supervivencia(x, xi: float, sigma: float) -> np.ndarray:
    """S(x): 1 bajo cero y 0 más allá del extremo superior."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    pos = x > 0
    if abs(xi) < XI_CERO:
        out[pos] = np.exp(-x[pos] / sigma)
        return out
    z = 1.0 + xi * x[pos] / sigma
    out[pos] = np.where(z > 0, np.power(np.clip(z, 1e-300, None), -1.0 / xi), 0.0)
    return out


def cuantil_supervivencia(p, xi: float, sigma: float) -> np.ndarray:
    """x con S(x) = p."""
    p = np.asarray(p, dtype=float)
    if abs(xi) < XI_CERO:
        return -sigma * np.log(p)
    return sigma / xi * (np.power(p, -xi) - 1.0)


def _negll(theta: np.ndarray, x: np.ndarray) -> float:
    xi, log_sigma = theta
    if xi <= XI_MIN:
        return np.inf
    ll = loglik(x, xi, float(np.exp(log_sigma)))
    return -ll if np.isfinite(ll) else np.inf


def arranque_pwm(x) -> tuple[float, float] | None:
    """Estimador por momentos ponderados por probabilidad (Hosking y Wallis, 1987), con ξ = −k.

    α1 = E[X·(1 − F(X))] se estima con los pesos (n − 1 − i)/(n − 1) sobre la muestra ordenada.
    Devuelve None si no hay solución con σ > 0.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    if n < 2:
        return None
    b0 = float(xs.mean())
    b1 = float(np.mean(xs * ((n - 1 - np.arange(n)) / (n - 1))))
    den = b0 - 2.0 * b1
    if abs(den) <= 1e-12 or 2.0 * b0 * b1 / den <= 0:
        return None
    return -(b0 / den - 2.0), 2.0 * b0 * b1 / den


def arranques(x: np.ndarray) -> list[tuple[str, float, float]]:
    """Arranques deterministas en orden: momentos, PWM y rejilla de ξ."""
    m = float(x.mean())
    v = float(x.var(ddof=1)) if x.size > 1 else 0.0
    out: list[tuple[str, float, float]] = []
    if v > 0 and m > 0:
        out.append(("momentos", 0.5 * (1.0 - m * m / v), 0.5 * m * (m * m / v + 1.0)))
    else:
        out.append(("momentos", 0.0, max(m, 1e-6)))
    pwm = arranque_pwm(x)
    out.append(("pwm", *pwm) if pwm else ("pwm", 0.0, max(m, 1e-6)))
    for xi in (-0.3, 0.0, 0.3):
        out.append((f"xi={xi:g}", xi, max(m * (1.0 - xi), 1e-6)))
    return out


def _reparar(xi: float, sigma: float, xmax: float) -> tuple[float, float]:
    """Lleva un arranque al interior del soporte: ξ > −1 y extremo superior > exceso máximo."""
    xi = max(xi, -0.95)
    if xi < 0:
        sigma = max(sigma, 1.05 * (-xi) * xmax)
    return xi, max(sigma, 1e-6)


def ajustar_gpd(excesos, n_min: int = N_MIN) -> AjusteGPD:
    """MLE de la GPD en ξ ≥ −1 con validación de soporte.

    Prueba los arranques en orden y acepta el primero que converge a un óptimo interior válido, salvo que la
    frontera ξ = −1 tenga mayor verosimilitud. Si todos derivan hacia ξ = −1, devuelve la frontera. Un ajuste
    fallido lleva su causa: ``n_insuficiente``, ``fuera_de_soporte`` o ``sin_convergencia``.
    """
    x = _comprobar(excesos)
    n = x.size
    if n < n_min:
        return AjusteGPD(np.nan, np.nan, -np.inf, n, False, 0, None, "n_insuficiente")
    xmax = float(x.max())
    lista = arranques(x)
    ll_frontera = loglik(x, XI_MIN, xmax) if xmax > 0 else -np.inf
    causa = "sin_convergencia"
    hacia_frontera = False
    for k, (nombre, xi0, s0) in enumerate(lista, start=1):
        xi0, s0 = _reparar(xi0, s0, xmax)
        res = optimize.minimize(
            _negll,
            np.array([xi0, np.log(s0)]),
            args=(x,),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000, "maxfev": 8000},
        )
        xi, sigma = float(res.x[0]), float(np.exp(res.x[1]))
        ll = loglik(x, xi, sigma)
        if xi <= XI_MIN + 1e-3:
            hacia_frontera = True
            continue
        if not res.success:
            causa = "sin_convergencia"
            continue
        extremo = -sigma / xi if xi < 0 else np.inf
        if not np.isfinite(ll) or extremo <= xmax:
            causa = "fuera_de_soporte"
            continue
        res2 = optimize.minimize(
            _negll, res.x, args=(x,), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000}
        )
        if res2.success and res2.fun <= res.fun:
            xi, sigma = float(res2.x[0]), float(np.exp(res2.x[1]))
            ll = loglik(x, xi, sigma)
        if np.isfinite(ll_frontera) and ll_frontera > ll + 1e-9:
            return AjusteGPD(XI_MIN, xmax, ll_frontera, n, True, k, "frontera", None, True)
        return AjusteGPD(xi, sigma, ll, n, True, k, nombre)
    if hacia_frontera and np.isfinite(ll_frontera):
        return AjusteGPD(XI_MIN, xmax, ll_frontera, n, True, len(lista), "frontera", None, True)
    return AjusteGPD(np.nan, np.nan, -np.inf, n, False, len(lista), None, causa)


def sigma_dado_xi(x, xi: float) -> tuple[float, float]:
    """(σ, ℓ) con σ = argmax_σ ℓ(ξ, σ) para ξ fijo; en ξ = −1, la frontera σ = máx x. ℓ = −∞ si no hay σ."""
    x = np.asarray(x, dtype=float)
    xmax, m = float(x.max()), float(x.mean())
    if xi <= XI_MIN:
        return xmax, loglik(x, XI_MIN, xmax)
    lo = np.log(max(-xi * xmax * (1 + 1e-12), 1e-12)) if xi < 0 else np.log(m) - 10

    def f(ls):
        v = loglik(x, xi, float(np.exp(ls)))
        return -v if np.isfinite(v) else 1e300

    r = optimize.minimize_scalar(f, bounds=(lo, np.log(m) + 10), method="bounded", options={"xatol": 1e-11})
    v = -float(r.fun)
    return float(np.exp(r.x)), (v if v > -1e299 else -np.inf)


def perfil_loglik_xi(excesos, xis) -> np.ndarray:
    """ℓ_p(ξ) = max_σ ℓ(ξ, σ) en cada ξ (NaN si no hay σ factible)."""
    x = _comprobar(excesos)
    out = np.full(len(xis), np.nan)
    for i, xi in enumerate(np.asarray(xis, dtype=float)):
        ll = sigma_dado_xi(x, float(xi))[1]
        out[i] = ll if np.isfinite(ll) else np.nan
    return out
