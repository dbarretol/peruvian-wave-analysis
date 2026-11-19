"""GEV de máximos por bloque (contrato de la F6, §3.9; ADR-0020, ADR-0021, ADR-0022).

Convención de Coles (2001, ec. 3.2)::

    G(z) = exp{−[1 + ξ (z − μ)/σ]^(−1/ξ)},   1 + ξ (z − μ)/σ > 0,

ξ > 0 de cola pesada. ``scipy.stats.genextreme`` usa c = −ξ.

- **Ajuste:** máxima verosimilitud con varios arranques deterministas (momentos de Gumbel con una rejilla de ξ y el
  ajuste de scipy), en (μ, log σ, ξ) con ξ ∈ [−1, 1.5]; se queda el de mayor verosimilitud.
- **Validación de soporte:** la log-verosimilitud es finita (todos los máximos dentro del soporte) y |ξ̂| ≤ 0.5. La
  cota inferior es la de Coles (§3.3.1, Smith 1985: con ξ > −0.5 el MLE es regular, y el perfil lo necesita); la
  superior es una elección del contrato.
- **Nivel de retorno:** el principal es el de ADR-0022, el cuantil exp(−1/T) de la GEV de máximos anuales; al lado,
  como referencia, el de Coles (ec. 3.4), el cuantil 1 − 1/T.
- **IC de perfil** de ξ y del nivel de retorno (Coles §2.6.6 y §3.3.4): {θ : 2[ℓ̂ − ℓ_p(θ)] ≤ χ²₁(nivel)}. Un
  límite que no se alcanza queda abierto (NaN) con su causa, como en ``perfil``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .perfil import IntervaloPerfil, _limite

XI_CERO = 1e-8
XI_MIN, XI_MAX = -1.0, 1.5
XI_SOPORTE = 0.5
N_MIN = 10
FACTOR_TOPE = 20.0
EULER = 0.5772156649015329


def _y(p: float) -> float:
    return -np.log(p)


def loglik(x, mu: float, sigma: float, xi: float) -> float:
    """Log-verosimilitud de la GEV; −∞ fuera del soporte, con σ ≤ 0 o con ξ fuera de [−1, 1.5]."""
    x = np.asarray(x, dtype=float)
    if not (np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(xi)) or sigma <= 0 or not XI_MIN <= xi <= XI_MAX:
        return -np.inf
    s = (x - mu) / sigma
    n = x.size
    if abs(xi) < XI_CERO:
        return float(-n * np.log(sigma) - s.sum() - np.exp(-s).sum())
    xs = xi * s
    if np.any(xs <= -1.0):
        return -np.inf
    lz = np.log1p(xs)  # precisión cerca de ξ = 0
    with np.errstate(over="ignore"):  # exp → ∞ solo lejos del soporte efectivo: verosimilitud −∞, correcta
        ll = float(-n * np.log(sigma) - (1.0 / xi + 1.0) * lz.sum() - np.exp(-lz / xi).sum())
    return ll if np.isfinite(ll) else -np.inf


def cuantil(p, mu: float, sigma: float, xi: float) -> np.ndarray:
    """z con G(z) = p."""
    y = -np.log(np.asarray(p, dtype=float))
    if abs(xi) < XI_CERO:
        return mu - sigma * np.log(y)
    return mu - sigma / xi * (1.0 - np.power(y, -xi))


def p_adr0022(T: float) -> float:
    """Probabilidad de no excedencia anual del RL_T de ADR-0022: exp(−1/T)."""
    return float(np.exp(-1.0 / T))


def p_coles(T: float) -> float:
    """Probabilidad de no excedencia del RL_T de Coles (ec. 3.4): 1 − 1/T."""
    return 1.0 - 1.0 / T


@dataclass(frozen=True)
class AjusteGEV:
    mu: float
    sigma: float
    xi: float
    loglik: float
    n: int
    valido: bool  # verosimilitud finita: todos los máximos dentro del soporte
    soporte_ok: bool  # válido y |ξ̂| ≤ 0.5
    arranque: str | None
    causa: str | None = None

    def rl(self, T: float, definicion: str = "adr0022") -> float:
        p = p_adr0022(T) if definicion == "adr0022" else p_coles(T)
        return float(cuantil(p, self.mu, self.sigma, self.xi))


def _negll(t: np.ndarray, x: np.ndarray) -> float:
    ll = loglik(x, t[0], float(np.exp(t[1])), t[2])
    return -ll if np.isfinite(ll) else np.inf


def _arranques(x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    sd = float(np.std(x, ddof=1))
    s0 = max(sd * np.sqrt(6) / np.pi, 1e-6)
    m0 = float(np.mean(x)) - EULER * s0
    out = [(f"gumbel_xi{xi:g}", np.array([m0, np.log(s0), xi])) for xi in (0.0, -0.2, 0.2, -0.4, 0.4)]
    try:
        c, loc, sc = stats.genextreme.fit(x)
        if np.isfinite([c, loc, sc]).all() and sc > 0:
            out.append(("scipy", np.array([loc, np.log(sc), -c])))
    except (ValueError, RuntimeError):
        pass
    return out


def ajustar_gev(x) -> AjusteGEV:
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("Los máximos deben ser finitos")
    if x.size < N_MIN:
        return AjusteGEV(np.nan, np.nan, np.nan, -np.inf, int(x.size), False, False, None, f"menos de {N_MIN} máximos")
    mejor, nombre = None, None
    for nom, t0 in _arranques(x):
        if not np.isfinite(_negll(t0, x)):
            continue  # el arranque de Gumbel (ξ = 0) siempre es factible
        r = optimize.minimize(
            _negll,
            t0,
            args=(x,),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000, "maxfev": 40000},
        )
        if np.isfinite(r.fun) and (mejor is None or r.fun < mejor.fun - 1e-10):
            mejor, nombre = r, nom
    if mejor is None:
        return AjusteGEV(np.nan, np.nan, np.nan, -np.inf, int(x.size), False, False, None, "sin arranque factible")
    mu, ls, xi = mejor.x
    ll = -float(mejor.fun)
    soporte = bool(abs(xi) <= XI_SOPORTE)
    causa = None if soporte else f"|ξ̂| = {abs(xi):.3f} > {XI_SOPORTE}"
    return AjusteGEV(float(mu), float(np.exp(ls)), float(xi), ll, int(x.size), True, soporte, nombre, causa)


# --- Perfiles ------------------------------------------------------------------------------------------


def _max_restringido(f, inicios) -> float:
    """máx de f(v) (2 parámetros libres) desde varios arranques; −∞ si ninguno es factible."""
    mejor = -np.inf
    for v0 in inicios:
        if not np.isfinite(f(v0)):
            continue
        r = optimize.minimize(
            lambda v: -f(v) if np.isfinite(f(v)) else np.inf,
            v0,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000},
        )
        if np.isfinite(r.fun):
            mejor = max(mejor, -float(r.fun))
    return mejor


def loglik_perfil_xi(x, xi: float, aj: AjusteGEV) -> float:
    def f(v):
        return loglik(x, v[0], float(np.exp(v[1])), xi)

    base = [np.array([aj.mu, np.log(aj.sigma)])]
    # arranque con μ por debajo del mínimo (soporte con ξ > 0) y por encima del máximo menos el extremo (ξ < 0)
    s = aj.sigma
    if xi < 0:
        base.append(np.array([float(np.max(x)) - s / (-xi) * 0.5, np.log(s)]))
    else:
        base.append(np.array([float(np.min(x)) + s / max(xi, 1e-3) * 0.5 - s, np.log(s)]))
    return _max_restringido(f, base)


def loglik_perfil_rl(x, z: float, p: float, aj: AjusteGEV) -> float:
    """Perfil del cuantil z_p: μ = z_p + σ/ξ [1 − y_p^(−ξ)], maximizado en (log σ, ξ)."""
    y = _y(p)

    def mu_de(sigma, xi):
        return z + sigma * np.log(y) if abs(xi) < XI_CERO else z + sigma / xi * (1.0 - y ** (-xi))

    def f(v):
        sigma, xi = float(np.exp(v[0])), float(v[1])
        return loglik(x, mu_de(sigma, xi), sigma, xi)

    inicios = [np.array([np.log(aj.sigma), xi0]) for xi0 in (aj.xi, aj.xi - 0.2, aj.xi + 0.2, 0.0)]
    return _max_restringido(f, inicios)


def _q(nivel: float) -> float:
    return float(stats.chi2.ppf(nivel, 1))


def perfil_xi(x, aj: AjusteGEV, nivel: float = 0.95) -> IntervaloPerfil:
    x = np.asarray(x, dtype=float)
    q = _q(nivel)

    def deficit(xi):
        return 2.0 * (aj.loglik - loglik_perfil_xi(x, xi, aj))

    lo, clo = _limite(deficit, aj.xi, q, XI_MIN)
    hi, chi = _limite(deficit, aj.xi, q, XI_MAX)
    return IntervaloPerfil(aj.xi, lo, hi, clo, chi, nivel, aj.loglik)


def perfil_rl(x, aj: AjusteGEV, T: float, nivel: float = 0.95, definicion: str = "adr0022") -> IntervaloPerfil:
    x = np.asarray(x, dtype=float)
    p = p_adr0022(T) if definicion == "adr0022" else p_coles(T)
    q = _q(nivel)
    z = aj.rl(T, definicion)

    def deficit(zz):
        return 2.0 * (aj.loglik - loglik_perfil_rl(x, zz, p, aj))

    escala = max(z - float(np.median(x)), aj.sigma)
    lo, clo = _limite(deficit, z, q, float(np.min(x)))
    hi, chi = _limite(deficit, z, q, z + FACTOR_TOPE * escala)
    return IntervaloPerfil(z, lo, hi, clo, chi, nivel, aj.loglik)
