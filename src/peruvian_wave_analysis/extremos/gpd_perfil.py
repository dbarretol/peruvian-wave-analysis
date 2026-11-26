"""Ajuste rápido de la GPD por perfil en una dimensión, **solo** para los reajustes del bootstrap de R3 (F7, §3.4).

Excepción a la regla de ``gpd.py`` («la única función de ajuste de la GPD del proyecto»), aceptada en ADR-0034. Es el
**mismo estimador** que ``gpd.ajustar_gpd`` (máxima verosimilitud en ξ ≥ −1, con la misma frontera y la misma
validación de soporte), calculado por otro camino. Toda estimación que se publica sale de ``ajustar_gpd``. Va en un
módulo aparte para no cambiar la huella de ``gpd.py``, que importan las etapas ya publicadas.

**Método** (Grimshaw, 1993). Con θ = ξ/σ, para θ fijo la verosimilitud se maximiza en ξ(θ) = mean log(1 + θx) y
σ = ξ/θ, y queda el perfil ℓ*(θ) = −n·log(ξ(θ)/θ) − n − n·ξ(θ), con θ > −1/máx x (el soporte: el extremo superior
−σ/ξ = −1/θ supera al exceso máximo). En θ → 0 es la exponencial: ℓ* = −n·log(mean x) − n. Se evalúa ℓ* en una
rejilla fija de θ y se refina con Brent entre los vecinos del mejor punto.

**Cuándo responde y cuándo delega.** El método rápido solo devuelve un resultado si encuentra un máximo **interior
claro**: ξ̂ > −1 + 10⁻³ (el criterio de ``ajustar_gpd`` para «hacia la frontera»), máximo que no está en el borde de la
rejilla, y log-verosimilitud mayor que la de la frontera ξ = −1 en más de 10⁻⁹. En cualquier otro caso (n < 10,
frontera, máximo en el borde, valores no finitos) **delega en** ``ajustar_gpd``. Así los fallos y la frontera se
tratan igual por construcción. La diferencia que queda posible es un máximo interior que ``ajustar_gpd`` no alcanza
(un fallo de convergencia, o un óptimo local); su frecuencia se mide en las pruebas y en el piloto del ADR.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from . import gpd

XI_HACIA_FRONTERA = gpd.XI_MIN + 1e-3
N_REJILLA_NEG = 60
N_REJILLA_POS = 60
TOL_FRONTERA = 1e-9


def _perfil(theta: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ℓ*(θ), ξ(θ)) en cada θ; −∞ donde no hay solución con σ > 0."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = x.size
    ll = np.full(theta.shape, -np.inf)
    xi = np.full(theta.shape, np.nan)
    cero = np.abs(theta) < 1e-12
    if np.any(cero):
        ll[cero] = -n * np.log(x.mean()) - n
        xi[cero] = 0.0
    t = theta[~cero]
    if t.size:
        z = 1.0 + np.outer(t, x)
        ok = np.all(z > 0, axis=1)
        k = np.full(t.shape, np.nan)
        k[ok] = np.log(z[ok]).mean(axis=1)
        s = k / t
        bien = ok & (s > 0)
        v = np.full(t.shape, -np.inf)
        v[bien] = -n * np.log(s[bien]) - n - n * k[bien]
        ll[~cero], xi[~cero] = v, k
    return ll, xi


def rejilla_theta(x: np.ndarray) -> np.ndarray:
    """Rejilla fija de θ: del lado negativo, θ = −a/máx x con a de 1 − 10⁻⁶ a 10⁻⁶ (densa en los dos extremos);
    del positivo, de 10⁻⁴/media a 50/media (ξ hasta ≈ 4)."""
    xmax, m = float(x.max()), float(x.mean())
    mitad = N_REJILLA_NEG // 2
    a = np.concatenate([1.0 - np.geomspace(1e-6, 0.5, mitad), np.geomspace(0.5, 1e-6, mitad + 1)[1:]])
    neg = -a / xmax
    pos = np.geomspace(1e-4, 50.0, N_REJILLA_POS) / m
    return np.sort(np.concatenate([neg, [0.0], pos]))


def ajustar_rapido(excesos, n_min: int = gpd.N_MIN) -> gpd.AjusteGPD:
    """MLE de la GPD por perfil; delega en ``gpd.ajustar_gpd`` si no hay un máximo interior claro."""
    x = gpd._comprobar(excesos)
    if x.size < n_min or float(x.max()) <= 0:
        return gpd.ajustar_gpd(x, n_min)
    th = rejilla_theta(x)
    ll, xis = _perfil(th, x)
    ll[~(xis > XI_HACIA_FRONTERA)] = -np.inf  # fuera de ξ ≥ −1 el perfil crece sin cota hacia el soporte
    if not np.any(np.isfinite(ll)):
        return gpd.ajustar_gpd(x, n_min)
    i = int(np.argmax(ll))
    if i == 0 or i == th.size - 1 or not (np.isfinite(ll[i - 1]) and np.isfinite(ll[i + 1])):
        return gpd.ajustar_gpd(x, n_min)  # máximo en el borde de la región válida: frontera o casi
    a, b = float(th[i - 1]), float(th[i + 1])
    r = optimize.minimize_scalar(
        lambda t: -_perfil(np.array([t]), x)[0][0], bounds=(a, b), method="bounded", options={"xatol": 1e-14}
    )
    t = float(r.x) if -r.fun >= ll[i] else float(th[i])
    llp, xip = _perfil(np.array([t]), x)
    xi = float(xip[0])
    if not np.isfinite(llp[0]) or not np.isfinite(xi) or xi <= XI_HACIA_FRONTERA:
        return gpd.ajustar_gpd(x, n_min)
    sigma = float(x.mean()) if abs(t) < 1e-12 else xi / t
    ll_fin = gpd.loglik(x, xi, sigma)
    ll_frontera = gpd.loglik(x, gpd.XI_MIN, float(x.max()))
    if not np.isfinite(ll_fin) or ll_fin <= ll_frontera + TOL_FRONTERA:
        return gpd.ajustar_gpd(x, n_min)
    extremo = -sigma / xi if xi < 0 else np.inf
    if extremo <= float(x.max()):
        return gpd.ajustar_gpd(x, n_min)
    return gpd.AjusteGPD(xi, sigma, ll_fin, x.size, True, 1, "perfil")
