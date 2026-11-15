"""IC por perfil de verosimilitud: RL_T de uno y de varios estratos, ξ y σ.

Verosimilitud de un estrato (n eventos en E años con dato)::

    ℓ(λ, σ, ξ) = n·log(λE) − λE + Σ log g(x_i; σ, ξ),

Poisson para el conteo y GPD para los excesos. El RL_T del estrato es u + z, con z = σ/ξ·[(λT)^ξ − 1]. Para el
perfil de z, σ se despeja de (z, ξ, λ) y se maximiza en (ξ, λ) con **λ libre** y ξ ≥ −1; el soporte se comprueba
observación por observación (``gpd.loglik`` vale −∞ fuera de él). Con varios estratos se maximiza la verosimilitud
conjunta sujeta a Σ_s λ_s·S_s(z) = 1/T, con **todas las λ_s libres**.

IC = {θ : 2·[ℓ̂ − ℓ_p(θ)] ≤ χ²₁(nivel)}. Topes de búsqueda: el RL hasta u + 20·(RL̂ − u) y hasta el umbral por
abajo; ξ hasta 1.5 y hasta −1; σ entre σ̂/20 y 20·σ̂. Un límite que no se alcanza se devuelve **abierto** (NaN)
con su causa, nunca como número:

- ``tope``: el déficit sigue bajo χ²₁ al llegar al tope; el IC es [lo, ∞) (o (umbral, hi]) de ese lado;
- ``frontera``: ξ llega a −1;
- ``borde_de_factibilidad``: el perfil deja de ser factible antes de alcanzar χ²₁;
- ``sin_convergencia``: el optimizador no cumplió la restricción; el límite es desconocido.

Con ξ̂ ≤ −0.5 la calibración χ²₁ no está garantizada (Smith, 1985); el perfil de ξ marca **no regular** cuando su
IC incluye valores ≤ −0.5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .gpd import XI_CERO, XI_MIN, XI_NO_REGULAR, AjusteGPD, cuantil_supervivencia, loglik, sigma_dado_xi, supervivencia
from .retorno import EstratoRetorno, nivel_retorno_multiestrato

XI_MAX = 1.5
FACTOR_TOPE = 20.0
_REJILLA_XI = np.linspace(XI_MIN, XI_MAX, 101)
TOPE = "tope"
FRONTERA = "frontera"
BORDE = "borde_de_factibilidad"
SIN_CONVERGENCIA = "sin_convergencia"


@dataclass(frozen=True)
class IntervaloPerfil:
    punto: float
    lo: float  # NaN si el límite queda abierto
    hi: float
    causa_lo: str | None
    causa_hi: str | None
    nivel: float
    loglik_max: float

    @property
    def abierto(self) -> bool:
        return self.causa_lo is not None or self.causa_hi is not None


@dataclass(frozen=True)
class PerfilXi(IntervaloPerfil):
    no_regular: bool = False  # ξ̂ ≤ −0.5 o el IC incluye valores ≤ −0.5


@dataclass(frozen=True)
class PerfilRL(IntervaloPerfil):
    tasa_por_anio: float = np.nan  # n/E
    delta_loglik_en_punto: float = np.nan  # ℓ_p(ẑ) − ℓ̂: debe ser ≈ 0
    deficit_min: float = np.nan  # mínimo de 2·[ℓ̂ − ℓ_p] evaluado: < 0 indica que el ajuste no era el máximo
    n_rescates: int = 0
    n_sin_resolver: int = 0


def loglik_estrato(x: np.ndarray, n: int, E: float, xi: float, sigma: float, lam: float) -> float:
    if not (lam > 0 and E > 0):
        return -np.inf
    ll = loglik(x, xi, sigma)
    return float(n * np.log(lam * E) - lam * E + ll) if np.isfinite(ll) else -np.inf


def _sigma_de_z(z: float, xi: float, lt: float) -> float:
    """σ tal que el RL_T del estrato es u + z, con lt = λT > 1."""
    if abs(xi) < XI_CERO:
        return z / np.log(lt)
    return z * xi / (np.power(lt, xi) - 1.0)


def _cota_log_lambda(z: float, xi: float, T: float, xmax: float) -> float:
    """log λ máximo con soporte: con ξ < 0 el extremo superior z/(1 − (λT)^ξ) debe superar al exceso máximo."""
    if xi >= 0 or z >= xmax:
        return np.inf
    return float(np.log((1.0 - z / xmax) ** (1.0 / xi) / T))


def _max_en_lambda(x, n, E, T, z, xi, log_lam_hat) -> tuple[float, float]:
    lo = np.log(1.0 / T) + 1e-9
    hi = min(log_lam_hat + 4.0, _cota_log_lambda(z, xi, T, float(x.max())) - 1e-12)
    if not hi > lo:
        return -np.inf, np.nan

    def f(l_):
        lam = float(np.exp(l_))
        v = loglik_estrato(x, n, E, xi, _sigma_de_z(z, xi, lam * T), lam)
        return -v if np.isfinite(v) else 1e300

    r = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    v = -float(r.fun)
    return (v, float(r.x)) if v > -1e299 else (-np.inf, np.nan)


def _perfil_z(x, n, E, T, z, log_lam_hat, inicios) -> tuple[float, float]:
    """ℓ_p(z) = max en (ξ, λ): rejilla de ξ con λ óptimo y pulido Nelder-Mead desde el mejor punto y los inicios."""
    mejor = (-np.inf, np.nan, np.nan)
    for xi in _REJILLA_XI:
        v, l_ = _max_en_lambda(x, n, E, T, z, xi, log_lam_hat)
        if v > mejor[0]:
            mejor = (v, xi, l_)

    def neg(th):
        xi, l_ = th
        if not (XI_MIN <= xi <= XI_MAX) or not np.isfinite(l_) or np.exp(l_) * T <= 1.0:
            return 1e300
        lam = float(np.exp(l_))
        v = loglik_estrato(x, n, E, xi, _sigma_de_z(z, xi, lam * T), lam)
        return -v if np.isfinite(v) else 1e300

    puntos = ([(mejor[1], mejor[2])] if np.isfinite(mejor[0]) else []) + [a for a in inicios if np.all(np.isfinite(a))]
    for a in puntos:
        r = optimize.minimize(
            neg,
            np.array(a, float),
            method="Nelder-Mead",
            options={"xatol": 1e-11, "fatol": 1e-12, "maxiter": 4000, "maxfev": 8000},
        )
        if -r.fun > mejor[0]:
            mejor = (-float(r.fun), float(r.x[0]), float(r.x[1]))
        if neg(a) < 1e300 and -neg(a) > mejor[0]:
            mejor = (-float(neg(a)), float(a[0]), float(a[1]))
    v, l_ = _max_en_lambda(x, n, E, T, z, XI_MIN, log_lam_hat)  # la frontera exacta, que el pulido no alcanza
    if v > mejor[0]:
        mejor = (v, XI_MIN, l_)
    return mejor[0], mejor[2]


def _limite(
    deficit: Callable[[float], float], punto: float, q: float, tope: float, n_pasos: int = 40
) -> tuple[float, str | None]:
    """Raíz de deficit(θ) = q entre el punto y el tope, en pasos geométricos desde el punto.

    (límite, None) si la alcanza; (NaN, BORDE) si el perfil deja de ser factible antes; (NaN, TOPE) si llega al
    tope sin alcanzarla.
    """
    prev = punto
    d = tope - punto
    for k in range(1, n_pasos + 1):
        frac = min(1.0, 0.02 * 1.35 ** (k - 1))
        th = punto + frac * d
        dv = deficit(th)
        if not np.isfinite(dv):
            lo, hi = prev, th
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                dm = deficit(mid)
                if np.isfinite(dm) and dm < q:
                    lo = mid
                elif np.isfinite(dm):
                    return float(optimize.brentq(lambda t: deficit(t) - q, prev, mid, xtol=1e-9, rtol=1e-10)), None
                else:
                    hi = mid
            return np.nan, BORDE
        if dv >= q:
            return float(optimize.brentq(lambda t: deficit(t) - q, prev, th, xtol=1e-9, rtol=1e-10)), None
        prev = th
        if frac >= 1.0:
            break
    return np.nan, TOPE


def perfil_rl(
    excesos, n: int, anios_con_dato: float, T: float, u: float, ajuste: AjusteGPD, nivel: float = 0.95
) -> PerfilRL:
    """IC por perfil del RL_T de un estrato (u + z) con λ libre. ``ajuste`` es el de ``ajustar_gpd`` sobre los
    mismos excesos; ``n`` el número de eventos; ``anios_con_dato`` los años de exposición (E)."""
    x = np.asarray(excesos, dtype=float)
    E = float(anios_con_dato)
    if n != x.size:
        raise ValueError("n debe ser el número de excesos")
    if not ajuste.valido:
        raise ValueError("El perfil requiere un ajuste válido")
    lam_hat = n / E
    if lam_hat * T <= 1.0:
        raise ValueError(f"λT = {lam_hat * T:.3g} ≤ 1: el RL_T está bajo el umbral")
    z_hat = float(cuantil_supervivencia(1.0 / (lam_hat * T), ajuste.xi, ajuste.sigma))
    l_hat = loglik_estrato(x, n, E, ajuste.xi, ajuste.sigma, lam_hat)
    inicio = [(ajuste.xi, float(np.log(lam_hat)))]
    q = float(stats.chi2.ppf(nivel, 1))
    evaluados: list[float] = []

    def deficit(z: float) -> float:
        if z <= 0:
            return np.inf
        v, _ = _perfil_z(x, n, E, T, z, float(np.log(lam_hat)), inicio)
        d = 2.0 * (l_hat - v) if np.isfinite(v) else np.inf
        evaluados.append(d)
        return d

    v_hat, _ = _perfil_z(x, n, E, T, z_hat, float(np.log(lam_hat)), inicio)
    hi, c_hi = _limite(deficit, z_hat, q, FACTOR_TOPE * z_hat)
    lo, c_lo = _limite(deficit, z_hat, q, z_hat * 1e-6)
    return PerfilRL(
        punto=u + z_hat,
        lo=u + lo,
        hi=u + hi,
        causa_lo=c_lo,
        causa_hi=c_hi,
        nivel=nivel,
        loglik_max=l_hat,
        tasa_por_anio=lam_hat,
        delta_loglik_en_punto=float(v_hat - l_hat),
        deficit_min=float(min([2.0 * (l_hat - v_hat), *evaluados])),
    )


# --- Varios estratos --------------------------------------------------------------------------------

_ETA_MIN = -30.0
_TOL_RESTRICCION = 1e-8
_N_RESCATE = 12
_ESCALA_PERTURBACION = (0.05, 0.2, 0.05)


class _Multi:
    """ℓ conjunta y restricción en θ_s = (ξ_s, η_s, log λ_s), con σ_s = max(0, −ξ_s)·x_max,s + e^η_s: el soporte se
    cumple siempre y el optimizador no choca con un precipicio de −∞."""

    def __init__(self, xs, n, E, T):
        self.xs, self.n, self.E, self.T = xs, n, E, T
        self.xm = [float(x.max()) for x in xs]
        self.k = len(xs)
        self.cotas = [(XI_MIN, XI_MAX), (_ETA_MIN, 5.0), (-10.0, 5.0)] * self.k
        self.lim = np.array(self.cotas, float).T

    def a_theta(self, tri) -> np.ndarray:
        out = []
        for j, (xi, sig, lam) in enumerate(tri):
            holgura = sig - max(0.0, -xi) * self.xm[j]
            out += [xi, float(np.log(holgura)) if holgura > np.exp(_ETA_MIN) else _ETA_MIN, np.log(lam)]
        return np.array(out, float)

    def de_theta(self, th):
        return [
            (
                float(th[3 * j]),
                max(0.0, -float(th[3 * j])) * self.xm[j] + float(np.exp(th[3 * j + 1])),
                float(np.exp(th[3 * j + 2])),
            )
            for j in range(self.k)
        ]

    def loglik(self, th) -> float:
        tot = 0.0
        for j, (xi, sig, lam) in enumerate(self.de_theta(th)):
            v = loglik_estrato(self.xs[j], self.n[j], self.E[j], xi, sig, lam)
            if not np.isfinite(v):
                return -np.inf
            tot += v
        return tot

    def restriccion(self, th, z: float) -> float:
        return (
            self.T * sum(lam * float(supervivencia(np.array([z]), xi, sig)[0]) for xi, sig, lam in self.de_theta(th))
            - 1.0
        )

    def _slsqp(self, z: float, th0: np.ndarray) -> tuple[float, np.ndarray | None]:
        def neg(th):
            v = self.loglik(th)
            return -v if np.isfinite(v) else 1e10

        r = optimize.minimize(
            neg,
            th0,
            method="SLSQP",
            bounds=self.cotas,
            constraints=[{"type": "eq", "fun": lambda th: self.restriccion(th, z)}],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        v = self.loglik(r.x)
        return (v, r.x) if np.isfinite(v) and abs(self.restriccion(r.x, z)) < _TOL_RESTRICCION else (-np.inf, None)

    def maximo(self, z: float, inicios: list[np.ndarray]) -> tuple[float, np.ndarray | None, bool]:
        """max ℓ sujeto a la restricción en z. Si ningún inicio la cumple, rescate con inicios perturbados
        (semilla derivada de z, determinista). Devuelve (ℓ, θ, hubo_rescate)."""
        mejor: tuple[float, np.ndarray | None] = (-np.inf, None)
        for th0 in inicios:
            v, th = self._slsqp(z, th0)
            if v > mejor[0]:
                mejor = (v, th)
        if mejor[1] is not None:
            return mejor[0], mejor[1], False
        rng = np.random.default_rng([int(np.round(z * 1e9)) % (2**63), 2026])
        escala = np.tile(_ESCALA_PERTURBACION, self.k)
        for _ in range(_N_RESCATE):
            v, th = self._slsqp(z, np.clip(inicios[0] + rng.normal(0.0, escala), *self.lim))
            if v > mejor[0]:
                mejor = (v, th)
        return mejor[0], mejor[1], True


def perfil_rl_multiestrato(
    excesos: Sequence,
    n: Sequence[int],
    anios_con_dato: Sequence[float],
    T: float,
    u: float,
    ajustes: Sequence[AjusteGPD],
    nivel: float = 0.95,
) -> PerfilRL:
    """IC por perfil del RL_T con varios estratos y todas las λ_s libres. Un elemento por estrato en ``excesos``,
    ``n``, ``anios_con_dato`` y ``ajustes``; las tasas del ajuste son λ̂_s = n_s/E_s."""
    xs = [np.asarray(x, dtype=float) for x in excesos]
    n, E = [int(v) for v in n], [float(v) for v in anios_con_dato]
    if not xs or not (len(xs) == len(n) == len(E) == len(ajustes)):
        raise ValueError("excesos, n, anios_con_dato y ajustes deben tener un elemento por estrato")
    if any(nj != x.size or nj == 0 for nj, x in zip(n, xs, strict=True)):
        raise ValueError("n de cada estrato debe ser su número de excesos (> 0)")
    if not all(a.valido for a in ajustes):
        raise ValueError("El perfil requiere ajustes válidos en todos los estratos")
    lam = [nj / ej for nj, ej in zip(n, E, strict=True)]
    rl = nivel_retorno_multiestrato(
        u, [EstratoRetorno(lj, a.xi, a.sigma) for lj, a in zip(lam, ajustes, strict=True)], T
    )
    if not np.isfinite(rl):
        raise ValueError("El RL_T con varios estratos no tiene solución sobre el umbral")
    z_hat = rl - u
    m = _Multi(xs, n, E, T)
    th_hat = m.a_theta([(a.xi, a.sigma, lj) for a, lj in zip(ajustes, lam, strict=True)])
    l_hat = m.loglik(th_hat)
    q = float(stats.chi2.ppf(nivel, 1))
    estado = {"th": th_hat, "rescates": 0, "sin_resolver": 0}
    memo: dict[float, float] = {}

    def ell(z: float) -> float:
        v, th, rescate = m.maximo(z, [estado["th"], th_hat])
        estado["rescates"] += int(rescate)
        if th is not None:
            estado["th"] = th
        else:
            estado["sin_resolver"] += 1
        return v

    def deficit(z: float) -> float:
        # Memorizado: con arranque en caliente ℓ_p(z) depende del camino; brentq vuelve a evaluar los extremos.
        if z <= 0:
            return np.inf
        if z not in memo:
            v = ell(z)
            memo[z] = 2.0 * (l_hat - v) if np.isfinite(v) else np.inf
        return memo[z]

    v_hat = ell(z_hat)
    if not np.isfinite(v_hat):
        raise RuntimeError("Perfil con varios estratos sin convergencia en el RL puntual")
    hi, c_hi = _limite(deficit, z_hat, q, FACTOR_TOPE * z_hat)
    estado["th"] = th_hat
    lo, c_lo = _limite(deficit, z_hat, q, z_hat * 1e-6)
    # Dentro de las cotas la restricción siempre es alcanzable: un déficit infinito es un fallo del optimizador.
    c_hi = SIN_CONVERGENCIA if c_hi == BORDE else c_hi
    c_lo = SIN_CONVERGENCIA if c_lo == BORDE else c_lo
    return PerfilRL(
        punto=rl,
        lo=u + lo,
        hi=u + hi,
        causa_lo=c_lo,
        causa_hi=c_hi,
        nivel=nivel,
        loglik_max=l_hat,
        tasa_por_anio=float(sum(lam)),
        delta_loglik_en_punto=float(v_hat - l_hat),
        deficit_min=float(min([2.0 * (l_hat - v_hat), *memo.values()])),
        n_rescates=estado["rescates"],
        n_sin_resolver=estado["sin_resolver"],
    )


# --- ξ y σ ----------------------------------------------------------------------------------------


def _perfil_sigma_1(x: np.ndarray, sigma: float) -> float:
    """max_ξ ℓ(ξ, σ) con ξ ∈ [max(−1, −σ/máx x), XI_MAX]."""
    lo = max(XI_MIN, -sigma / float(x.max()))

    def f(xi):
        v = loglik(x, xi, sigma)
        return -v if np.isfinite(v) else 1e300

    r = optimize.minimize_scalar(f, bounds=(lo, XI_MAX), method="bounded", options={"xatol": 1e-11})
    v = -float(r.fun)
    return max(v if v > -1e299 else -np.inf, loglik(x, lo, sigma))


def perfil_xi(excesos, ajuste: AjusteGPD, nivel: float = 0.95) -> PerfilXi:
    x = np.asarray(excesos, dtype=float)
    q = float(stats.chi2.ppf(nivel, 1))
    l_hat = ajuste.loglik

    def deficit(xi: float) -> float:
        v = sigma_dado_xi(x, xi)[1]
        return 2.0 * (l_hat - v) if np.isfinite(v) else np.inf

    hi, c_hi = _limite(deficit, ajuste.xi, q, XI_MAX)
    if ajuste.xi <= XI_MIN:
        lo, c_lo = np.nan, FRONTERA
    else:
        lo, c_lo = _limite(deficit, ajuste.xi, q, XI_MIN)
        c_lo = FRONTERA if c_lo == TOPE else c_lo
    no_regular = bool(ajuste.no_regular or c_lo is not None or lo <= XI_NO_REGULAR)
    return PerfilXi(ajuste.xi, lo, hi, c_lo, c_hi, nivel, l_hat, no_regular)


def perfil_sigma(excesos, ajuste: AjusteGPD, nivel: float = 0.95) -> IntervaloPerfil:
    x = np.asarray(excesos, dtype=float)
    q = float(stats.chi2.ppf(nivel, 1))
    l_hat = ajuste.loglik

    def deficit(log_s: float) -> float:
        v = _perfil_sigma_1(x, float(np.exp(log_s)))
        return 2.0 * (l_hat - v) if np.isfinite(v) else np.inf

    ls = float(np.log(ajuste.sigma))
    hi, c_hi = _limite(deficit, ls, q, ls + np.log(FACTOR_TOPE))
    lo, c_lo = _limite(deficit, ls, q, ls - np.log(FACTOR_TOPE))
    return IntervaloPerfil(ajuste.sigma, float(np.exp(lo)), float(np.exp(hi)), c_lo, c_hi, nivel, l_hat)
