"""GPD, niveles de retorno y perfiles: pruebas de respuesta conocida (contrato de la F2, §5.1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, stats

from peruvian_wave_analysis.extremos import gpd, perfil, retorno
from peruvian_wave_analysis.extremos.retorno import EstratoRetorno


def _simular(xi, sigma, n, semilla):
    return stats.genpareto.rvs(c=xi, scale=sigma, size=n, random_state=np.random.default_rng(semilla))


def _ll_vec(x, xi, sigma):
    """Log-verosimilitud GPD vectorizada en (ξ, σ) para las pruebas (implementación independiente)."""
    xi = np.asarray(xi, float)[..., None]
    sigma = np.asarray(sigma, float)[..., None]
    z = 1.0 + xi * x / sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = -np.log(sigma) - (1.0 / xi + 1.0) * np.log(z)
        exp = -np.log(sigma) - x / sigma
    dens = np.where(np.abs(xi) < 1e-8, exp, dens)
    dens = np.where(z > 0, dens, -np.inf)
    return dens.sum(axis=-1)


# --- GPD ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [-0.9, -0.3, 0.0, 0.3])
def test_loglik_igual_a_scipy(xi):
    sigma = 1.3
    top = 0.95 * (-sigma / xi) if xi < 0 else 8.0
    x = np.linspace(0.0, top, 50)
    esperado = stats.genpareto.logpdf(x, c=xi, scale=sigma).sum()
    assert abs(gpd.loglik(x, xi, sigma) - esperado) < 1e-10


def test_loglik_fuera_de_soporte_y_sigma_invalida():
    assert gpd.loglik([0.5, 2.0], -0.5, 0.5) == -np.inf  # extremo superior 1.0 < 2.0
    assert gpd.loglik([0.5], 0.1, 0.0) == -np.inf
    assert gpd.loglik([0.5], -1.2, 1.0) == -np.inf
    with pytest.raises(ValueError):
        gpd.ajustar_gpd([1.0, -0.1] * 10)
    with pytest.raises(ValueError):
        gpd.ajustar_gpd([1.0, np.nan] * 10)


def test_supervivencia_y_cuantil_inversos():
    for xi in (-0.4, 0.0, 0.2):
        p = np.array([0.5, 0.1, 0.01])
        x = gpd.cuantil_supervivencia(p, xi, 1.5)
        np.testing.assert_allclose(gpd.supervivencia(x, xi, 1.5), p, rtol=1e-12)
        np.testing.assert_allclose(gpd.supervivencia(x, xi, 1.5), stats.genpareto.sf(x, c=xi, scale=1.5), rtol=1e-12)


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_ajuste_recupera_parametros(xi):
    n = 2000
    a = gpd.ajustar_gpd(_simular(xi, 1.0, n, 11))
    assert a.valido and not a.frontera
    ee_xi = (1 + xi) / np.sqrt(n)
    ee_sigma = np.sqrt(2 * (1 + xi) / n)
    assert abs(a.xi - xi) < 3 * ee_xi
    assert abs(a.sigma - 1.0) < 3 * ee_sigma


def test_ajuste_es_maximo_global_frente_a_rejilla():
    x = _simular(-0.2, 1.0, 100, 5)
    a = gpd.ajustar_gpd(x)
    xis = np.linspace(-0.99, 0.8, 301)
    sigmas = np.exp(np.linspace(np.log(0.3), np.log(3.0), 301))
    XI, SI = np.meshgrid(xis, sigmas, indexing="ij")
    rejilla = _ll_vec(x, XI, SI)
    assert a.loglik >= rejilla.max() - 1e-6


def test_soporte_en_muestras_pequenas():
    rng = np.random.default_rng(2026)
    fallidos = 0
    for _ in range(2000):
        n = int(rng.integers(10, 31))
        xi = float(rng.uniform(-0.9, 0.5))
        x = stats.genpareto.rvs(c=xi, scale=1.0, size=n, random_state=rng)
        a = gpd.ajustar_gpd(x)
        if not a.valido:
            fallidos += 1
            assert a.causa in {"fuera_de_soporte", "sin_convergencia"}
            continue
        assert np.isfinite(a.loglik)
        if a.frontera:
            assert a.xi == -1.0 and a.sigma == x.max() and a.extremo_superior == x.max() and a.no_regular
        else:
            assert a.extremo_superior > x.max()
    assert fallidos < 20  # informativo: el contrato pide causa, no cero fallos


def test_n_insuficiente():
    a = gpd.ajustar_gpd([1.0, 2.0, 3.0])
    assert not a.valido and a.causa == "n_insuficiente" and np.isnan(a.xi)


def test_frontera_confirmada_por_rejilla():
    # Excesos concentrados cerca del máximo: n + Σ log(1 − x/x_max) < 0, el máximo restringido está en ξ = −1.
    x = np.concatenate([np.linspace(0.9, 1.0, 19), [0.2]])
    xis = np.linspace(-0.999, 0.5, 600)
    sigmas = np.exp(np.linspace(np.log(0.5), np.log(5.0), 600))
    XI, SI = np.meshgrid(xis, sigmas, indexing="ij")
    ll_frontera = gpd.loglik(x, -1.0, x.max())
    assert _ll_vec(x, XI, SI).max() < ll_frontera  # la rejilla confirma que el interior no supera a la frontera
    a = gpd.ajustar_gpd(x)
    assert a.valido and a.frontera and a.xi == -1.0 and a.sigma == x.max() and a.no_regular


@pytest.mark.parametrize("xi", [0.1, -0.2])
def test_pwm_recupera_parametros(xi):
    pwm = gpd.arranque_pwm(_simular(xi, 1.0, 20000, 3))
    assert abs(pwm[0] - xi) < 0.03 and abs(pwm[1] - 1.0) < 0.03


def test_no_regular():
    assert gpd.AjusteGPD(-0.5, 1.0, 0.0, 10, True, 1, "x").no_regular
    assert not gpd.AjusteGPD(-0.49, 1.0, 0.0, 10, True, 1, "x").no_regular


# --- Retorno ------------------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_retorno_un_estrato_igual_a_multiestrato(xi):
    for T in (1.0, 10.0, 100.0):
        a = retorno.nivel_retorno(2.0, xi, 0.8, 12.0, T)
        b = retorno.nivel_retorno_multiestrato(2.0, [EstratoRetorno(12.0, xi, 0.8)], T)
        assert abs(a - b) < 1e-9
        assert abs(retorno.tasa_de_excedencia(a, 2.0, [EstratoRetorno(12.0, xi, 0.8)]) - 1 / T) < 1e-12


def test_retorno_dos_estratos_iguales_es_tasa_doble():
    e = EstratoRetorno(5.0, -0.1, 0.7)
    a = retorno.nivel_retorno_multiestrato(1.0, [e, e], 50.0)
    b = retorno.nivel_retorno(1.0, -0.1, 0.7, 10.0, 50.0)
    assert abs(a - b) < 1e-9


def test_retorno_bajo_el_umbral_y_probabilidad():
    assert np.isnan(retorno.nivel_retorno(1.0, 0.0, 1.0, 0.5, 1.0))
    assert np.isnan(retorno.nivel_retorno_multiestrato(1.0, [EstratoRetorno(0.5, 0.0, 1.0)], 1.0))
    assert abs(retorno.probabilidad_en_n_anios(100, 50) - (1 - np.exp(-0.5))) < 1e-15


# --- Perfiles -----------------------------------------------------------------------------------


def _sigma_max_rejilla(x, xi):
    """max_σ ℓ por rejilla densa en log σ (independiente del optimizador acotado del código)."""
    xmax = x.max()
    lo = np.log(-xi * xmax * (1 + 1e-12)) if xi < 0 else np.log(x.mean()) - 4
    s = np.exp(np.linspace(lo, np.log(x.mean()) + 4, 20001))
    ll = _ll_vec(x, np.full_like(s, xi), s)
    i = int(np.argmax(ll))
    r = optimize.minimize_scalar(
        lambda ls: -_ll_vec(x, xi, np.exp(ls)),
        bounds=(np.log(s[max(i - 1, 0)]), np.log(s[min(i + 1, s.size - 1)])),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return max(ll[i], -float(r.fun))


def test_perfil_xi_frente_a_fuerza_bruta():
    x = _simular(-0.1, 1.0, 150, 8)
    a = gpd.ajustar_gpd(x)
    ic = perfil.perfil_xi(x, a)
    q = stats.chi2.ppf(0.95, 1)

    def deficit(xi):
        return 2 * (a.loglik - _sigma_max_rejilla(x, xi)) - q

    lo = optimize.brentq(deficit, a.xi - 0.8, a.xi, xtol=1e-8)
    hi = optimize.brentq(deficit, a.xi, a.xi + 0.8, xtol=1e-8)
    assert abs(ic.lo - lo) < 1e-3 and abs(ic.hi - hi) < 1e-3
    assert ic.causa_lo is None and ic.causa_hi is None


def test_perfil_xi_cerca_de_menos_uno():
    x = np.concatenate([np.linspace(0.9, 1.0, 19), [0.2]])
    a = gpd.ajustar_gpd(x)
    ic = perfil.perfil_xi(x, a)
    assert np.isnan(ic.lo) and ic.causa_lo == "frontera" and ic.no_regular


def test_perfil_xi_marca_no_regular_por_intervalo():
    x = _simular(-0.35, 1.0, 40, 21)
    a = gpd.ajustar_gpd(x)
    ic = perfil.perfil_xi(x, a)
    assert a.xi > -0.5 and (ic.lo <= -0.5 or ic.causa_lo is not None)
    assert ic.no_regular


def _ll_estrato_vec(x, n, E, xi, sigma, lam):
    return n * np.log(lam * E) - lam * E + _ll_vec(x, xi, sigma)


def _sigma_de_z(z, xi, lt):
    with np.errstate(divide="ignore", invalid="ignore"):
        s = z * xi / (np.power(lt, xi) - 1.0)
    return np.where(np.abs(xi) < 1e-8, z / np.log(lt), s)


def _perfil_rl_fuerza_bruta(x, n, E, T, z, lam_hat):
    """max en (ξ, λ) por rejilla fina y pulido con Powell (optimizador distinto del Nelder-Mead del código)."""
    xis = np.linspace(-0.995, 1.0, 400)
    lls = np.log(lam_hat) + np.linspace(-1.2, 1.2, 241)
    XI, LL = np.meshgrid(xis, lls, indexing="ij")
    lam = np.exp(LL)
    S = _sigma_de_z(z, XI, lam * T)
    v = np.where(S > 0, _ll_estrato_vec(x, n, E, XI, np.where(S > 0, S, 1.0), lam), -np.inf)
    i, j = np.unravel_index(np.argmax(v), v.shape)

    def neg(th):
        xi, l_ = th
        if not -1 < xi < 1.5 or np.exp(l_) * T <= 1:
            return 1e300
        s = float(_sigma_de_z(z, np.array(xi), np.exp(l_) * T))
        if s <= 0:
            return 1e300
        r = float(_ll_estrato_vec(x, n, E, np.array(xi), np.array(s), np.exp(l_)))
        return -r if np.isfinite(r) else 1e300

    r = optimize.minimize(neg, [XI[i, j], LL[i, j]], method="Powell", options={"xtol": 1e-10, "ftol": 1e-13})
    return max(float(v[i, j]), -float(r.fun))


def test_perfil_rl_un_estrato_frente_a_fuerza_bruta():
    E, T, u = 30.0, 100.0, 2.0
    x = _simular(-0.1, 0.8, 240, 13)
    n = x.size
    a = gpd.ajustar_gpd(x)
    ic = perfil.perfil_rl(x, n, E, T, u, a)
    lam_hat = n / E
    assert abs(ic.punto - retorno.nivel_retorno(u, a.xi, a.sigma, lam_hat, T)) < 1e-9
    assert abs(ic.delta_loglik_en_punto) < 1e-6 and ic.deficit_min > -1e-6
    l_hat = perfil.loglik_estrato(x, n, E, a.xi, a.sigma, lam_hat)
    q = stats.chi2.ppf(0.95, 1)

    def deficit(rl):
        return 2 * (l_hat - _perfil_rl_fuerza_bruta(x, n, E, T, rl - u, lam_hat)) - q

    lo = optimize.brentq(deficit, u + 0.3 * (ic.punto - u), ic.punto - 1e-6, xtol=1e-7)
    hi = optimize.brentq(deficit, ic.punto + 1e-6, u + 3.0 * (ic.punto - u), xtol=1e-7)
    assert abs(ic.lo - lo) / lo < 1e-3 and abs(ic.hi - hi) / hi < 1e-3


def test_perfil_rl_multiestrato_con_un_estrato_igual_al_de_un_estrato():
    E, T, u = 30.0, 100.0, 2.0
    x = _simular(-0.1, 0.8, 240, 13)
    a = gpd.ajustar_gpd(x)
    uno = perfil.perfil_rl(x, x.size, E, T, u, a)
    multi = perfil.perfil_rl_multiestrato([x], [x.size], [E], T, u, [a])
    assert abs(multi.punto - uno.punto) / uno.punto < 1e-6
    assert abs(multi.lo - uno.lo) / uno.lo < 1e-6 and abs(multi.hi - uno.hi) / uno.hi < 1e-6


def _sigma_de_p(z, xi, p):
    """σ con S(z; ξ, σ) = p."""
    return float(z / -np.log(p)) if abs(xi) < 1e-8 else float(z * xi / (p ** (-xi) - 1.0))


def _perfil_multi_fuerza_bruta(xs, n, E, T, z, ajustes):
    """Dos estratos: la σ de un estrato se despeja de la restricción y se maximiza sin restricción en los otros cinco
    parámetros con Powell desde varios inicios (formulación distinta del SLSQP del código). Se prueban los dos
    despejes, porque en la cola un estrato acotado puede tener S(z) = 0 y entonces solo el otro puede despejarse."""

    def neg_factory(libre, despejado):
        xl, xd = xs[libre], xs[despejado]

        def neg(th):
            xi_l, ls_l, l_l, xi_d, l_d = th
            if not (-1 < xi_l < 1.5 and -1 < xi_d < 1.5):
                return 1e300
            s_l, lam_l, lam_d = np.exp(ls_l), np.exp(l_l), np.exp(l_d)
            p_d = (1.0 / T - lam_l * gpd.supervivencia(np.array([z]), xi_l, s_l)[0]) / lam_d
            if not 0 < p_d < 1:
                return 1e300
            s_d = _sigma_de_p(z, xi_d, p_d)
            if s_d <= 0:
                return 1e300
            v = perfil.loglik_estrato(xl, n[libre], E[libre], xi_l, s_l, lam_l) + perfil.loglik_estrato(
                xd, n[despejado], E[despejado], xi_d, s_d, lam_d
            )
            return -v if np.isfinite(v) else 1e300

        return neg

    mejor = np.inf
    for libre, despejado in ((0, 1), (1, 0)):
        al, ad = ajustes[libre], ajustes[despejado]
        neg = neg_factory(libre, despejado)
        for d in (0.0, -0.15, 0.15):
            for f in (0.0, -0.3, 0.3):
                th0 = [
                    al.xi + d,
                    np.log(al.sigma),
                    np.log(n[libre] / E[libre]) + f,
                    ad.xi - d,
                    np.log(n[despejado] / E[despejado]) - f,
                ]
                r = optimize.minimize(
                    neg, th0, method="Powell", options={"xtol": 1e-10, "ftol": 1e-13, "maxfev": 40000}
                )
                mejor = min(mejor, float(r.fun))
    return -mejor


def test_perfil_rl_multiestrato_dos_estratos_frente_a_fuerza_bruta():
    T, u = 100.0, 2.0
    x1, x2 = _simular(-0.2, 1.0, 180, 31), _simular(0.0, 0.7, 120, 32)
    n, E = [180, 120], [30.0, 30.0]
    a1, a2 = gpd.ajustar_gpd(x1), gpd.ajustar_gpd(x2)
    ic = perfil.perfil_rl_multiestrato([x1, x2], n, E, T, u, [a1, a2])
    assert ic.causa_lo is None and ic.causa_hi is None
    l_hat = perfil.loglik_estrato(x1, n[0], E[0], a1.xi, a1.sigma, n[0] / E[0]) + perfil.loglik_estrato(
        x2, n[1], E[1], a2.xi, a2.sigma, n[1] / E[1]
    )
    q = stats.chi2.ppf(0.95, 1)
    for lim in (ic.lo, ic.hi):
        d = 2 * (l_hat - _perfil_multi_fuerza_bruta([x1, x2], n, E, T, lim - u, [a1, a2]))
        assert abs(d - q) < 1e-3  # en el límite del código el déficit independiente vale χ²₁
        # el déficit cambia más de 0.02 al mover el límite un 1e-3 relativo: la tolerancia equivale a < 1e-3 en el RL
        d2 = 2 * (
            l_hat
            - _perfil_multi_fuerza_bruta([x1, x2], n, E, T, lim * (1.001 if lim > ic.punto else 0.999) - u, [a1, a2])
        )
        assert d2 - d > 0.02


def test_perfil_rl_limite_superior_abierto_por_tope():
    x = _simular(0.6, 1.0, 12, 4)
    a = gpd.ajustar_gpd(x)
    ic = perfil.perfil_rl(x, x.size, 40.0, 100.0, 1.0, a)
    assert np.isnan(ic.hi) and ic.causa_hi == "tope"
