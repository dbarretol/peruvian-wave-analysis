"""Pruebas de la GEV de máximos por bloque (contrato de la F6, §3.9 y §8)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from peruvian_wave_analysis.comun import semillas
from peruvian_wave_analysis.extremos import gev as G


def _muestra(xi, n, rng, mu=5.0, sigma=0.6):
    return stats.genextreme.rvs(-xi, loc=mu, scale=sigma, size=n, random_state=rng)  # scipy: c = −ξ


def test_signo_de_xi_frente_a_scipy():
    x = np.linspace(4.0, 8.0, 7)
    for xi in (-0.2, 0.0, 0.2):
        assert G.loglik(x, 5.0, 0.6, xi) == pytest.approx(stats.genextreme.logpdf(x, -xi, 5.0, 0.6).sum())
        for p in (0.5, 0.9, 0.99):
            assert G.cuantil(p, 5.0, 0.6, xi) == pytest.approx(stats.genextreme.ppf(p, -xi, 5.0, 0.6))


def test_niveles_de_retorno_analiticos():
    aj = G.AjusteGEV(5.0, 0.6, 0.1, 0.0, 46, True, True, None)
    for T in (50.0, 100.0):
        p = np.exp(-1 / T)
        z = 5.0 - 0.6 / 0.1 * (1 - (-np.log(p)) ** -0.1)
        assert aj.rl(T) == pytest.approx(z, rel=1e-12)
        zc = 5.0 - 0.6 / 0.1 * (1 - (-np.log(1 - 1 / T)) ** -0.1)
        assert aj.rl(T, "coles") == pytest.approx(zc, rel=1e-12)
    assert aj.rl(100.0) > aj.rl(100.0, "coles")  # exp(−1/T) > 1 − 1/T


def test_fuera_del_soporte_y_validacion():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isfinite(G.loglik(x, 2.0, 1.0, -0.5))  # extremo superior μ + σ/0.5 = 4 > 3: dentro
    assert G.loglik(x, 2.0, 1.0, -1.0) == -np.inf  # extremo superior 3: el máximo toca el borde
    assert np.isfinite(G.loglik(x, 2.0, 1.0, 0.5))  # extremo inferior μ − σ/0.5 = 0 < 1: dentro…
    assert G.loglik(np.r_[x, -0.5], 2.0, 1.0, 0.5) == -np.inf  # …y −0.5 queda fuera
    assert not G.ajustar_gev(np.arange(5.0)).valido  # menos de 10 máximos
    # |ξ̂| > 0.5 se marca fuera de soporte
    aj = G.ajustar_gev(_muestra(0.9, 200, np.random.default_rng(3)))
    assert aj.valido and not aj.soporte_ok and aj.causa


@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.2])
@pytest.mark.parametrize("n", [46, 32])
def test_recupera_parametros(xi, n):
    """Mediana de ξ̂, μ̂ y σ̂ en 100 réplicas cerca de la verdad (sesgo de muestra chica tolerado)."""
    est = np.array(
        [
            [a.mu, a.sigma, a.xi]
            for a in (
                G.ajustar_gev(_muestra(xi, n, semillas.generador("prueba", "gev", str(xi), str(n), str(i))))
                for i in range(100)
            )
        ]
    )
    med = np.median(est, axis=0)
    assert med[0] == pytest.approx(5.0, abs=0.08)
    assert med[1] == pytest.approx(0.6, abs=0.08)
    assert med[2] == pytest.approx(xi, abs=0.08)
    # el ajuste propio no es peor que el de scipy
    x = _muestra(xi, n, np.random.default_rng(0))
    c, loc, sc = stats.genextreme.fit(x)
    assert G.ajustar_gev(x).loglik >= stats.genextreme.logpdf(x, c, loc, sc).sum() - 1e-6


def test_perfil_contiene_el_punto_y_es_finito():
    x = _muestra(0.0, 46, np.random.default_rng(5))
    aj = G.ajustar_gev(x)
    for iv in (G.perfil_xi(x, aj), G.perfil_rl(x, aj, 100.0)):
        assert not iv.abierto and iv.lo < iv.punto < iv.hi
    # en los límites, el déficit es χ²₁(0.95)
    iv = G.perfil_xi(x, aj)
    assert 2 * (aj.loglik - G.loglik_perfil_xi(x, iv.hi, aj)) == pytest.approx(stats.chi2.ppf(0.95, 1), abs=1e-3)


@pytest.mark.lento
@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.2])
def test_cobertura_del_perfil(xi):
    """§8: cobertura del IC de perfil en 500 réplicas (n = 46) dentro de ±3 puntos del 95 %, para ξ y el RL100."""
    T = 100.0
    verdad_rl = float(G.cuantil(np.exp(-1 / T), 5.0, 0.6, xi))
    cub_xi, cub_rl, n = 0, 0, 0
    for i in range(500):
        x = _muestra(xi, 46, semillas.generador("prueba", "gev-cobertura", str(xi), str(i)))
        aj = G.ajustar_gev(x)
        px, pr = G.perfil_xi(x, aj), G.perfil_rl(x, aj, T)
        lo_x, hi_x = (-np.inf if np.isnan(px.lo) else px.lo), (np.inf if np.isnan(px.hi) else px.hi)
        lo_r, hi_r = (-np.inf if np.isnan(pr.lo) else pr.lo), (np.inf if np.isnan(pr.hi) else pr.hi)
        cub_xi += lo_x <= xi <= hi_x
        cub_rl += lo_r <= verdad_rl <= hi_r
        n += 1
    assert abs(cub_xi / n - 0.95) <= 0.03, cub_xi / n
    assert abs(cub_rl / n - 0.95) <= 0.03, cub_rl / n
