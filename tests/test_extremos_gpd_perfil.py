"""Ajuste rápido de la GPD por perfil frente a ``ajustar_gpd`` (ADR-0034): mismo estimador en los casos difíciles."""

from __future__ import annotations

import numpy as np
import pytest

from peruvian_wave_analysis.extremos import gpd, gpd_perfil

TOL_XI = 1e-5
TOL_LL = 1e-7


def _muestra(rng, n, xi, sigma=0.35, empates=True):
    x = gpd.cuantil_supervivencia(rng.random(n), xi, sigma)
    if empates:  # picos a 0.01 m sobre un umbral que es punto medio, como en los datos
        x = np.round(2.7 + x, 2) - 2.705
        x = np.where(x <= 0, 0.005, x)
    return x


def comparar(x) -> tuple[str, float, float]:
    """(«igual», «delega», «distinto»), |Δξ| y Δℓ (rápido − F2)."""
    a, b = gpd.ajustar_gpd(x), gpd_perfil.ajustar_rapido(x)
    if b.arranque != "perfil":
        assert b == a  # delegó: idéntico por construcción
        return "delega", 0.0, 0.0
    if a.valido != b.valido or a.frontera != b.frontera:
        return "distinto", np.inf, b.loglik - a.loglik
    dxi, dll = abs(a.xi - b.xi), b.loglik - a.loglik
    return ("igual" if dxi < TOL_XI and abs(dll) < TOL_LL else "distinto"), dxi, dll


@pytest.mark.parametrize("n", [30, 60, 250])
@pytest.mark.parametrize("xi", [-0.45, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
def test_mismo_estimador_en_casos_dificiles(n, xi):
    rng = np.random.default_rng(1000 + n + int(100 * xi))
    resultados = [comparar(_muestra(rng, n, xi)) for _ in range(40)]
    assert not [r for r in resultados if r[0] == "distinto"]  # 0 en 4 200 muestras al escribir el ADR


def test_sin_empates_y_casi_constantes():
    rng = np.random.default_rng(5)
    for _ in range(30):
        assert comparar(_muestra(rng, 40, 0.0, empates=False))[0] in ("igual", "delega")
    casi = 0.3 + np.array([0.0, 0.01] * 15) + 1e-4 * np.arange(30)  # excesos casi constantes
    assert comparar(casi)[0] in ("igual", "delega")
    assert comparar(np.full(30, 0.3))[0] in ("igual", "delega")


def test_delega_en_frontera_y_n_insuficiente():
    x = np.linspace(0.01, 1.0, 30)  # uniforme: ξ̂ en la frontera −1
    b = gpd_perfil.ajustar_rapido(x)
    assert b == gpd.ajustar_gpd(x) and b.arranque != "perfil"
    y = np.array([0.1, 0.2, 0.3])
    assert gpd_perfil.ajustar_rapido(y) == gpd.ajustar_gpd(y)
    assert not gpd_perfil.ajustar_rapido(y).valido


def test_fraccion_que_responde_el_perfil():
    """El atajo sirve si responde en la gran mayoría de las muestras típicas del bootstrap."""
    rng = np.random.default_rng(9)
    tipos = [comparar(_muestra(rng, int(rng.integers(30, 300)), float(rng.choice([-0.1, 0.0, 0.1]))))[0]
             for _ in range(200)]  # fmt: skip
    assert tipos.count("igual") >= 190 and tipos.count("distinto") == 0
