"""F7: barrido y regla del IET (contrato §3.3; §8 «Regla del IET»)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from peruvian_wave_analysis.eventos import identificacion as ID
from peruvian_wave_analysis.eventos import iet as IE
from peruvian_wave_analysis.eventos import ruido_iet as RI
from peruvian_wave_analysis.extremos import gpd
from peruvian_wave_analysis.extremos import perfil as PF


def test_rejilla_y_funciones_de_la_regla_son_las_de_la_calibracion():
    assert IE.IETS_H == (42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 84.0, 96.0, 120.0)
    assert IE.Q_REFERENCIA == (0.98, 0.99, 0.995)
    assert IE.RI.cambios is RI.cambios and IE.RI.regla_iet is RI.regla_iet


def test_perfil_de_sigma_mod_con_u_cero_es_el_de_sigma():
    x = stats.genpareto.rvs(0.1, scale=0.4, size=300, random_state=np.random.default_rng(1))
    a = gpd.ajustar_gpd(x)
    ps = IE.perfil_sigma_mod(x, 0.0 + 1e-12, a)
    ref = PF.perfil_sigma(x, a)
    assert ps.punto == pytest.approx(a.sigma, rel=1e-9)
    assert ps.lo == pytest.approx(ref.lo, rel=2e-3) and ps.hi == pytest.approx(ref.hi, rel=2e-3)


def test_perfil_de_sigma_mod_limites_en_chi2_y_cubre_el_punto():
    rng = np.random.default_rng(7)
    u = 3.0
    x = stats.genpareto.rvs(-0.1, scale=0.5, size=200, random_state=rng)
    a = gpd.ajustar_gpd(x)
    ps = IE.perfil_sigma_mod(x, u, a)
    assert ps.lo < ps.punto < ps.hi
    q = stats.chi2.ppf(0.95, 1)
    for lim in (ps.lo, ps.hi):
        d = 2 * (a.loglik - IE._loglik_sigma_mod(x, u, lim))
        assert d == pytest.approx(q, abs=1e-3)
    assert 2 * (a.loglik - IE._loglik_sigma_mod(x, u, ps.punto)) == pytest.approx(0.0, abs=1e-6)


def test_separados_f6_a_mano():
    """Caso de la F6 = un evento suyo con dos rachas sobre u_tent; separado si la F7 las pone en eventos distintos."""
    idx = pd.date_range("2000-01-01", periods=200, freq="3h")
    hs = np.full(idx.size, 1.5)
    # caso A: dos crestas sobre u_tent = 3.15 a 60 h, con bajada a 2.75 (≥ 0.5 m bajo cada una) → la F7 lo parte
    hs[20:41] = 2.9
    hs[[20, 40]] = [3.5, 3.4]
    hs[30] = 2.75
    # caso B: dos crestas a 30 h → la F7 no lo parte
    hs[100:111] = 2.9
    hs[[100, 110]] = [3.5, 3.4]
    hs[105] = 2.75
    serie = pd.Series(hs, index=idx)
    ev = ID.identificar(serie, 2.7, 48.0)
    rev = pd.DataFrame({"clasificacion_visual": ["fusionado", "dudoso", "fusionado"],
                        "t_pico": [idx[20], idx[100], idx[20]],  # el tercero repite el primero (tipo «mayor»)
                        "t_pico_siguiente": [None, None, None]})  # fmt: skip
    s = IE.separados_f6(ev, rev, hs, 3.15)
    assert s["fusionado"] == {"casos": 1, "separados": 1, "pico_fuera_de_eventos": 0}
    assert s["dudoso"] == {"casos": 1, "separados": 0, "pico_fuera_de_eventos": 0}


def test_cuantil_del_sitio_en_los_procesos():
    filas = [{"proceso": "ar1_central", "iet_h": iet, "maximo_cambio": m}
             for iet in IE.IETS_H for m in (0.0, 0.005, 0.01, 0.02)]  # fmt: skip
    sitio = np.full(len(IE.IETS_H), 0.01)
    c = IE.cuantil_en_procesos(sitio, pd.DataFrame(filas))
    assert c["ar1_central"]["48"] == pytest.approx(0.75)


def test_barrido_y_regla_en_una_serie_sintetica():
    rng = np.random.default_rng(11)
    idx = pd.date_range("1990-01-01", "2025-12-31 21:00", freq="3h")
    hs = pd.Series(np.round(rng.gamma(3.0, 0.45, idx.size), 2), index=idx)
    u_id = float(np.quantile(hs, 0.95))
    rev = pd.DataFrame(columns=["clasificacion_visual", "t_pico", "t_pico_siguiente"])
    tabla, rl, us, anios = IE.barrido_iet(hs, u_id, rev, 3.15, iets=(42.0, 48.0, 54.0))
    assert rl.shape == (3, 3) and np.isfinite(rl).all()
    assert anios == pytest.approx(36.0, abs=1e-6)
    ev = tabla.groupby("iet_h")["eventos"].first().to_numpy()
    assert (np.diff(ev) <= 0).all()  # más IET, menos eventos
    assert (us[0] < us[1] < us[2]) and set(tabla["cuantil_referencia"]) == set(IE.Q_REFERENCIA)
    ok = tabla["valido"]
    assert (tabla.loc[ok, "xi_lo"] < tabla.loc[ok, "xi"]).all() and (
        tabla.loc[ok, "rl100_lo"] < tabla.loc[ok, "rl100"]
    ).all()
    m = RI.cambios(rl)
    iet, sensible = RI.regla_iet(m, (42.0, 48.0, 54.0), 0.01)
    assert iet in (48.0, 54.0)


def test_limite_privado_de_extremos_firma_y_resultado():
    """`iet.perfil_sigma_mod` usa `perfil._limite`, privada de `extremos/` (F07-R, revisión de código): si cambia su
    firma o su resultado, falla aquí y no al correr el barrido."""
    assert list(inspect.signature(PF._limite).parameters) == ["deficit", "punto", "q", "tope", "n_pasos"]
    q = 3.841458820694124  # χ²₁ del 95 %
    hi, c_hi = PF._limite(lambda t: (t - 2.0) ** 2, 2.0, q, 10.0)  # raíz en 2 + √q, barriendo hacia arriba
    lo, c_lo = PF._limite(lambda t: (t - 2.0) ** 2, 2.0, q, -10.0)  # hacia abajo
    assert c_hi is None and c_lo is None
    assert hi == pytest.approx(2.0 + np.sqrt(q), abs=1e-8) and lo == pytest.approx(2.0 - np.sqrt(q), abs=1e-8)
    assert PF._limite(lambda t: (t - 2.0) ** 2, 2.0, q, 3.0) == (pytest.approx(np.nan, nan_ok=True), PF.TOPE)
    v, c = PF._limite(lambda t: (t - 2.0) ** 2 if t < 3.0 else np.nan, 2.0, q, 10.0)
    assert np.isnan(v) and c == PF.BORDE
