"""Pruebas de la F6, §8: descriptivos (dirección, climatología, Tp censurado, peralte, bimodalidad circular)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from peruvian_wave_analysis.clima import descriptivos as DS
from peruvian_wave_analysis.clima import estratificacion as E
from peruvian_wave_analysis.comun import semillas
from peruvian_wave_analysis.extremos.gpd import sigma_dado_xi


def test_media_circular_y_rosa():
    m, disp = DS.media_circular([350.0, 10.0])
    assert m == pytest.approx(0.0, abs=1e-9) or m == pytest.approx(360.0, abs=1e-9)
    assert disp == pytest.approx(1 - np.cos(np.deg2rad(10)))
    assert DS.media_circular([90, 270])[1] == pytest.approx(1.0)
    assert np.isnan(DS.media_circular([])[0])
    assert list(DS.sector([0, 11.2, 11.3, 348.8, 348.7, 180])) == [0, 0, 1, 0, 15, 8]
    r = DS.rosa([0, 1, 90, np.nan])
    assert len(r) == 16 and r["n"].sum() == 3 and r.loc[4, "n"] == 1 and r.loc[0, "fraccion"] == pytest.approx(2 / 3)


def test_clima_mensual():
    t = pd.date_range("2000-01-01", "2001-12-31 21:00", freq="3h")
    df = pd.DataFrame({"VHM0": np.where(t.month == 7, 3.0, 1.0), "VTPK": 12.0, "VPED": 200.0}, index=t)
    c = DS.clima_mensual(df)
    assert len(c) == 12 and c.loc[c.mes == 7, "hs_max"].item() == 3.0 and c.loc[c.mes == 1, "hs_p99"].item() == 1.0
    assert c.loc[c.mes == 2, "anios"].item() == 2 and c.loc[0, "dir_media"] == pytest.approx(200.0)


def test_intervalos_de_la_cola_se_unen_hasta_el_minimo():
    hs = np.concatenate([np.full(50, 1.1), np.full(40, 1.3), [1.6, 1.7, 2.6, 3.9]])
    b = DS.intervalos_cola(hs, 0.25, 30)
    # 1.0–1.25 (50), 1.25–1.5 (40); la cola (1.5–máximo) junta 4 < 30 y absorbe 1.25–1.5 → 44
    assert b[0] == 1.0 and b[-1] == 3.9 and list(np.round(b, 6)) == [1.0, 1.25, 3.9]
    tc = DS.tp_condicional(hs, np.full(hs.size, 10.0), np.zeros(hs.size, bool), 0.25, 30)
    assert list(tc["n"]) == [50, 44] and tc["n"].sum() == hs.size


def test_cuantiles_con_censura_conocida():
    tp = np.r_[np.linspace(8, 15, 80), np.full(20, 25.0)]
    c = np.r_[np.zeros(80, bool), np.ones(20, bool)]  # 20 % censurado en 25 s
    q = DS.cuantiles_censurados(tp, c)
    assert q["fraccion_censurada"] == pytest.approx(0.2)
    assert q["tp_q95"] == "≥ 25" and q["tp_q75"] == pytest.approx(np.quantile(tp, 0.75))
    assert isinstance(q["tp_q50"], float)
    q = DS.cuantiles_censurados(tp[60:], c[60:])  # 50 % censurado: la mediana sigue siendo número, P75 no
    assert q["tp_q50"] != "≥ 25" and q["tp_q75"] == "≥ 25"


def test_peralte_y_limite_dnv():
    assert DS.limite_peralte_dnv([8.0, 11.5, 15.0]) == pytest.approx([1 / 15, (1 / 15 + 1 / 25) / 2, 1 / 25])
    assert DS.limite_peralte_dnv([5.0, 20.0]) == pytest.approx([1 / 15, 1 / 25])
    assert DS.peralte(2.0, 10.0) == pytest.approx(2 * np.pi * 2 / (9.80665 * 100))


def _mezcla_vm(n1, n2, mu1, mu2, kappa, nombre):
    rng = semillas.generador("prueba", "von-mises", nombre)
    a = stats.vonmises.rvs(kappa, loc=np.deg2rad(mu1), size=n1, random_state=rng)
    b = stats.vonmises.rvs(kappa, loc=np.deg2rad(mu2), size=n2, random_state=rng)
    return np.rad2deg(np.r_[a, b]) % 360


def test_bimodalidad_con_mezcla_conocida():
    g = _mezcla_vm(400, 100, 200.0, 300.0, 60.0, "bimodal")
    b = DS.bimodalidad(g)
    assert b.candidata and b.n_modas == 2 and not b.mas_de_dos_modas
    assert b.modas_grados[0] == pytest.approx(200, abs=4) and b.modas_grados[1] == pytest.approx(300, abs=6)
    assert b.masas[1] == pytest.approx(0.2, abs=0.04) and sum(b.masas) == pytest.approx(1.0, abs=1e-6)
    desde, hasta = b.cortes_grados
    assert 200 < desde < 300  # el mínimo entre las modas
    en_c = DS.en_arco(g, desde, hasta)
    assert en_c.mean() == pytest.approx(0.2, abs=0.04)


def test_bimodalidad_que_cruza_el_norte_y_caso_unimodal():
    g = _mezcla_vm(300, 200, 340.0, 60.0, 60.0, "norte")
    b = DS.bimodalidad(g)
    assert b.candidata and b.masas[1] == pytest.approx(0.4, abs=0.05)
    u = DS.bimodalidad(_mezcla_vm(500, 0, 210.0, 0.0, 30.0, "unimodal"))
    assert not u.candidata and u.n_modas == 1 and u.cortes_grados is None
    chica = DS.bimodalidad(_mezcla_vm(480, 20, 200.0, 320.0, 60.0, "chica"))  # 4 %: segunda moda < 10 %
    assert not chica.candidata


def test_anderson_darling_sin_aviso_de_variante():
    """§8: la variante se pasa explícita; falla si scipy emite el aviso de cambio de variante."""
    rng = semillas.generador("prueba", "ad-variante")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = E.anderson_darling(rng.normal(size=60), rng.normal(size=40))
    assert 0.001 <= p <= 0.25


def test_xi_comun_maximiza_la_suma_de_perfiles():
    """§8: el ajuste conjunto con ξ común maximiza ℓ₁(ξ, σ̂₁(ξ)) + ℓ₂(ξ, σ̂₂(ξ))."""
    a, b = E.muestra(-0.1, -0.1, 1.8, 200, 50, semillas.generador("prueba", "xi-comun-perfil"))
    co = E.ajustar_xi_comun(a, b)
    rej = np.linspace(-0.6, 0.5, 221)
    suma = [sigma_dado_xi(a, x)[1] + sigma_dado_xi(b, x)[1] for x in rej]
    assert co.loglik >= max(suma) - 1e-6
    assert co.xi == pytest.approx(rej[int(np.nanargmax(suma))], abs=0.01)


def test_rl100_combinado_con_estratos_identicos_y_caso_analitico():
    """§8: con dos estratos idénticos, igual al modelo sin estratos; con ξ = 0, caso analítico."""
    assert E.rl_mezcla([(2.0, 0.1, 0.4), (3.0, 0.1, 0.4)]) == pytest.approx(E.rl_mezcla([(5.0, 0.1, 0.4)]), rel=1e-10)
    # ξ = 0 en los dos: λ₁ e^{−x/σ₁} + λ₂ e^{−x/σ₂} = 1/T con σ₁ = σ₂ = s → x = s ln(λT)
    assert E.rl_mezcla([(1.0, 0.0, 0.5), (4.0, 0.0, 0.5)]) == pytest.approx(E.U_SIM + 0.5 * np.log(500), rel=1e-9)
