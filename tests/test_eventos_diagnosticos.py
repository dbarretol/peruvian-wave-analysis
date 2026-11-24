"""F7: diagnósticos de §3.5 (contrato §8: «Prueba de Poisson sobre rachas», «Prueba de Poisson con intensidad
armónica», «Correlación entre picos», «Süveges y Davison», «Picos de verano»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from peruvian_wave_analysis.eventos import diagnosticos as D
from peruvian_wave_analysis.extremos import reescalado as RS

_rejilla = D.rejilla_sintetica
_proceso = D.proceso_sintetico


def test_proceso_sintetico_concentrado_en_invierno():
    t = _rejilla(20)
    ini, _, marc = _proceso(t, np.random.default_rng(0))
    invierno = np.isin(t[ini].month, D.MESES_MAYO_OCTUBRE)
    assert 0.86 <= invierno.mean() <= 0.94
    assert marc[invierno].mean() > 1.5 * marc[~invierno].mean()


def test_paso_del_anio_y_diseno():
    t = pd.DatetimeIndex(["2001-01-01 00:00", "2001-01-01 21:00", "2001-12-31 21:00", "2004-12-31 21:00"])
    assert D.paso_del_anio(t).tolist() == [0, 7, 364 * 8 + 7, 365 * 8 + 7]
    X = D.diseno_armonico()
    assert X.shape == (2928, 5) and np.allclose(X[0], [1, 0, 1, 0, 1])


def test_regresion_armonica_recupera_los_coeficientes():
    X = D.diseno_armonico()
    beta = np.array([-6.0, 0.4, 1.1, -0.3, 0.2])
    E = np.full(X.shape[0], 3.0 * 46)
    y = np.random.default_rng(2).poisson(E * np.exp(X @ beta))
    b = D._glm_poisson(y.astype(float), E, X)
    assert b == pytest.approx(beta, abs=0.12)
    # el estimador cumple las ecuaciones de score: Σ y = Σ μ y Xᵀ(y − μ) = 0
    assert np.abs(X.T @ (y - E * np.exp(X @ b))).max() < 1e-6 * y.sum()


def test_intensidad_por_estaciones_es_cuentas_sobre_horas():
    t = _rejilla(2)
    mod = D.modelo_estaciones(t, D.ESTACIONES_2)
    riesgo = np.ones(t.size, bool)
    lleg = np.flatnonzero(t.month == 7)[:5]
    lam, par = D.ajustar_intensidad(mod, riesgo, lleg)
    horas_mo = D.PASO_H * np.isin(t.month, D.MESES_MAYO_OCTUBRE).sum()
    assert par["may-oct"] == pytest.approx(5 / horas_mo) and par["nov-abr"] == 0.0
    assert lam[lleg[0]] == pytest.approx(5 / horas_mo)
    with pytest.raises(ValueError):
        D.modelo_estaciones(t, {"may-oct": D.MESES_MAYO_OCTUBRE})


def test_reescalado_observado_coincide_con_la_biblioteca():
    """Con intensidad constante (una estación) y todas las rachas como llegadas, A² es el de ``prueba_poisson``."""
    t = _rejilla(8)
    rng = np.random.default_rng(4)
    ini, fin, _ = _proceso(t, rng)
    validos = np.ones(t.size, bool)
    mod = D.modelo_estaciones(t, {"todo": tuple(range(1, 13))})
    r = D.prueba_poisson_rachas(t, validos, ini, fin, np.ones(ini.size, bool), 48.0, mod, rng, B=19)
    ref = RS.intervalos_reescalados(t, validos, t[ini], t[fin], np.zeros(t.size), 48.0, D.PASO_H)
    assert r["ad"] == pytest.approx(RS.estadistico_ad(ref.z)) and r["n_intervalos"] == ref.z.size
    lam, _ = D.ajustar_intensidad(mod, ref.en_riesgo, ini)
    assert D._z(ref.en_riesgo, lam, ini, D.PASO_H) == pytest.approx(ref.z)


def test_prueba_poisson_reproducible_y_marcadas_subconjunto():
    t = _rejilla(8)
    ini, fin, marc = _proceso(t, np.random.default_rng(5))
    mod = D.modelo_armonico(t)
    a = D.prueba_poisson_rachas(t, np.ones(t.size, bool), ini, fin, marc, 48.0, mod, np.random.default_rng(1), B=19)
    b = D.prueba_poisson_rachas(t, np.ones(t.size, bool), ini, fin, marc, 48.0, mod, np.random.default_rng(1), B=19)
    assert a["p_ad"] == b["p_ad"] and a["n_llegadas"] == marc.sum() and a["n_rachas"] == ini.size
    assert a["n_intervalos"] == marc.sum() - 1


@pytest.mark.lento
@pytest.mark.parametrize("modelo", ["armonico", "2_estaciones"])
def test_tamano_de_la_prueba_completa(modelo):
    """§8: rachas de Poisson concentradas en invierno, fracción > u* estacional (el doble en mayo–octubre); dos
    regresiones, adelgazamiento estacional y tiempo muerto de todas las rachas. Armónico: ≈ 5 % (±3 puntos).
    Con 2 estaciones fijas el tamaño se mide y se publica (la etapa lo rotula «frágil» si pasa del 8 %).
    B = 99 por réplica para acotar el tiempo (la calibración no depende de B)."""
    t = _rejilla(10)
    validos = np.ones(t.size, bool)
    mod = D.modelo_armonico(t) if modelo == "armonico" else D.modelo_estaciones(t, D.ESTACIONES_2)
    rng = np.random.default_rng(11)
    rech = 0
    for _ in range(500):
        ini, fin, marc = _proceso(t, rng)
        rech += D.prueba_poisson_rachas(t, validos, ini, fin, marc, 48.0, mod, rng, B=99)["rechaza"]
    tam = rech / 500
    if modelo == "armonico":
        assert abs(tam - 0.05) <= 0.03, tam
    else:
        assert 0.0 <= tam <= 0.30, tam


def test_dispersion_a_mano_y_sobredispersion():
    expo = pd.Series([1.0, 1.0, 0.5], index=[2001, 2002, 2003])
    anios = [2001] * 4 + [2002] * 1 + [2003] * 1  # λ = 6 / 2.5 = 2.4; μ = 2.4, 2.4, 1.2
    r = D.dispersion_anual(anios, expo, np.random.default_rng(0), B=999)
    esperado = ((4 - 2.4) ** 2 / 2.4 + (1 - 2.4) ** 2 / 2.4 + (1 - 1.2) ** 2 / 1.2) / 2
    assert r["indice"] == pytest.approx(esperado) and r["lambda_por_anio"] == pytest.approx(2.4)
    rng = np.random.default_rng(3)
    expo = pd.Series(1.0, index=range(1980, 2026))
    sobre = np.repeat(expo.index, rng.negative_binomial(2, 0.2, expo.size))
    assert D.dispersion_anual(sobre, expo, rng, B=999)["p"] < 0.01
    with pytest.raises(ValueError):
        D.dispersion_anual([1970], expo, rng, B=9)


def test_tau_a_mano_y_subconjuntos():
    t = pd.to_datetime(["2001-06-01", "2001-06-03", "2001-08-01", "2002-06-01", "2002-06-20", "2002-07-30"])
    x = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.4])
    racha = np.array([0, 0, 1, 2, 3, 4])
    r = D.tau_consecutivos(t, x, racha, np.random.default_rng(0), B=99)
    assert r["todos"]["pares"] == 5 and r["todos"]["tau"] == pytest.approx(stats.kendalltau(x[:-1], x[1:]).statistic)
    assert r["menos_de_7_dias"]["pares"] == 1 and r["menos_de_7_dias"]["tau"] is None
    assert r["misma_racha"]["pares"] == 1 and r["misma_racha"]["p"] is None
    assert 0 < r["todos"]["p"] <= 1


@pytest.mark.lento
def test_tamano_tau_con_picos_independientes():
    rng = np.random.default_rng(8)
    t = pd.date_range("1990-01-01", periods=60, freq="9D")
    rech = sum(
        D.tau_consecutivos(t, rng.exponential(size=60), np.arange(60), rng, B=199)["todos"]["rechaza"]
        for _ in range(500)
    )
    assert abs(rech / 500 - 0.05) <= 0.03


def test_separaciones():
    t = pd.to_datetime(["2001-01-01 00:00", "2001-01-03 00:00", "2001-01-04 12:00", "2001-02-01 00:00"])
    r = D.separaciones(t, 48.0)
    assert r["horas"].tolist() == [48.0, 36.0, 660.0] and r["en_franja"] == 1
    assert r["fraccion_en_franja"] == pytest.approx(1 / 3)


def test_a2_gpd_con_xi_cero_es_el_de_la_exponencial():
    e = np.random.default_rng(1).exponential(size=40)
    assert D.a2_gpd(e, 0.0, 1.0) == pytest.approx(RS.estadistico_ad(e))


def test_bondad_a2_con_gpd_exacta():
    rng = np.random.default_rng(2)
    picos = np.round(3.0 + stats.genpareto.rvs(-0.1, scale=0.4, size=60, random_state=rng), 2)
    r = D.bondad_a2(picos, 3.0 - 0.005, rng, B=49)
    assert r["n"] == 60 and r["a2"] > 0 and 0 < r["p"] <= 1 and r["generadas"] >= 49


def test_celdas_sd_once_sin_leer_datos():
    c = D.celdas_sd((48, 54, 60, 66, 72, 84, 96, 120))
    assert len(c) == 11
    assert {(x["q"], x["iet_h"]) for x in c if x["q"] == 0.99} == {(0.99, 48.0), (0.99, 54.0), (0.99, 60.0)}
    assert all(x["KF"] <= 0.2 + 1e-9 for x in c) and c[0]["K"] == 16


def test_kgaps_cumple_el_score_y_bloques():
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.integers(1, 400, 300))
    r = D.kgaps(pos, 3, 0.01)
    c = 0.01 * np.maximum(np.diff(pos) - 3, 0)
    th = r["theta"]
    score = -np.sum(c == 0) / (1 - th) + 2 * np.sum(c > 0) / th - c.sum()
    assert abs(score) < 1e-6 * len(c)
    rb = D.kgaps(pos, 3, 0.01, bloques=np.arange(pos.size) // 10)
    assert rb["N_huecos"] == pos.size - 1 - (pos.size - 1) // 10


def _max_ar(n, phi, rng):
    """X_t = max(φ X_{t−1}, (1 − φ) Z_t), Z Fréchet(1): índice extremal θ = 1 − φ."""
    z = 1.0 / -np.log(rng.random(n))
    x = np.empty(n)
    x[0] = z[0]
    for i in range(1, n):
        x[i] = max(phi * x[i - 1], (1 - phi) * z[i])
    return x


def test_kgaps_p_chi2_y_p_calibrado():
    rng = np.random.default_rng(6)
    x = _max_ar(20000, 0.5, rng)
    j = np.flatnonzero(x > np.quantile(x, 0.99))
    r = D.kgaps(j, 1, 0.01, rng=np.random.default_rng(1), B=99)
    assert r["p_chi2"] == pytest.approx(stats.chi2.sf(r["T"], 1))
    assert r["p"] in {(1 + k) / 100 for k in range(100)} and r["B"] == 99
    assert D.kgaps(j, 1, 0.01)["p"] is None
    assert r == D.kgaps(j, 1, 0.01, rng=np.random.default_rng(1), B=99)


def test_kgaps_recupera_theta_en_un_max_ar():
    rng = np.random.default_rng(3)
    x = _max_ar(60000, 0.5, rng)
    u = np.quantile(x, 0.99)
    r = D.kgaps(np.flatnonzero(x > u), 1, 0.01)
    assert r["theta"] == pytest.approx(0.5, abs=0.08)


@pytest.mark.lento
def test_tamano_de_la_matriz_de_informacion():
    rng = np.random.default_rng(4)
    rech = 0
    for _ in range(500):
        x = _max_ar(20000, 0.5, rng)
        u = np.quantile(x, 0.99)
        rech += D.kgaps(np.flatnonzero(x > u), 1, 0.01, rng=rng)["p"] <= 0.05
    assert abs(rech / 500 - 0.05) <= 0.03, rech / 500  # §8 con el p calibrado (precisión n.º 27)


def test_picos_de_verano():
    t = pd.to_datetime(["2001-01-10", "2001-07-01", "2001-11-30", "2001-04-30"])
    assert D.picos_de_verano(t, [3.0, 4.0, 3.5, 3.2], 3.1) == 2


def test_rachas_de_era_con_racha_que_cruza_el_inicio():
    ev = pd.DataFrame({"id_racha": [0, 1, 1, 2], "i_inicio": [2, 8, 12, 40], "i_fin": [4, 11, 14, 42]})
    ini, fin, ids, validos, cruzan = D.rachas_de_era(ev, 60, 10, 50, 5, np.ones(60, bool))
    assert ids.tolist() == [2] and ini.tolist() == [40] and fin.tolist() == [42] and cruzan == 1
    assert not validos[:20].any() and validos[20:51].all() and not validos[51:].any()
