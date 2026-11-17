"""Motor de prueba y potencia de la F5 (contrato v6 §3.4, §3.5, §4, §5 y §8), con lados sintéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.validacion import config as C
from peruvian_wave_analysis.validacion import potencia as P
from peruvian_wave_analysis.validacion import simulacion as S


def _lado(nombre="A", n=1500, inicio="2012-01-01", mision="HY-2A", tramo="HY-2A_2", seed=0, boya=None):
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(inicio) + pd.to_timedelta(np.sort(rng.uniform(0, 4 * 365, n)), unit="D")
    m = np.exp(0.6 + 0.3 * np.sin(2 * np.pi * t.day_of_year.to_numpy() / 365) + rng.normal(0, 0.25, n))
    df = pd.DataFrame({"tiempo": t, "hs_mod": m, "unidad": [f"u{i // 5}" for i in range(n)]})
    if boya:
        df = df.assign(boya=boya, k=3)
    else:
        df = df.assign(mision=mision, tramo_cal=tramo, n=8)
    return S.LadoSim.desde(nombre, df)


def test_simex_descuenta_el_ruido_y_no_cambia_sin_ruido():
    """§8: el ruido infla los cuantiles altos y SIMEX los devuelve a menos de 1 punto (media de 10 muestras)."""
    L = _lado(n=4000)
    o = L.m.copy()
    verdad = S.estadisticos_filas(L, o[None, :])[0]
    assert S.simex(L, o, np.zeros(L.n), np.random.default_rng(0), R=20) == pytest.approx(verdad)
    sig = np.full(L.n, 0.25)
    bruto, corr = [], []
    for k in range(10):
        rng = np.random.default_rng(k)
        x = o + sig * rng.standard_normal(L.n)
        bruto.append(S.estadisticos_filas(L, x[None, :])[0] / verdad - 1)
        corr.append(S.simex(L, x, sig, rng, R=50) / verdad - 1)
    bruto, corr = np.mean(bruto, axis=0), np.mean(corr, axis=0)
    for e in ("P90", "max"):
        assert bruto[S.IDX[e]] > 0.015
        assert abs(corr[S.IDX[e]]) < 0.01


def test_generador_conserva_la_marginal_y_alcanza_el_si():
    L = _lado(n=3000)
    r = S.calibrar_r(L, 0.16, 0.8, semilla=3)
    o = S.verdadero(L, r, 0.8, np.random.default_rng(4))
    assert S.si(L, o) == pytest.approx(0.16, abs=0.02)
    q = [0.5, 0.9, 0.99]
    assert np.quantile(o, q) == pytest.approx(np.quantile(L.m, q), rel=0.05)


def test_nulidad_del_generador():
    """§8: b = 0, SI = 0.16, ρ = 0.8: B(P90) y B(P99) descontados con media dentro del error de Monte Carlo de 0."""
    L = _lado(n=2500)
    r = S.calibrar_r(L, 0.16, 0.8, semilla=5)
    b90, b99 = [], []
    for i in range(40):
        rng = np.random.default_rng(100 + i)
        o = S.verdadero(L, r, 0.8, rng)
        sig = S.sigma_ruido(L, o, 1.0)
        med = S.medir(L, o + sig * rng.standard_normal(L.n), sig, 49, rng)
        b90.append(med.B[S.IDX["P90"]])
        b99.append(med.B[S.IDX["P99"]])
    for x in (np.array(b90), np.array(b99)):
        assert abs(x.mean()) <= 3 * x.std(ddof=1) / np.sqrt(x.size)


def test_piso_de_calibracion_no_baja_con_n():
    for n in (500, 5000):
        L = _lado(n=n, mision="GEOSAT", tramo="GEOSAT", inicio="1985-06-01")
        err = S.sortear_errores(S._claves(L, False), 1.0, np.random.default_rng(0), size=20000)
        Q = np.full(5, 3.0)
        sd = S.termino_calibracion(L, Q, err, False)[:, S.IDX["P99"]].std()
        sa, sb = C.CALIBRACION["GEOSAT"]
        esperado = np.sqrt(sa**2 + (sb / 3.0) ** 2 + C.SIGMA_ESTADO_MAR**2 + C.SIGMA_RED_EPOCA**2)
        assert sd == pytest.approx(esperado, rel=0.03)


def test_epoca_se_cancela_en_v2_y_no_en_v1():
    a1 = _lado("A1", inicio="2012-01-01", tramo="HY-2A_2", seed=1)
    a2 = _lado("A2", inicio="2016-06-01", tramo="HY-2A_2", seed=2)
    t0 = _lado("T0", inicio="1985-06-01", mision="GEOSAT", tramo="GEOSAT", seed=3)
    claves = set(S._claves(a1, False)) | set(S._claves(a2, False)) | set(S._claves(t0, False))
    err = S.sortear_errores(claves, 1.0, np.random.default_rng(0), size=20000)
    Q = np.full(5, 3.0)
    d2 = S.termino_calibracion(a2, Q, err, False) - S.termino_calibracion(a1, Q, err, False)
    d1 = S.termino_calibracion(a2, Q, err, False) - S.termino_calibracion(t0, Q, err, False)
    assert d2.std() < 1e-12  # mismo tramo y misma época: se cancela entero en V2
    assert d1.std() > 0.03  # T0 frente a A: no se cancela


def test_termino_por_lado_de_la_boya_solo_en_sensibilidad():
    b = _lado("T1b", boya="32302", inicio="1993-03-01")
    assert "lado_T1b" not in S._claves(b, False)
    assert S._claves(b, True)["lado_T1b"] == 1.0
    assert sum(v for k, v in S._claves(b, False).items() if k.startswith("32302_")) == pytest.approx(1.0)


def test_max_t_error_de_tipo_I_y_potencia_por_desplazamiento():
    rng = np.random.default_rng(7)
    rech, rech_b = [], []
    for _ in range(600):
        ee = np.array([0.02, 0.03, 0.05])
        est = rng.normal(0, ee)
        reps = est + rng.normal(0, ee, (999, 3))
        pr = S.max_t(est, reps, np.array([True, True, True]))
        rech.append(S.rechaza(pr, 0.0, 0.05))
        rech_b.append(S.rechaza(pr, 0.08, 0.05))
    assert 0.03 <= np.mean(rech) <= 0.07
    assert np.mean(rech_b) > 0.9


def test_no_estimable_no_entra_y_no_falla():
    pr = S.max_t(np.array([0.01, 0.5]), np.random.default_rng(0).normal(0, 0.01, (99, 2)), np.array([True, False]))
    assert np.isnan(pr.est[1]) and np.isfinite(pr.est[0])
    nada = S.max_t(np.array([0.1]), np.zeros((9, 1)), np.array([False]))
    assert not S.rechaza(nada, 1.0, 0.05)


def test_estimabilidad_exige_diez_unidades_sobre_el_cuantil():
    L = _lado(n=300)
    o = L.m.copy()
    Q = S.estadisticos_filas(L, o[None, :])[0]
    ok = S.estimables(L, o, Q)
    assert ok[S.IDX["P50"]] and not ok[S.IDX["P99.9"]]  # 300 colocaciones en unidades de 5: P99.9 deja < 10


def test_regla_de_variantes_con_sesgo_no_concluyente():
    nula = S.Prueba(np.array([0.0]), np.array([1.0]), {0.025: 9.0, 0.05: 9.0}, 1.0)
    reps = [
        {"V3a": nula, "V3b": nula, "subida": {"B_cola": -0.02, "B_alta": -0.05}},
        {"V3a": nula, "V3b": nula, "subida": {"B_cola": 0.01, "B_alta": 0.03}},
    ]
    s = P.subida_esperada(reps, 0.2, 0.2)
    assert s["subida_media"] == pytest.approx((np.exp(0.05) + 1.0) / 2 - 1)


def test_recalibracion_del_critico_lleva_la_tasa_a_alfa():
    rng = np.random.default_rng(3)
    reps = []
    for _ in range(400):
        est = rng.normal(0, 1.0, 1)
        reps.append({"X": S.Prueba(est, np.array([0.7]), {0.05: 1.96}, np.nan)})  # EE subestimado: tasa alta
    rc = P.recalibrar(reps, "X", 0.05)
    assert rc["recalibrado"] and rc["tasa"] > 0.07
    assert np.mean([S.rechaza(rp["X"], 0.0, 0.05, rc["factor"]) for rp in reps]) == pytest.approx(0.05, abs=0.01)


def test_rejilla_tiene_el_nominal():
    nombres = [p.nombre for p in P.rejilla()]
    assert P.NOMINAL in nombres and len(nombres) == 10
