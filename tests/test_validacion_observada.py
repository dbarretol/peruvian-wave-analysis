"""Etapa 3 de la F5 (contrato v6, §3–§6, §8, §19 y §20), con datos sintéticos. Ninguna prueba abre valores reales."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis import manifiesto
from peruvian_wave_analysis.comun import etapa as ET
from peruvian_wave_analysis.comun import orquestador
from peruvian_wave_analysis.comun.errores import GateFallido
from peruvian_wave_analysis.validacion import config as CV
from peruvian_wave_analysis.validacion import potencia as P
from peruvian_wave_analysis.validacion import simulacion as S
from peruvian_wave_analysis.validacion.etapa_potencia import _fila_replica
from peruvian_wave_analysis.validacion_obs import complementos, lectura, notas, ruido
from peruvian_wave_analysis.validacion_obs import config as C
from peruvian_wave_analysis.validacion_obs import decision as D
from peruvian_wave_analysis.validacion_obs import etapa as EO
from peruvian_wave_analysis.validacion_obs import prueba as PR
from peruvian_wave_analysis.validacion_obs import simulacion_obs as SO

B_PRUEBA = 299


def _df_alt(n=1500, inicio="2012-01-01", b=0.0, b_nivel=0.0, si=0.10, seed=0, mision="HY-2A", tramo="HY-2A_2"):
    """Superobservaciones sintéticas: o = m·exp(−b − b_nivel·g) con un error por pasada y otro propio."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(inicio) + pd.to_timedelta(np.sort(rng.uniform(0, 4 * 365, n)), unit="D")
    m = np.exp(0.6 + 0.3 * np.sin(2 * np.pi * t.day_of_year.to_numpy() / 365) + rng.normal(0, 0.25, n))
    u = np.argsort(np.argsort(m)) / n
    g = (u - 0.5) * 2
    pas = np.arange(n) // 4
    e = si * (0.8 * rng.standard_normal(pas.max() + 1)[pas] + 0.6 * rng.standard_normal(n))
    o = m * np.exp(-b - b_nivel * g + e)
    return pd.DataFrame(
        {
            "tiempo": t,
            "hs_mod": m,
            "hs_obs": o,
            "hs_nodo": m * 1.05,
            "unidad": [f"p{seed}_{i}" for i in pas],
            "pasada": [f"p{seed}_{i}" for i in pas],
            "mision": mision,
            "tramo_cal": tramo,
            "n": 8,
            "dist_costa_km": rng.uniform(50, 200, n),
            "costa_min_km": 50.0,
        }
    )


def _df_boya(n=1500, inicio="1986-01-01", b=0.0, seed=0, boya="32302"):
    d = _df_alt(n, inicio, b, seed=seed).drop(columns=["mision", "tramo_cal", "n", "pasada"])
    return d.assign(boya=boya, k=3)


def _medir(nombre, df, alt=True):
    df = PR.ordenar(df)
    s = np.zeros(len(df))  # sin ruido: el núcleo se prueba aparte de SIMEX
    sig = {v: s for v in EO.VARIANTES}
    return PR.medir_lado(nombre, df, sig, B_PRUEBA)


# --- Lectura y control de calidad ----------------------------------------------------------------------------


def test_qc_boya_pico_aislado_fuera_y_tormenta_suave_dentro():
    t = pd.date_range("2010-01-01", periods=96, freq="h")
    hs = 2.0 + 0.02 * np.sin(np.arange(96) / 5) + np.linspace(0, 3, 96)  # tormenta que crece suave
    hs[40] = 9.0  # pico aislado
    reg = pd.DataFrame({"tiempo": t, "WVHT": hs, "DPD": 12.0, "APD": 8.0})
    reg.loc[10, "WVHT"] = np.nan  # centinela ya convertido
    r, qc = lectura.control_boya(reg, "32302")
    assert np.isnan(r.loc[40, "WVHT"])
    assert r["WVHT"].notna().sum() == 96 - 2
    assert [q["filtro"] for q in qc][-1] == "prueba de pico (manual de NDBC)"


def test_pasos_de_boya_media_y_minimo_de_dos_registros():
    t = pd.to_datetime(["2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 04:00", "2010-01-01 07:00"])
    reg = pd.DataFrame({"tiempo": t, "WVHT": [1.0, 3.0, 2.0, 5.0], "DPD": [10, 12, np.nan, 9], "APD": 7.0})
    p = lectura.pasos(reg)
    # 01:00 y 02:00 caen en el paso de las 03:00 ([01:30, 04:30) no: el paso de 03:00 cubre [01:30, 04:30))
    assert list(p["k"]) == [2]
    fila = p.iloc[0]
    assert fila["tiempo"] == pd.Timestamp("2010-01-01 03:00")
    assert fila["hs_obs"] == pytest.approx(2.5)
    assert np.isnan(fila["dpd_obs"])  # un solo DPD válido en la ventana


def test_superobservacion_con_hs_y_minimo_de_tres_puntos():
    d = pd.DataFrame(
        {
            "mision": "HY-2A",
            "pasada": ["a"] * 3 + ["b"] * 2,
            "celda": "x",
            "tiempo": pd.date_range("2012-01-01", periods=5, freq="s"),
            "lat": -15.0,
            "lon": -76.0,
            "dist_costa_km": 60.0,
            "dist_nodo_km": 50.0,
            "tramo_cal": "HY-2A_2",
            "hs": [1.0, 2.0, 3.0, 9.0, 9.0],
        }
    )
    so = lectura.superobservaciones(d, 50.0)
    assert len(so) == 1 and so.loc[0, "hs_obs"] == pytest.approx(2.0) and so.loc[0, "n"] == 3


# --- Ruido y función de estructura -----------------------------------------------------------------------------


def _puntos(n_pasadas=300, sigma=0.2, ar=0.0, campo=0.0, seed=1):
    rng = np.random.default_rng(seed)
    filas = []
    for p in range(n_pasadas):
        k = 40
        base = rng.uniform(1.0, 1.8)
        e = np.empty(k)
        e[0] = rng.standard_normal()
        for i in range(1, k):
            e[i] = ar * e[i - 1] + np.sqrt(1 - ar * ar) * rng.standard_normal()
        lat = -14.0 - 0.06 * np.arange(k)  # unos 6.7 km entre puntos
        peq = campo * rng.standard_normal(k)  # variabilidad de pequeña escala, independiente entre puntos
        filas.append(
            pd.DataFrame(
                {
                    "mision": "HY-2A",
                    "pasada": f"p{p}",
                    "tiempo": pd.Timestamp("2012-01-01") + pd.to_timedelta(p * 3600 + np.arange(k), unit="s"),
                    "lat": lat,
                    "lon": -76.0,
                    "hs": base + peq + sigma * e,
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_estimador_de_sigma_blanco_y_ar1():
    sig = ruido.sigma_1hz(_puntos(sigma=0.2))
    assert sig.loc[sig["nivel"] == 0, "sigma"].iloc[0] == pytest.approx(0.2, rel=0.10)
    sig_ar = ruido.sigma_1hz(_puntos(sigma=0.2, ar=0.5))
    assert sig_ar.loc[sig_ar["nivel"] == 0, "sigma"].iloc[0] < 0.2 * 0.85  # subestima, como se declara


def test_funcion_de_estructura_sin_y_con_pequena_escala():
    p0 = _puntos(sigma=0.2)
    tabla = ruido.tabla_sigma(ruido.sigma_1hz(p0))
    e0 = ruido.estructura(p0, tabla)
    assert abs(e0["varianza_pequena_escala"]) < 0.1 * 0.04 and not e0["supera_umbral"]
    p1 = _puntos(sigma=0.2, campo=0.15)
    t1 = {k: 0.2 for k in tabla}  # el σ verdadero del ruido
    e1 = ruido.estructura(p1, t1)
    assert e1["varianza_pequena_escala"] == pytest.approx(0.15**2, rel=0.15) and e1["supera_umbral"]


# --- Cocientes, cola combinada y estimabilidad -----------------------------------------------------------------


def test_cociente_por_cuantil_recupera_un_sesgo_constante_y_uno_creciente():
    lo = _medir("A", _df_alt(b=0.05, si=0.0))
    B = lo.med["principal"].B
    assert B[:3] == pytest.approx([0.05] * 3, abs=1e-9)
    lo2 = _medir("A", _df_alt(b=0.0, b_nivel=0.04, si=0.0))
    B2 = lo2.med["principal"].B
    assert B2[S.IDX["P99"]] > B2[S.IDX["P90"]] > B2[S.IDX["P50"]] - 1e-9


def test_cola_combinada_pesos_inversos_y_no_estimable_sin_fallar():
    lo = _medir("A", _df_alt(si=0.08))
    errores = PR.errores_calibracion({"A": lo}, B_PRUEBA)
    v = PR.v3(lo, "principal", errores, True, {})
    pesos = v["B_cola"]["pesos"]
    assert sum(pesos.values()) == pytest.approx(1.0)
    rA = PR.con_cal(lo, "principal", errores)
    var = np.var(rA[:, P._idx(P.ESTS_V3A)], axis=0, ddof=1)
    esperado = (1 / var) / (1 / var).sum()
    assert list(pesos.values()) == pytest.approx(list(esperado))
    lo.med["principal"].estimable[S.IDX["P99"]] = False
    v2 = PR.v3(lo, "principal", errores, True, {})
    assert v2["B_cola"]["pesos"]["P99"] == 0.0 and v2["B_cola"]["estimado"] is not None


# --- SI y ρ observados, punto de la potencia recalculada -----------------------------------------------------------


def test_icc_recupera_la_fraccion_comun():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(2000), 4)
    x = np.sqrt(0.8) * rng.standard_normal(2000)[g] + np.sqrt(0.2) * rng.standard_normal(g.size)
    assert PR.icc(x, g) == pytest.approx(0.8, abs=0.03)


def test_si_observado_sin_sesgo_ni_ruido():
    df = _df_alt(si=0.0)
    df["hs_obs"] = df["hs_mod"] * 0.9  # solo sesgo: SI 0
    lo = _medir("A", df)
    assert PR.si_observado(lo, np.zeros(lo.L.n)) == pytest.approx(0.0, abs=1e-9)


def test_punto_observado_toma_el_peor():
    p, d = SO.punto_observado({"A": 0.10, "A1": 0.14, "T0": 0.12, "T0b": 0.2}, {"A": 0.7, "T0b": 0.9})
    assert (p.si_asimilado, p.si_t0, p.rho, p.factor_cal) == (0.14, 0.2, 0.9, 1.0)
    p2, _ = SO.punto_observado({"A": 0.05}, {"A": 0.5})
    assert (p2.si_asimilado, p2.si_t0, p2.rho) == (0.12, 0.16, 0.8)


# --- Potencia: reconstrucción desde la tabla de la etapa 2 --------------------------------------------------------


def test_potencia_desde_replicas_publicadas_es_la_misma():
    rng = np.random.default_rng(0)
    reps = []
    for _ in range(60):
        rp = {}
        for k, _a in EO.EP.PRUEBAS:
            n = {"V3a": 3, "V3a_sinmax": 2, "V3b": 1, "V3b_sinmax": 1}.get(k, 4)
            est = rng.normal(0, 0.03, n)
            est[rng.random(n) < 0.2] = np.nan
            rp[k] = S.Prueba(est, np.full(n, 0.02), {0.025: 2.4, 0.05: 2.1}, 0.5)
        reps.append(rp)
    tabla = pd.DataFrame([f for i, rp in enumerate(reps) for f in _fila_replica("x", i, rp)])
    rec = D.replicas_desde_tabla(tabla)
    for k, a in EO.EP.PRUEBAS:
        for b in (0.0, 0.03, -0.05):
            assert P.potencia(rec, k, b, a) == P.potencia(reps, k, b, a)


# --- Categorías, familias y escenarios ------------------------------------------------------------------------


def _f(atr=True, rech=False, signo=1, desc=False):
    return {"atribuye": atr, "rechaza": rech, "signo": signo, "descartada": desc}


def test_combinacion_de_fuentes():
    assert D.familia({"a": _f(rech=True), "b": _f()}, 0.3)["categoria"] == D.ESTABLECIDO
    r = D.familia({"a": _f(rech=True, signo=1), "b": _f(rech=True, signo=-1)}, 0.3)
    assert r["categoria"] == D.NO_CONCLUYENTE and D.M_CONTRADICCION in r["marcas"]
    r = D.familia({"a": _f(rech=True), "b": _f(desc=True)}, 0.3)
    assert r["categoria"] == D.NO_CONCLUYENTE and D.M_EPOCA in r["marcas"]
    assert D.familia({"a": _f(), "b": _f()}, 0.85)["categoria"] == D.DESCARTADO
    assert D.familia({"a": _f(), "b": _f()}, 0.5)["categoria"] == D.NO_CONCLUYENTE
    # La fuente que no atribuye no cuenta: su rechazo no establece nada
    assert D.familia({"a": _f(), "b": _f(atr=False, rech=True)}, 0.3)["categoria"] == D.NO_CONCLUYENTE


def _fam(cat, rechazan=(), marcas=(), mf_a=()):
    return {
        "categoria": cat,
        "fuentes_que_rechazan": list(rechazan),
        "marcas": list(marcas),
        "marcas_fuente": {"a": list(mf_a)},
    }


def test_regla_6_1_ninguna_retira_e1_y_las_marcas_solo_impiden():
    cats = (D.ESTABLECIDO, D.DESCARTADO, D.NO_CONCLUYENTE)
    for c1 in cats:
        for c2 in cats:
            for rech in ((), ("a",), ("b",)):
                for mf in ((), (D.M_TRASLADO,)):
                    e = D.escenarios(_fam(c1, rech, (), mf), _fam(c2, rech, (), mf), D.ESTABLECIDO, True)
                    assert e["E1"] == "activo"
                    retira = e["E0"] != "activo" or e["E2"] != "activo"
                    if mf or "a" not in rech:
                        assert not retira
    e = D.escenarios(_fam(D.ESTABLECIDO, ("a",)), _fam(D.NO_CONCLUYENTE), D.NO_CONCLUYENTE, None)
    assert (e["E0"], e["base"]) == ("sensibilidad", "E1")
    e = D.escenarios(_fam(D.ESTABLECIDO, ("b",)), _fam(D.NO_CONCLUYENTE), D.NO_CONCLUYENTE, None)
    assert e["E0"] == "activo" and D.M_CUENCA in e["marcas"]["V1"]


def test_v2_establecido_retira_e2_solo_con_las_tres_condiciones():
    v1 = _fam(D.NO_CONCLUYENTE)
    assert D.escenarios(v1, _fam(D.ESTABLECIDO, ("a",)), D.ESTABLECIDO, True)["E2"] == "retirado"
    assert D.escenarios(v1, _fam(D.ESTABLECIDO, ("a",)), D.ESTABLECIDO, False)["E2"] == "activo"
    assert D.escenarios(v1, _fam(D.ESTABLECIDO, ("a",)), D.NO_CONCLUYENTE, True)["E2"] == "activo"
    e = D.escenarios(v1, _fam(D.ESTABLECIDO, ("b",)), D.ESTABLECIDO, True)
    assert e["E2"] == "activo" and D.M_CUENCA in e["marcas"]["V2"]
    assert D.escenarios(v1, _fam(D.DESCARTADO), D.ESTABLECIDO, True)["E2"] == "activo"


def test_marcas_de_robustez_y_signo():
    res = {"rechaza": True, "decide": "P90", "signo": -1}
    m = D.marcas_fuente(res, {"rechaza": False}, {"rechaza": False}, {"P90": 0.04}, "válido")
    assert set(m) == {D.M_SIGNO, D.M_CALIBRACION, D.M_SISTEMA}
    m = D.marcas_fuente(res | {"signo": 1}, {"rechaza": True}, None, {"P90": 0.04}, "dudoso")
    assert m == [D.M_TRASLADO]


# --- Corrección (§6.2) -------------------------------------------------------------------------------------------


def _v3(bc=-0.03, ba=-0.05):
    return {
        "B_cola": {"estimado": bc},
        "B_alta": {"estimado": ba, "estadistico": "max"},
        "B_q": {"P50": 0.0, "P90": -0.02, "P99": -0.04, "P99.9": None},
    }


def test_regla_6_2_variantes_y_la_mas_desfavorable():
    r = D.variantes(D.NO_CONCLUYENTE, D.NO_CONCLUYENTE, _v3())
    assert [v["nombre"] for v in r["variantes"]] == ["sin_corregir", "B_cola", "B_alta"]
    assert r["mas_desfavorable"] == "B_alta"  # el producto subestima la cola: B_alta más negativo
    r = D.variantes(D.NO_CONCLUYENTE, D.DESCARTADO, _v3(bc=0.03))
    assert r["mas_desfavorable"] == "sin_corregir"  # sobrestima: corregir bajaría el diseño
    assert [v["nombre"] for v in D.variantes(D.ESTABLECIDO, D.DESCARTADO, _v3())["variantes"]] == ["B_cola"]
    assert [v["nombre"] for v in D.variantes(D.ESTABLECIDO, D.ESTABLECIDO, _v3())["variantes"]] == ["cuantiles"]
    assert [v["nombre"] for v in D.variantes(D.DESCARTADO, D.NO_CONCLUYENTE, _v3())["variantes"]] == ["sin_corregir"]


def test_correccion_por_cuantiles_monotona_y_sin_extrapolar():
    t = pd.date_range("2000-01-01", periods=5000, freq="3h")
    hs = np.random.default_rng(0).lognormal(0.6, 0.3, t.size)
    v = {"tipo": "corrección por cuantiles", "B_q": {"P50": 0.0, "P90": -0.10, "P99": -0.20, "P99.9": None}}
    c = D.corregir(t, hs, v, np.ones(t.size, bool))
    o = np.argsort(hs)
    assert np.all(np.diff(c[o]) >= -1e-12)
    top = hs >= np.quantile(hs, 0.995)
    assert np.allclose(c[top] / hs[top], np.exp(0.20), rtol=1e-6)  # constante por encima del último estimable
    low = hs <= np.quantile(hs, 0.3)
    assert np.allclose(c[low], hs[low])  # por debajo del P50, el valor del P50 (0)


def test_serie_corregida_con_factor_no_crea_escalon():
    t = pd.date_range("1980-01-01", "2026-08-31 21:00", freq="3h")
    hs = np.random.default_rng(1).lognormal(0.6, 0.3, t.size)
    v = {"tipo": "factor constante", "B": -0.04}
    toda = D.corregir(t, hs, v, np.ones(t.size, bool))
    assert not any(f["agranda"] for f in complementos.escalones_serie(t, hs, toda))
    asim = D.corregir(t, hs, v, t >= pd.Timestamp(C.INICIO_ASIMILADO))
    assert not any(f["agranda"] for f in complementos.escalones_serie(t, hs, asim, C.CAMBIOS_SERIE[1:]))
    malo = asim.copy()
    malo[t >= pd.Timestamp("2016-03-01")] *= 1.05
    assert any(f["agranda"] for f in complementos.escalones_serie(t, hs, malo, C.CAMBIOS_SERIE[1:]))


# --- Traslado por el gradiente y absorción de V2 ----------------------------------------------------------------


def test_traslado_por_gradiente_valido_con_mucha_muestra_y_dudoso_con_poca():
    grande = _medir("A", _df_alt(n=8000, si=0.03, seed=4))
    g = PR.gradiente(grande, np.zeros(grande.L.n), 4, B_PRUEBA, 23.0, C.TOL_GRADIENTE)
    assert g["estado"] == "válido"
    chico = _medir("A", _df_alt(n=400, si=0.15, seed=5))
    g2 = PR.gradiente(chico, np.zeros(chico.L.n), 4, B_PRUEBA, 23.0, C.TOL_GRADIENTE)
    assert g2["estado"] in ("dudoso", "no evaluable (menos de dos bandas estimables)")


def test_absorcion_v2_con_el_mismo_escalon_y_no_con_otro():
    """§6.1 (iii): el producto y el nodo tienen el mismo escalón de +8 % en 2016-03 → la corrección por lado lo absorbe.
    Si el nodo no lo tiene, no."""
    a1 = _df_alt(n=6000, inicio="2011-10-15", si=0.0, seed=6)
    a1["hs_obs"] = np.exp(
        0.6 + 0.02 * np.random.default_rng(1).standard_normal(len(a1))
    )  # poca dispersión: IC estrecho
    a1["hs_mod"] = a1["hs_nodo"] = a1["hs_obs"]
    a2 = a1.copy()
    a2["tiempo"] = a2["tiempo"] + pd.Timedelta(days=1600)
    a2["hs_mod"] *= 1.08
    a2["hs_nodo"] *= 1.08
    l1, l2 = _medir("A1", a1), _medir("A2", a2)
    cero = PR.escalar(PR.errores_calibracion({"A1": l1, "A2": l2}, B_PRUEBA), 0.0)
    assert PR.absorcion_v2(l1, l2, cero, C.TOL_V2_ABSORBE)["absorbe"]
    a2b = a2.assign(hs_nodo=a2["hs_nodo"] / 1.08)  # el nodo sin escalón
    assert not PR.absorcion_v2(l1, _medir("A2", a2b), cero, C.TOL_V2_ABSORBE)["absorbe"]


# --- Extremo a extremo del núcleo: sesgo de −12 % en T0 -------------------------------------------------------------


def _pj():
    comps = [c.nombre for c in CV.COMPARACIONES]
    return {
        "atribuye": {c: c != "V2b" for c in comps},
        "traslado_p90": {c: "válido" for c in comps},
        "validacion_por_prueba": {"V1a": True, "V1b": True, "V3a": True, "V3b": False},
    }


def _esc():
    return pd.DataFrame(
        [{"comparacion": c.nombre, "estadistico": e, "s_mod": 0.08} for c in CV.COMPARACIONES for e in PR.ESTS]
    )


def _nucleo(b_t0):
    """T0 y T0b con o = m·exp(−b_t0), es decir, B = b_t0; A y T1b sin sesgo."""
    lados_df = {
        "T0": _df_alt(n=2500, inicio="1985-03-31", b=b_t0, si=0.05, seed=10, mision="GEOSAT", tramo="GEOSAT"),
        "A": _df_alt(n=4000, inicio="2012-01-01", si=0.05, seed=11),
        "A1": _df_alt(n=2500, inicio="2011-10-15", si=0.05, seed=12),
        "A2": _df_alt(n=2500, inicio="2016-03-01", si=0.05, seed=13),
        "T0b": _df_boya(n=2500, inicio="1986-01-01", b=b_t0, seed=14),
        "T1b": _df_boya(n=2500, inicio="1993-02-01", seed=15),
    }
    lados = {n: _medir(n, d, alt=not n.endswith("b")) for n, d in lados_df.items()}
    errores = PR.escalar(PR.errores_calibracion(lados, B_PRUEBA), 0.0)  # sin calibración: prueba del núcleo
    comps = EO.comparaciones(lados, errores, {})
    v3 = PR.v3(lados["A"], "principal", errores, True, {}) | {"con_maximos": True}
    pot = {"V1a": 0.5, "V1b": 0.99, "V1b_lado": 0.5, "V2a": 0.2, "V3a": 0.2, "V3b": 0.1, "V1 conjunta": 0.99}
    pot["V1 conjunta con lado"] = 0.6
    return EO.decidir(comps, {"principal": v3}, "principal", _pj(), _esc(), {"incoherencia": False}, False, pot)


def test_extremo_a_extremo_sesgo_en_t0_establece_v1_y_e1_pasa_a_base():
    # Sesgo del producto de −12 % en T0 (el modelo 12 % por debajo de la observación): B_T0 = log(0.88), y
    # Δ1 = B_A − B_T0 = +0.128, del mismo signo que el escalón del modelo (+0.08): sin la marca de signo.
    d = _nucleo(math.log(0.88))
    v1 = d["familias"]["V1"]
    assert v1["categoria"] == D.ESTABLECIDO and set(v1["fuentes_que_rechazan"]) == {"a", "b"}
    assert d["escenarios"]["E0"] == "sensibilidad" and d["escenarios"]["base"] == "E1"
    assert d["escenarios"]["E1"] == "activo" and d["escenarios"]["E2"] == "activo"


def test_extremo_a_extremo_mismo_sesgo_a_los_dos_lados_no_rechaza():
    d = _nucleo(0.0)
    assert d["familias"]["V1"]["fuentes_que_rechazan"] == []
    assert d["escenarios"]["E0"] == "activo"


def test_boya_no_descartada_si_no_se_sostiene_con_el_termino_por_lado():
    d = _nucleo(0.0)
    # V1b: potencia 0.99 sin el término por lado y 0.5 con él: no puede salir «descartada»
    assert d["familias"]["V1"]["categorias_por_fuente"]["V1b"] == D.NO_CONCLUYENTE
    assert d["familias"]["V2"]["categorias_por_fuente"]["V2b"] in (D.SIN_ESCALON, "sin colocaciones")


# --- §20.6 ------------------------------------------------------------------------------------------------------


def test_rango_de_la_subida_esperada_20_6():
    rejilla = [P.Punto(f"p{i}", sa, s0, rho, f) for i, (sa, s0, rho, f) in enumerate(
        [(0.09, 0.12, 0.8, 1.0), (0.09, 0.12, 0.8, 2.0), (0.16, 0.2, 0.8, 1.0), (0.16, 0.2, 0.8, 2.0),
         (0.12, 0.16, 0.8, 0.5), (0.12, 0.16, 0.5, 1.0)])]  # fmt: skip
    valores = [0.0118, 0.0314, 0.0101, 0.0225, 0.005, 0.0089]
    pj = {
        "rejilla": [{"nombre": p.nombre, "rho": p.rho, "factor_cal": p.factor_cal} for p in rejilla],
        "subida_esperada": {
            p.nombre: {"con_maximos": {"subida_media": v}, "sin_maximos": {"subida_media": v + 0.005}}
            for p, v in zip(rejilla, valores, strict=True)
        },
    }
    r = D.rango_subida(pj)
    assert r["con_maximos"]["limite_inferior"] == pytest.approx(0.0101)
    assert r["con_maximos"]["limite_superior"] == pytest.approx(0.0314)
    assert "2.25 a 3.14 %" in D.hallazgo_20_6(r)  # × 0.5 y ρ = 0.5 fuera de la envolvente


# --- Períodos y saltos ------------------------------------------------------------------------------------------


def _boy_periodos(shift2: float):
    rng = np.random.default_rng(2)
    filas = []
    for inicio, sh in (("1986-01-01", 0.0), ("1993-02-01", shift2)):
        t = pd.date_range(inicio, periods=4000, freq="9h")
        hs = rng.lognormal(0.7, 0.3, t.size)
        apd = 8 + rng.normal(0, 0.2, t.size)
        filas.append(
            pd.DataFrame(
                {
                    "tiempo": t,
                    "boya": "32302",
                    "hs_obs": hs,
                    "apd_obs": apd,
                    "tm02_mod": apd + 0.5 + sh,
                    "dpd_obs": 12.0,
                    "tp_mod": 12.5,
                }
            )  # fmt: skip
        )
    return pd.concat(filas, ignore_index=True)


def test_periodos_desplazamiento_del_producto_y_del_clima():
    per = complementos.periodos(_boy_periodos(1.0), 199)
    fila = per[(per["boya"] == "32302") & (per["lectura"] == "principal") & (per["lado"] == "T1b − T0b")]
    assert fila["clase"].iloc[0].startswith("desplazamiento de Tp del producto")
    per0 = complementos.periodos(_boy_periodos(0.0), 199)
    fila0 = per0[(per0["boya"] == "32302") & (per0["lectura"] == "principal") & (per0["lado"] == "T1b − T0b")]
    assert fila0["clase"].iloc[0].startswith("desplazamiento de Tp del clima")


def test_saltos_con_pasada_a_hora_y_media():
    t = pd.date_range("2015-01-01", periods=10, freq="3h")
    hs = pd.Series([2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], index=t)
    so = pd.DataFrame(
        {"mision": ["HY-2A", "HY-2A"], "tiempo": [t[3] + pd.Timedelta(minutes=80), t[3] + pd.Timedelta(hours=2)],
         "celda": "x", "dist_nodo_km": 50.0, "hs_mod": 2.9, "hs_obs": 3.1, "costa_min_km": 50.0}
    )  # fmt: skip
    s = complementos.saltos_validados(hs, 0.2, so)
    assert len(s) == 1 and s.loc[0, "hs_nodo_antes"] == 2.0 and s.loc[0, "hs_observada"] == 3.1


# --- G4: la etapa 3 no lee valores sin la potencia publicada e intacta ---------------------------------------


def _potencia_publicada(raiz, monkeypatch, valores_leidos=False):
    d = EO.EP.ruta_potencia(raiz)
    d.mkdir(parents=True)
    for f in EO.ARCHIVOS_POTENCIA:
        (d / f).write_text("x\n", encoding="utf-8")
    (d / "potencia.json").write_text(json.dumps({"valores_observados_leidos": valores_leidos}), encoding="utf-8")
    man = {
        "gate": {"aprobado": True},
        "huella_codigo": "h-etapa2",
        "salidas": {f: manifiesto.sha256(d / f) for f in EO.ARCHIVOS_POTENCIA},
    }
    (d / ET.MANIFIESTO).write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(EO.orquestador, "huella_etapa", lambda et, *a, **k: "h-etapa2")
    return d


def test_g4_falla_sin_potencia_o_con_hash_distinto_y_no_lee_valores(tmp_path, monkeypatch):
    def prohibido(*a, **k):
        raise AssertionError("leyó valores observados antes de comprobar G4")

    monkeypatch.setattr(lectura, "leer_altimetria", prohibido)
    monkeypatch.setattr(lectura, "leer_boyas", prohibido)
    with pytest.raises(GateFallido, match="falta potencia.csv"):
        EO.validacion_etapa(tmp_path)
    d = _potencia_publicada(tmp_path, monkeypatch)
    assert EO.verificar_g4(tmp_path)["comprobado_antes_de_leer_valores"]
    (d / "potencia.csv").write_text("otra cosa\n", encoding="utf-8")
    with pytest.raises(Exception, match="cambió después de publicarse"):
        EO.validacion_etapa(tmp_path)


def test_g4_falla_si_el_codigo_de_la_etapa_2_cambio_o_si_ya_leyo_valores(tmp_path, monkeypatch):
    _potencia_publicada(tmp_path, monkeypatch)
    monkeypatch.setattr(EO.orquestador, "huella_etapa", lambda et, *a, **k: "otra")
    with pytest.raises(GateFallido, match="código de la etapa 2 cambió"):
        EO.verificar_g4(tmp_path)
    otra = tmp_path / "otra"
    _potencia_publicada(otra, monkeypatch, valores_leidos=True)
    with pytest.raises(GateFallido, match="valores_observados_leidos"):
        EO.verificar_g4(otra)


def test_la_etapa_3_no_cambia_la_huella_de_la_etapa_2():
    """El código de la etapa 3 vive fuera de ``validacion/``: la huella de la etapa 2 no lo incluye."""
    rutas = orquestador.rutas_codigo(EO.EP.ETAPA_POTENCIA)
    assert not any("validacion_obs" in str(p) for p in rutas)


def test_signo_opuesto_al_escalon_del_modelo_impide_retirar_e0():
    d = _nucleo(math.log(1 / 0.88))  # Δ1 negativo frente a un escalón del modelo positivo
    assert d["familias"]["V1"]["categoria"] == D.ESTABLECIDO
    assert D.M_SIGNO in d["familias"]["V1"]["marcas_fuente"]["a"]
    assert d["escenarios"]["E0"] == "activo"


def test_g6_fallido_conserva_las_salidas_en_interim(tmp_path):
    """§20.7, punto 19: con un estimado fuera de su IC, las salidas se copian a interim/ antes de que falle el gate."""
    tabla = pd.DataFrame(
        [
            {
                "lado": "A",
                "estadistico": "P99",
                "estimable": True,
                "R_descontado": 1.10,
                "R_ic95_lo": 0.9,
                "R_ic95_hi": 1.05,
            }
        ]
    )
    fuera = EO.dentro_de_ic(tabla, {})
    assert fuera == ["A P99"]
    st = tmp_path / "staging"
    (st / "figuras").mkdir(parents=True)
    (st / "resumen_f05.json").write_text("{}", encoding="utf-8")
    (st / "figuras" / "a.png").write_bytes(b"x")
    destino = EO.conservar_gate_fallido(st, EO.ruta_gate_fallido(tmp_path), fuera, ["G6"])
    assert (destino / "resumen_f05.json").is_file() and (destino / "figuras" / "a.png").is_file()
    info = json.loads((destino / "G6_fuera_de_ic.json").read_text(encoding="utf-8"))
    assert info["estimados_fuera_de_su_ic"] == ["A P99"]
    assert "interim" in destino.parts and "processed" not in destino.parts


# --- Notas posteriores a los datos (§20.8) ------------------------------------------------------------------------


def _comp(rech, p, p2, rech2, p90):
    return {
        "principal": {"rechaza": rech, "p": p, "estimado": {"P90": p90}},
        "calibracion_x2": {"rechaza": rech2, "p": p2},
    }


def test_nota_discrepancia_v1_solo_si_una_fuente_rechaza():
    comps = {"V1a": _comp(True, 0.003, 0.10, False, 0.092), "V1b": _comp(False, 0.12, 0.32, False, 0.006)}
    pot = {"V1a": 0.091, "V1b_lado": 0.527}
    s_mod = {"V1a": 0.041, "V1b": 0.089}
    n = notas.discrepancia_v1(comps, pot, s_mod, {"fuentes_que_rechazan": ["a"]})
    assert "discrepan" in n["texto"] and "2.2 veces" in n["texto"] and "no se sostiene" in n["texto"]
    assert n["rotulo"].startswith("Posterior a los datos")
    assert notas.discrepancia_v1(comps, pot, s_mod, {"fuentes_que_rechazan": ["a", "b"]}) is None


def test_nota_serie_corregida_frente_al_dos_por_ciento():
    filas = [
        {"variante": "cuantiles", "alcance": "toda la serie (E0)", "cambio": "1993-02", "estadistico": "P90",
         "S_sin_corregir": 0.0953, "S_corregida": 0.1133, "agranda": True},
        {"variante": "cuantiles", "alcance": "toda la serie (E0)", "cambio": "1993-02", "estadistico": "max",
         "S_sin_corregir": 0.0978, "S_corregida": 0.1108, "agranda": True},
        {"variante": "cuantiles", "alcance": "tramo asimilado (E1, E2)", "cambio": "2016-03", "estadistico": "P50",
         "S_sin_corregir": -0.0216, "S_corregida": -0.0233, "agranda": True},
    ]  # fmt: skip
    n = notas.hallazgo_serie_corregida(filas)
    assert n["aumento_maximo"] == pytest.approx(0.018) and not n["llega_al_efecto_relevante"]
    e0 = [m for m in n["marcas_para_la_F9"] if m["cambio"] == "1993-02"][0]
    assert e0["marca"] == "agranda el escalón de 1993-02 en 0.018 en P90"
    assert notas.hallazgo_serie_corregida([dict(filas[0], agranda=False)]) is None


def test_nota_linea_448_y_hallazgo_20_6_con_la_configuracion_elegida():
    assert notas.lectura_linea_448("establecido", "activo", True) is not None
    assert notas.lectura_linea_448("establecido", "sensibilidad", True) is None
    assert notas.lectura_linea_448("no concluyente", "activo", True) is None
    r = {
        c: {"rejilla_x2_min": a, "rejilla_x2_max": b, "limite_inferior": li, "limite_superior": b}
        for c, a, b, li in (("con_maximos", 0.0225, 0.0314, 0.0101), ("sin_maximos", 0.0257, 0.0280, 0.0169))
    }
    t = D.hallazgo_20_6(r, "sin_maximos")
    assert "2.25 a 3.14 %" in t and "sin máximos; × 2 de 2.57 a 2.80 %" in t and "1.69–2.80 %" in t
