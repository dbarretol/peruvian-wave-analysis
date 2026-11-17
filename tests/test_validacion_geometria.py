"""Etapa 2, parte A (contrato v6 §3.2–§3.4, §3.7, §4.4 y §8): colocalización geométrica y escalón del modelo."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
import shapely

from peruvian_wave_analysis.homogeneidad import cuantiles
from peruvian_wave_analysis.sitio.observaciones import Costa
from peruvian_wave_analysis.validacion import colocacion as K
from peruvian_wave_analysis.validacion import config as C
from peruvian_wave_analysis.validacion import escalon as E
from peruvian_wave_analysis.validacion import geometria as G
from peruvian_wave_analysis.validacion import ndbc

SECTORES = pd.DataFrame({"sector": range(16), "clase": ["abierto"] * 16})


def _costa():
    # tierra al este de −75.0: la costa es el meridiano −75.0 (≈ 86 km del nodo)
    return Costa.desde_tierra(shapely.box(-75.0, -20.0, -70.0, -10.0), *C.NODO)


def _med(puntos):
    return pd.DataFrame(
        [
            {"mision": m, "banda": "KU", "tiempo": pd.Timestamp(t), "lat": la, "lon": lo, "prof_archivo_m": -3000.0}
            for m, t, la, lo in puntos
        ]
    )


def _prof(la, lo):
    return np.full(np.shape(la), 3000.0)


def test_colocalizacion_celdas_costa_y_superobservaciones():
    t0 = pd.Timestamp("2014-05-01 10:00:00")
    pts = []
    # pasada 1: 4 puntos en la celda (−15.0, −76.4), a > 50 km de la costa; 2 en (−15.2, −76.4): no forman
    for i in range(4):
        pts.append(("HY-2", t0 + pd.Timedelta(seconds=i), -15.01 + 0.001 * i, -76.41))
    for i in range(2):
        pts.append(("HY-2", t0 + pd.Timedelta(seconds=10 + i), -15.19, -76.39))
    # pasada 2 (hueco > 60 s): 3 puntos a ~ 30 km de la costa (solo en la sensibilidad de 25 km)
    for i in range(3):
        pts.append(("HY-2", t0 + pd.Timedelta(seconds=200 + i), -15.0, -75.28))
    # otra misión fuera de la F5 y un punto a más de 200 km
    pts.append(("SARAL", t0, -15.0, -76.4))
    pts.append(("HY-2", t0 + pd.Timedelta(seconds=400), -17.0, -76.4))
    celdas = ["-15.0_-76.4", "-15.2_-76.4", "-15.0_-75.2"]
    d, qc = K.filtrar_altimetria(_med(pts), _costa(), _prof, SECTORES, celdas)
    assert set(d["mision"]) == {"HY-2A"} and (d["tramo_cal"] == "HY-2A_2").all()
    assert d["pasada"].nunique() == 2
    p50 = K.superobservaciones(d, C.COSTA_MIN_KM)
    assert list(p50["celda"]) == ["-15.0_-76.4"] and int(p50["n"].iloc[0]) == 4
    p25 = K.superobservaciones(d, C.COSTA_MIN_SENS_KM)
    assert sorted(p25["celda"]) == ["-15.0_-75.2", "-15.0_-76.4"]
    assert qc[1]["quedan"] == len(pts) - 1  # sin SARAL


def test_hy2a_sin_calibracion_entre_tramos_no_entra():
    m = pd.Series(["HY-2A", "HY-2A", "HY-2A", "HY-2B"])
    t = pd.Series(pd.to_datetime(["2013-04-01 12:00", "2013-04-08 00:00", "2013-04-15 00:00", "2023-03-31 00:00"]))
    assert list(K.tramo_calibracion(m, t)) == ["HY-2A_1", "", "HY-2A_2", "HY-2B"]


def test_ventana_de_hy2b():
    df = pd.DataFrame(
        {
            "mision": ["HY-2B"] * 3,
            "tiempo": pd.to_datetime(["2023-03-31 23:00", "2023-04-01 00:00", "2025-12-31 00:00"]),
        }
    )
    assert list(K.en_lado(df, C.LADOS["A"])) == [True, False, False]
    assert list(K.en_lado(df, C.LADOS["A_hasta_2026"])) == [True, True, True]
    assert list(K.en_lado(df, C.LADOS["A2+HY2B_2023"])) == [True, False, False]
    assert not K.en_lado(df, C.LADOS["A2"]).any()


def test_interpolacion_reproduce_una_rampa():
    idx = pd.date_range("2014-01-01", periods=5, freq="3h")
    s = pd.DataFrame({"c": np.arange(5.0)}, index=idx)
    t = [idx[0] + pd.Timedelta(minutes=90), idx[2], idx[3] + pd.Timedelta(hours=1)]
    assert K.interpolar(s, ["c"] * 3, t) == pytest.approx([0.5, 2.0, 3 + 1 / 3])
    s.iloc[1, 0] = np.nan  # registro excluido por la etapa 1
    assert np.isnan(K.interpolar(s, ["c"], [idx[0] + pd.Timedelta(hours=1)])[0])
    assert np.isnan(K.interpolar(s, ["otra"], [idx[0]])[0])


def test_pasos_de_boya_ventana_y_minimo():
    t = pd.to_datetime(
        [
            "2016-01-01 22:50",
            "2016-01-01 23:50",
            "2016-01-02 00:50",
            "2016-01-02 01:50",
            "2016-01-02 03:50",
            "2016-01-02 23:50",
        ]
    )
    pres = pd.DataFrame({"tiempo": t, "WVHT": [True, True, True, False, True, True]})
    p = K.pasos_boya(pd.concat([pres, pres.iloc[[1]]]))  # un duplicado de borde cuenta una vez
    # 00:00 recibe 22:50, 23:50 y 00:50 (01:50 sin dato); 03:00 solo 03:50; 00:00 del 3 solo 23:50
    assert list(p["tiempo"]) == [pd.Timestamp("2016-01-02 00:00")] and list(p["k"]) == [3]


def test_presencia_con_valores_ilegibles(tmp_path):
    texto = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi    ft
2016 01 01 00 50 999 99.0 99.0  ##### ##### ##### 999 9999.0 999.0  ##### 999.0 99.0 99.00
2016 01 01 01 50 999 99.0 99.0 99.00 99.00 99.00 999 9999.0 999.0  ##### 999.0 99.0 99.00
"""
    r = tmp_path / "32012h2016.txt.gz"
    with gzip.open(r, "wt", encoding="ascii") as f:
        f.write(texto)
    p = ndbc.presencia(r)
    assert p["WVHT"].tolist() == [True, False] and p["APD"].tolist() == [True, False]
    with pytest.raises(ndbc.ErrorNDBC, match="no numéricos"):
        ndbc.estructura(r)


def test_cuantil_con_peso_mensual_igual_que_la_f4():
    rng = np.random.default_rng(1)
    t = pd.to_datetime("2012-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 365, 800)), unit="D")
    x = rng.lognormal(0.5, 0.3, t.size)
    lb = E.LadoBoot.desde(t, x)
    p = E.puntual(lb)[0]
    w = cuantiles.pesos_mensuales(t)
    ref = cuantiles.cuantil_ponderado(x, w, [0.5, 0.9, 0.99, 0.999])
    assert p[:4] == pytest.approx(ref)
    maxm = pd.Series(x, index=t).groupby(t.to_period("M")).max().mean()
    assert p[4] == pytest.approx(maxm)


def test_escalon_igual_en_colocacion_y_nodo_es_valido_y_distinto_es_dudoso():
    rng = np.random.default_rng(2)

    def lado(inicio, factor, ruido):
        t = pd.date_range(inicio, periods=1500, freq="17h")
        clima = np.exp(0.3 * np.sin(2 * np.pi * t.day_of_year.to_numpy() / 365) + rng.normal(0, 0.25, t.size))
        nodo = 2.0 * clima * factor
        col = 2.2 * clima * factor * np.exp(rng.normal(0, ruido, t.size))
        return E.LadoBoot.desde(t, col, nodo)

    es = E.escalon(lado("1986-01-01", 1.0, 0.01), lado("1994-01-01", 1.09, 0.01), 999, rng)
    lo, hi = E.ic(es.replicas[:, 0, 1] - es.replicas[:, 1, 1])
    assert es.puntual[0, 1] - es.puntual[1, 1] == pytest.approx(0.0, abs=0.01)
    assert -0.03 <= lo and hi <= 0.03  # ⅓ de log(1.09): válido
    es2 = E.escalon(lado("1986-01-01", 1.0, 0.3), lado("1994-01-01", 1.09, 0.3), 999, rng)
    lo2, hi2 = E.ic(es2.replicas[:, 0, 1] - es2.replicas[:, 1, 1])
    assert hi2 - lo2 > hi - lo  # IC más ancho con colocaciones ruidosas


def test_regla_del_tercio_y_efecto_por_fuente():
    t1 = pd.date_range("1986-01-01", periods=600, freq="1D")
    t2 = pd.date_range("1994-01-01", periods=600, freq="1D")
    base = 2 + np.sin(np.arange(600) / 30)

    def alt(t, f):
        return pd.DataFrame(
            {
                "mision": "GEOSAT" if t[0].year < 1990 else "HY-2A",
                "tiempo": t,
                "hs_mod": base * f,
                "hs_nodo": base * f,
                "unidad": [str(i) for i in range(600)],
                "costa_min_km": C.COSTA_MIN_KM,
            }
        )

    a = pd.concat([alt(t1.where(t1 <= "1988-08-31").dropna(), 1.0)])
    a = pd.concat([a, alt(pd.date_range("2012-01-01", periods=600, freq="1D"), 1.02)], ignore_index=True)
    res = G.escalones(
        a,
        pd.DataFrame(columns=["boya", "tiempo"]),
        {"V1": np.log(1.09), "V2": np.log(1.03)},
        199,
        np.random.default_rng(0),
    )
    f = res[(res["comparacion"] == "V1a") & (res["estadistico"] == "P90")].iloc[0]
    assert f["efecto_fuente"] == pytest.approx(np.log(1.02), abs=2e-3)
    assert not f["atribuye"]  # log(1.02) < ⅓·log(1.09)
    assert (res.loc[res["comparacion"] == "V1b", "estado"] == "sin colocaciones").all()
    del t2


def test_efectos_del_nodo_se_leen_de_la_f4():
    p = {
        "A": {
            "C1": {"delta_rel_corregido": {"por_lado": {"lado1": {"delta": 0.09}}}},
            "C2": {"delta_rel_corregido": {"por_lado": {"lado1": {"delta": 0.03}}}},
        }
    }
    assert G.efectos_nodo(p) == pytest.approx({"V1": np.log(1.09), "V2": np.log(1.03)})
