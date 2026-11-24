"""F7: ramas y picos (contrato §3.1, §3.2 p. 5 y 6, §3.4 E-29, §3.5 G8, §3.6; §8 «Identificación común», «Borde de
las variantes», «Protección de cola», «Exposición»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.eventos import identificacion as ID
from peruvian_wave_analysis.eventos import ramas as R

INI_E1 = pd.Timestamp("1993-02-01")


def _serie(n=400, inicio="1993-01-20", base=1.5):
    idx = pd.date_range(inicio, periods=n, freq="3h")
    return pd.DataFrame({"VHM0": np.full(n, base), "VTPK": np.arange(n, dtype=float), "VMDR": 200.0,
                         "VPED": 210.0, "VHM0_WW": 0.5, "VHM0_SW1": 1.0, "VHM0_SW2": 0.2}, index=idx)  # fmt: skip


def _tormenta(df, t, alto, ancho=3):
    i = df.index.get_loc(pd.Timestamp(t))
    df.iloc[i - ancho : i + ancho + 1, 0] = np.maximum(df.iloc[i - ancho : i + ancho + 1, 0], alto - 0.3)
    df.iloc[i, 0] = alto
    return df


def test_pico_de_la_variante_es_su_maximo_en_la_ventana():
    df = _tormenta(_serie(), "1993-02-10 12:00", 3.5)
    ev = ID.identificar(df["VHM0"], 2.7, 48.0)
    assert len(ev) == 1
    var = df["VHM0"].copy()
    i = df.index.get_loc(pd.Timestamp("1993-02-10 15:00"))
    var.iloc[i] = 3.8  # la variante tiene su máximo un paso después, dentro de la ventana
    p = R.picos_de_rama(ev, var, df)
    assert p.loc[0, "t_pico"] == pd.Timestamp("1993-02-10 15:00") and p.loc[0, "hs"] == 3.8
    assert p.loc[0, "pico_movido"] and p.loc[0, "hs_e0_en_pico"] == pytest.approx(3.2)
    assert p.loc[0, "tp"] == df.loc[pd.Timestamp("1993-02-10 15:00"), "VTPK"]  # etiquetas en el pico de la rama
    assert p.loc[0, "particion_dominante"] == "SW1" and p.loc[0, "estrato"] == "unico"
    q = R.picos_de_rama(ev, df["VHM0"], df)
    assert not q.loc[0, "pico_movido"] and q.loc[0, "t_pico"] == ev.loc[0, "t_pico"]


def test_e1_toma_los_eventos_con_pico_desde_su_inicio_y_recorta_la_ventana():
    df = _serie()
    df = _tormenta(df, "1993-01-25 00:00", 3.4)  # antes de E1: no entra
    # tormenta que empieza antes del inicio de E1 y tiene el pico de E0 después
    for t, h in (("1993-01-31 18:00", 3.3), ("1993-01-31 21:00", 3.6), ("1993-02-01 00:00", 3.0),
                 ("1993-02-01 03:00", 3.7)):  # fmt: skip
        df.loc[pd.Timestamp(t), "VHM0"] = h
    ev = ID.identificar(df["VHM0"], 2.7, 48.0)
    e1 = R.eventos_de_rama(ev, INI_E1)
    assert len(ev) == 2 and len(e1) == 1
    hs_e1 = df["VHM0"][df.index >= INI_E1]
    var = hs_e1.copy()
    var.loc[pd.Timestamp("1993-02-01 00:00")] = 3.65  # la variante, ya dentro de E1
    p = R.picos_de_rama(e1, var, df)
    assert p.loc[0, "recortado"]
    assert p.loc[0, "t_pico"] == pd.Timestamp("1993-02-01 03:00")
    # el máximo de la variante antes del inicio de E1 (21:00 del 31) no puede ser pico: está fuera de la rama
    var2 = hs_e1.copy()
    var2.loc[pd.Timestamp("1993-02-01 03:00")] = 3.1
    p2 = R.picos_de_rama(e1, var2, df)
    assert p2.loc[0, "t_pico"] == pd.Timestamp("1993-02-01 03:00") and p2.loc[0, "hs"] == 3.1


def test_u_id_llevado_sin_discordancias_con_un_mapa_monotono_fijo():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=5000, freq="3h")
    x = pd.Series(np.round(rng.gamma(4, 0.4, idx.size), 2), index=idx)
    u = float(np.quantile(x, 0.95))
    y = np.maximum.accumulate(np.sort(x.to_numpy()) * 1.07)  # mapa monótono fijo
    var = pd.Series(np.interp(x, np.sort(x.to_numpy()), y), index=idx)
    info = R.u_id_en_variante(x, var, u)
    assert info["estados_discordantes"] == 0
    assert info["u_id_max_m"] == pytest.approx(info["u_id_variante_m"])
    assert R.borde_inferior(pd.DataFrame({"hs": [1.0, 9.0]}), info)[1]["activo"] is False


def test_borde_inferior_con_un_mapa_que_cambia_en_el_tiempo():
    idx = pd.date_range("2000-01-01", periods=2000, freq="3h")
    x = pd.Series(np.tile(np.linspace(1.0, 3.0, 1000), 2), index=idx)
    factor = np.where(np.arange(2000) < 1000, 1.0, 1.2)  # el mapa sube en la segunda mitad
    var = x * factor
    u = 2.5
    info = R.u_id_en_variante(x, var, u)
    assert info["estados_discordantes"] > 0
    assert info["u_id_max_m"] == pytest.approx(x[x <= u].max() * 1.2)  # el máximo en el tiempo de u_id llevado
    picos = pd.DataFrame({"hs": [2.8, 3.1, 3.5], "id_evento": [0, 1, 2]})
    quedan, res = R.borde_inferior(picos, info)
    assert res["activo"] and res["N0"] == int((picos["hs"] > info["u_id_max_m"]).sum())
    assert (quedan["hs"] > info["u_id_max_m"]).all()


def test_proteccion_de_cola_detiene_con_revision_requerida():
    p = pd.DataFrame({"id_evento": [0, 1], "t_pico": pd.to_datetime(["2000-01-01", "2000-02-01"]),
                      "hs_e0_en_pico": [2.9, 2.6]})  # fmt: skip
    g = R.proteccion_cola(p, 2.70)
    assert g["estado"] == R.REVISION_REQUERIDA and g["picos_bajo_p95"] == 1 and g["lista"][0]["id_evento"] == 1
    assert R.proteccion_cola(p.iloc[:1], 2.70)["estado"] == "cumple"
    # un NaN de E0 en el instante del pico tampoco protege la cola
    assert R.proteccion_cola(p.assign(hs_e0_en_pico=[np.nan, 3.0]), 2.70)["picos_bajo_p95"] == 1


def test_exposicion_hasta_2025_y_e1_desde_febrero():
    idx = pd.date_range("1992-01-01", "2026-08-31 21:00", freq="3h")
    hs = pd.Series(1.0, index=idx)
    e0 = R.exposicion(hs, idx[0], n_picos=100)
    assert e0["anios"] == pytest.approx(34.0, abs=1e-6)  # 1992–2025, sin 2026
    e1 = R.exposicion(hs[hs.index >= INI_E1], INI_E1)
    assert e1["anios"] == pytest.approx(33 - 31 / 365, abs=0.01)  # sin enero de 1993
    lo, hi = e0["ic_garwood_95"]
    assert lo < e0["lambda_por_anio"] < hi
    assert R.exposicion(hs, idx[0], n_picos=100, rama="E0_cuantiles")["nota_ic"] == R.NOTA_GARWOOD_E0
    assert R.exposicion(hs[hs.index >= INI_E1], INI_E1, n_picos=80, rama="E1_sin_corregir")["nota_ic"] is None


def test_banderas_dudoso_y_revision_visual():
    p = pd.DataFrame({"t_inicio": pd.to_datetime(["2000-01-01", "2000-03-01"]),
                      "t_fin": pd.to_datetime(["2000-01-03", "2000-03-02"])})  # fmt: skip
    cand = pd.DataFrame({"tiempo": ["2000-01-02 03:00", "2000-03-01 06:00"],
                         "decision": ["mantener", "mantener"],
                         "motivo": ["clase:dudoso_vecinas_comparten", "clase:evento_sostenido"]})  # fmt: skip
    assert R.marcar_dudosos_f3(p, cand).tolist() == [True, False]
    rev = pd.DataFrame({"tipo": ["casi fusionados"], "t_pico": ["2000-03-01 12:00"],
                        "t_pico_siguiente": ["2000-03-04 00:00"], "clasificacion_visual": ["bien"]})  # fmt: skip
    assert R.marcar_revision_f6(p, rev).tolist() == ["", "casi fusionados: bien"]


def test_empate_numerico_devuelve_el_pico_al_instante_de_e0():
    """Precisión n.º 24 (propuesta): un desempate de la variante menor que 10⁻⁴ relativo no mueve el pico."""
    df = _tormenta(_serie(), "1993-02-10 12:00", 3.5)
    i = df.index.get_loc(pd.Timestamp("1993-02-10 12:00"))
    df.iloc[i + 2, 0] = 3.5  # empate en E0: misma Hs 6 h después; E0 toma la primera aparición
    df.iloc[i + 2, 1] = 99.0  # Tp distinto en el instante del empate
    ev = ID.identificar(df["VHM0"], 2.7, 48.0)
    var = df["VHM0"].copy()
    var.iloc[i + 2] = 3.5 * (1 + 2e-5)  # la corrección por rango desempata hacia el instante posterior
    sin = R.picos_de_rama(ev, var, df)
    assert sin.loc[0, "pico_movido"] and sin.loc[0, "tp"] == 99.0 and not sin.loc[0, "empate_resuelto"]
    con = R.picos_de_rama(ev, var, df, tol_empate=R.TOL_EMPATE)
    assert not con.loc[0, "pico_movido"] and con.loc[0, "empate_resuelto"]
    assert con.loc[0, "t_pico"] == pd.Timestamp("1993-02-10 12:00") and con.loc[0, "tp"] != 99.0
    assert con.loc[0, "hs"] == pytest.approx(sin.loc[0, "hs"], rel=1e-4)
    # una diferencia real (mayor que la tolerancia) sí mueve el pico
    var.iloc[i + 2] = 3.5 * (1 + 1e-3)
    real = R.picos_de_rama(ev, var, df, tol_empate=R.TOL_EMPATE)
    assert real.loc[0, "pico_movido"] and not real.loc[0, "empate_resuelto"]
