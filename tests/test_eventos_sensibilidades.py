"""F7: sensibilidades de §3.7 (regla de Tp, E-06 y E-28; u* e IET* fijos, precisión n.º 28)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.clima import insumos as CI
from peruvian_wave_analysis.eventos import identificacion as ID
from peruvian_wave_analysis.eventos import ramas as R
from peruvian_wave_analysis.eventos import sensibilidades as S

# dos crestas a 30 h (10 pasos), mínimo 0.5 m bajo la menor; Tp sube 4 s y VTM01_SW1 2.5 s del mínimo a la 2.ª
HS = np.array([3.0, 3.4, 3.6, 3.3, 3.1, 3.0, 2.9, 3.0, 3.2, 3.5, 3.7, 3.8, 3.4, 3.0])
TP = np.array([12.0] * 7 + [14.0, 15.0, 16.0, 16.0, 16.0, 16.0, 16.0])
TM_SW1 = np.array([10.0] * 7 + [11.0, 12.0, 12.5, 12.5, 12.5, 12.5, 12.5])
TM_SW2 = np.full(HS.size, 8.0)
DOM = np.array(["SW1"] * HS.size, dtype=object)


def _corte(hs=HS, tp=TP, sw1=TM_SW1, dom=DOM):
    return S.corte_tp(hs, tp, {"SW1": sw1, "SW2": np.full(len(hs), 8.0)}, dom, 0, len(hs) - 1)


def test_corte_tp_cumple_las_cuatro_condiciones():
    c = _corte()
    assert c is not None and c[0] == 6 and c[1] == pytest.approx(3.6 - 2.9)


@pytest.mark.parametrize(
    "cambio",
    [
        "tp_sube_poco",
        "domina_viento",
        "tm01_sube_poco",
        "crestas_cerca",
        "bajada_corta",
        "tp_nan",
    ],
)
def test_corte_tp_falla_si_falta_una_condicion(cambio):
    hs, tp, sw1, dom = HS.copy(), TP.copy(), TM_SW1.copy(), DOM.copy()
    if cambio == "tp_sube_poco":
        tp[7:] = 14.5  # 2.5 s
    elif cambio == "domina_viento":
        dom[11] = "WW"
    elif cambio == "tm01_sube_poco":
        sw1[7:] = 11.5  # 1.5 s
    elif cambio == "crestas_cerca":
        hs = np.array([3.0, 3.6, 3.3, 3.0, 2.9, 3.0, 3.4, 3.8, 3.4, 3.0])  # crestas a 6 pasos (18 h)
        tp, sw1, dom = tp[:10], sw1[:10], dom[:10]
        tp[5:] = 16.0
        sw1[5:] = 12.5
    elif cambio == "bajada_corta":
        hs[3:9] = [3.45, 3.4, 3.38, 3.35, 3.4, 3.45]  # mínimo 0.25 m bajo la cresta menor
    elif cambio == "tp_nan":
        tp[11] = np.nan
    assert _corte(hs, tp, sw1, dom) is None


def test_comprobacion_frente_a_la_vecina():
    """Mismo procedimiento que la bajada: la subparte junto a una vecina ya cortada cumple las condiciones frente a
    ella. Con la cresta vecina a 4 pasos (12 h) de la cresta izquierda, el corte no vale; a 10 pasos, sí."""
    pre = 12
    hs = np.concatenate([np.full(pre, 2.0), HS])
    tp, sw1 = np.concatenate([np.full(pre, 8.0), TP]), np.concatenate([np.full(pre, 8.0), TM_SW1])
    dom = np.array(["SW1"] * hs.size, dtype=object)
    tm = {"SW1": sw1, "SW2": np.full(hs.size, 8.0)}
    hs[pre - 1] = 2.5  # corte compartido con la vecina
    for vecina, vale in ((pre + 2 - 4, False), (pre + 2 - 10, True)):
        hs[vecina] = 3.9
        c = S.corte_tp(hs, tp, tm, dom, pre, hs.size - 1, izquierda=vecina)
        assert (c is not None) is vale
        hs[vecina] = 2.0


def test_un_salto_de_tp_por_cambio_de_particion_no_parte():
    """E-28: si en la segunda cresta domina SW2 y su VTM01 no sube, no cuenta aunque VTPK salte."""
    dom = DOM.copy()
    dom[11] = "SW2"
    assert _corte(dom=dom) is None


def _serie(n_eventos=3):
    idx = pd.date_range("2020-01-01", periods=400, freq="3h")
    hs = np.full(idx.size, 1.5)
    tp = np.full(idx.size, 12.0)
    sw1 = np.full(idx.size, 10.0)
    for k in range(n_eventos):
        s = 40 + 100 * k
        hs[s : s + HS.size] = HS
        tp[s : s + HS.size] = TP
        sw1[s : s + HS.size] = TM_SW1
    return pd.DataFrame({"VHM0": hs, "VTPK": tp, "VTM01_SW1": sw1, "VTM01_SW2": 8.0, "VHM0_WW": 0.3,
                         "VHM0_SW1": hs * 0.9, "VHM0_SW2": 0.2, "VMDR": 200.0, "VPED": 210.0}, index=idx)  # fmt: skip


def test_aplicar_regla_tp_parte_cada_evento_y_conserva_la_racha():
    serie = _serie()
    ev = ID.identificar(serie["VHM0"], 2.7, 48.0)
    assert len(ev) == 3 and not ev["dividido_por_bajada"].any()  # la bajada (0.5 m) tampoco parte: 0.7 m, ≥ 48 h no
    nuevo = S.aplicar_regla_tp(ev, serie, 2.7)
    assert len(nuevo) == 6 and nuevo["partido_por_tp"].all()
    assert (nuevo.groupby("id_evento_original")["id_racha"].nunique() == 1).all()
    assert list(nuevo["id_evento"]) == list(range(6))
    # bordes C-03: la primera parte termina en el mínimo, la segunda empieza en el paso siguiente
    p = nuevo[nuevo["id_evento_original"] == 0]
    assert p["i_fin"].iloc[0] + 1 == p["i_inicio"].iloc[1]
    assert (nuevo["i_pico"] >= nuevo["i_inicio"]).all() and (nuevo["i_pico"] <= nuevo["i_fin"]).all()


def test_dudosos_con_rango_en_la_cola():
    p = pd.DataFrame({"t_pico": pd.date_range("2020-01-01", periods=5, freq="10D"),
                      "hs": [3.1, 4.0, 3.6, 2.9, 3.8], "dudoso_f3": [False, False, True, True, False]})  # fmt: skip
    d = S.dudosos(p, "E0_sin_corregir", 3.0)
    assert list(d["hs"]) == [3.6] and list(d["rango"]) == [3] and list(d["picos_sobre_u"]) == [4]


# --- Integración sobre datos sintéticos -----------------------------------------------------------

INI_E1 = pd.Timestamp("2021-02-01")


def _datos(seed=0):
    idx = pd.date_range("2016-01-01", "2026-08-31 21:00", freq="3h")
    rng = np.random.default_rng(seed)
    hs = np.round(1.2 + 0.4 * rng.random(idx.size), 2)
    tp = np.full(idx.size, 12.0)
    sw1 = np.full(idx.size, 10.0)
    pos = np.sort(rng.choice(np.arange(20, idx.size - 20, 60), 230, replace=False))
    for j, i in enumerate(pos):
        pico = np.round(2.9 + rng.exponential(0.45), 2)
        if j % 4 == 0:  # tren nuevo: dos crestas a 30 h con subida de Tp
            hs[i - 2 : i - 2 + HS.size] = np.round(HS - 3.8 + pico, 2)
            tp[i - 2 : i - 2 + HS.size] = TP
            sw1[i - 2 : i - 2 + HS.size] = TM_SW1
        else:
            hs[i - 2 : i + 3] = [2.8, pico - 0.2, pico, pico - 0.25, 2.75]
    serie = pd.DataFrame({"VHM0": hs, "VTPK": tp, "VMDR": 200.0, "VPED": 210.0, "VHM0_WW": 0.3, "VHM0_SW1": hs * 0.9,
                          "VHM0_SW2": 0.2, "VTM01_SW1": sw1, "VTM01_SW2": 8.0}, index=idx)  # fmt: skip
    corr = pd.DataFrame({"VHM0_cuantiles_toda": hs * 1.05, "VHM0_cuantiles_asimilado": hs * 1.05}, index=idx)
    return CI.Datos(serie, CI.armar_ramas(serie, corr, INI_E1), INI_E1, pd.DatetimeIndex([]), {}, {})


@pytest.fixture(scope="module")
def corrida():
    cand = pd.DataFrame(columns=["tiempo", "decision", "motivo"])
    rev = pd.DataFrame(columns=["tipo", "t_pico", "t_pico_siguiente", "clasificacion_visual"])
    u = {"E0_sin_corregir": 3.205, "E1_sin_corregir": 3.205, "E0_cuantiles": 3.205 * 1.05,
         "E1_cuantiles": 3.205 * 1.05}  # fmt: skip
    return S.calcular(_datos(), 2.7, 2.7, u, 48.0, 54.0, cand, rev)


def test_tabla_de_sensibilidades(corrida):
    tabla, dud, tp = corrida
    assert len(tabla) == len(S.SENSIBILIDADES) * len(R.RAMAS)
    t = tabla.set_index(["sensibilidad", "rama"])
    assert (t.xs("principal")["d_rl100"] == 0).all()
    for r in R.RAMAS:
        # con 2026 hay más exposición y no menos picos
        assert t.loc[("con_2026", r), "anios"] > t.loc[("principal", r), "anios"]
        assert t.loc[("con_2026", r), "picos"] >= t.loc[("principal", r), "picos"]
        # sin bajada no puede haber más eventos; con la regla de Tp, no menos
        assert t.loc[("sin_bajada", r), "eventos"] <= t.loc[("principal", r), "eventos"]
        assert t.loc[("regla_tp", r), "eventos"] >= t.loc[("principal", r), "eventos"]
        assert t.loc[("regla_tp", r), "eventos_partidos_por_tp"] > 0
        assert np.isfinite(t.loc[("principal", r), "rl100"])
    # la variante por cuantiles es la sin corregir × 1.05 con u* × 1.05: mismo número de picos sobre u*
    for s in S.SENSIBILIDADES:
        for e in ("E0", "E1"):
            assert t.loc[(s, f"{e}_cuantiles"), "n_sobre_u"] == t.loc[(s, f"{e}_sin_corregir"), "n_sobre_u"]
    assert tp["eventos_despues"] - tp["eventos_antes"] >= tp["eventos_partidos"] > 0
    assert dud.empty
