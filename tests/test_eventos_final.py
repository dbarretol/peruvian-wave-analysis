"""F7: salidas finales (``eventos-final``; contrato §2, §3.2 punto 4, §3.6 y §4; ADR-0038)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis import manifiesto
from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import final as F
from peruvian_wave_analysis.eventos import identificacion as ID


def _r5(b99=-0.053636223867429056, ic=(-0.0918, 0.0219)):
    return {"correccion": {"variantes": [{"nombre": "sin_corregir", "factor_cola": 1.0},
                                         {"nombre": "cuantiles", "factor_cola": float(np.exp(-b99)),
                                          "B_q": {"P99": b99}}]},
            "fila_fuente": {"ic95_B_q": {"P99": list(ic)}}}  # fmt: skip


def test_factor_de_hs_con_su_intervalo_y_solo_hs():
    f = F.factor_hs(_r5())
    assert f["valor"] == pytest.approx(1.0551, abs=1e-4)
    assert f["intervalo_95"] == pytest.approx([np.exp(-0.0219), np.exp(0.0918)])
    assert f["intervalo_95"][0] < 1.0 < f["intervalo_95"][1] and f["significativo"] is False
    assert f["corrige"] == "Hs" and f["tp"] == "no se corrige"


def test_factor_significativo_si_el_intervalo_no_incluye_1():
    assert F.factor_hs(_r5(ic=(-0.09, -0.01)))["significativo"] is True


def test_factor_incoherente_con_b_falla():
    r = _r5()
    r["correccion"]["variantes"][1]["factor_cola"] = 1.06
    with pytest.raises(ValueError):
        F.factor_hs(r)


def test_comprobar_factor():
    hs = np.array([3.5, 4.0, 4.6])
    f = {"sin_corregir": 1.0, "cuantiles": 1.0551}
    p = pd.concat([pd.DataFrame({"rama": f"{e}_{v}", "id_evento": [1, 2, 3], "hs": hs * f[v]})
                   for e in ("E0", "E1") for v in f])  # fmt: skip
    u = {"E0_sin_corregir": 3.6, "E1_sin_corregir": 3.6}
    d = F.comprobar_factor(p, 1.0551, u)
    assert d["E0"]["picos"] == 2 and max(v["desviacion_relativa_maxima"] for v in d.values()) < 1e-12
    p.loc[(p["rama"] == "E1_cuantiles") & (p["id_evento"] == 1), "hs"] = 3.5  # bajo u*: no se comprueba
    assert F.comprobar_factor(p, 1.0551, u)["E1"]["desviacion_relativa_maxima"] < 1e-12
    p.loc[(p["rama"] == "E1_cuantiles") & (p["id_evento"] == 2), "hs"] = 4.0
    assert F.comprobar_factor(p, 1.0551, u)["E1"]["desviacion_relativa_maxima"] > 0.05


def _serie():
    idx = pd.date_range("1992-12-01", "1993-03-31 21:00", freq="3h")
    hs = np.full(idx.size, 1.5)
    for i, n in ((40, 6), (500, 90), (800, 4)):  # el segundo dura 270 h (> 10 días)
        hs[i : i + n] = 3.0
        hs[i + n // 2] = 3.6
    return pd.DataFrame({"VHM0": hs, "VTPK": 12.0, "VMDR": 200.0, "VPED": 210.0, "VHM0_WW": 0.3, "VHM0_SW1": hs * 0.9,
                         "VHM0_SW2": 0.2}, index=idx)  # fmt: skip


def test_tabla_de_eventos_asigna_eras_y_marca_picos_y_duracion():
    s = _serie()
    ev = ID.identificar(s["VHM0"], 2.7, 48.0)
    assert len(ev) == 3
    picos = {
        "E1_sin_corregir": pd.DataFrame({"id_evento": [2]}),
        "E0_sin_corregir": pd.DataFrame({"id_evento": [0, 2]}),
    }
    t = F.tabla_eventos(ev, s, pd.Timestamp("1993-02-01"), picos, fin=pd.Timestamp("1993-03-01"))
    assert list(t["hs_pico"]) == [3.6, 3.6, 3.6] and (t["tp"] == 12.0).all()
    assert list(t["en_E0"]) == [True, True, False]  # el tercero cae después del fin
    assert list(t["en_E1"]) == [False, True, False]  # el primero, antes del inicio de E1
    assert list(t["sobre_u_E1_sin_corregir"]) == [False, False, True]
    assert list(t["sobre_u_E0_sin_corregir"]) == [True, False, True]
    assert list(t["duracion_larga"]) == [False, True, False]
    a = F.alerta_rachas_largas(t)
    assert a["toda_la_serie"]["largos"] == 1 and a["marca"] is True  # 1 de 3 > 5 %
    assert a["E1"] == {"eventos": 1, "largos": 1, "fraccion": 1.0}


def test_anidados():
    b = pd.DataFrame({"rama": ["a"] * 3 + ["b"] * 3, "u_m": [3.0, 3.2, 3.4] * 2, "n": [50, 40, 30, 50, 41, 42]})
    assert F.anidados(b) == {"a": True, "b": False}


def test_comprobar_factor_sin_picos_sobre_u_da_none():
    p = pd.DataFrame({"rama": [f"{e}_{v}" for e in ("E0", "E1") for v in ("sin_corregir", "cuantiles")],
                      "id_evento": 1, "hs": 3.0})  # fmt: skip
    d = F.comprobar_factor(p, 1.0551, {"E0_sin_corregir": 3.6, "E1_sin_corregir": 3.6})
    assert d["E0"] == {"picos": 0, "desviacion_relativa_maxima": None}


def test_alerta_marca_si_una_era_pasa_del_5_por_ciento():
    """F07-R-06: E1 con 1 largo de 10 marca aunque toda la serie no pase del 5 %."""
    n = 40
    ev = pd.DataFrame({"en_E0": True, "en_E1": np.arange(n) < 10, "duracion_larga": np.arange(n) == 0})
    a = F.alerta_rachas_largas(ev)
    assert a["toda_la_serie"]["fraccion"] == pytest.approx(1 / 40) and a["E1"]["fraccion"] == pytest.approx(0.1)
    assert a["marca"] is True
    assert F.alerta_rachas_largas(ev.assign(duracion_larga=False))["marca"] is False


def test_cola_del_factor_repite_la_seleccion_y_los_pesos_de_la_f5():
    t = pd.date_range("1990-01-01", "1995-12-31 21:00", freq="3h")
    hs = np.random.default_rng(3).gamma(4.0, 0.4, t.size)
    b_q = {"P50": -0.01, "P90": -0.03, "P99": -0.05, "P99.9": None}
    u = {"E0_sin_corregir": 3.5, "E0_cuantiles": 3.5 * 1.05, "E1_sin_corregir": 3.5, "E1_cuantiles": 3.0}
    c = F.cola_del_factor(t, hs, b_q, u, 1.05)
    assert c["q_max"] == 0.99
    e1 = t >= pd.Timestamp("1993-01-01")
    for e, m in (("E0", np.ones(t.size, bool)), ("E1", e1)):
        h = np.quantile(hs[m], 0.99)  # pesos mensuales casi iguales en años completos
        assert c[f"{e}_sin_corregir"]["umbral_cola_m"] == pytest.approx(h, rel=0.02)
        assert c[f"{e}_cuantiles"]["umbral_cola_m"] == pytest.approx(c[f"{e}_sin_corregir"]["umbral_cola_m"] * 1.05)
    u_e1 = c["E1_sin_corregir"]["umbral_cola_m"]
    assert c["E1_sin_corregir"]["cumple"] is (3.5 >= u_e1) and c["E1_cuantiles"]["cumple"] is (3.0 >= u_e1 * 1.05)
    bajo = F.cola_del_factor(t, hs, b_q, {k: 0.5 for k in u}, 1.05)
    assert not any(bajo[k]["cumple"] for k in u)


def test_comprobar_eventos_g2():
    s = _serie()
    ev = ID.identificar(s["VHM0"], 2.7, 48.0)
    g = F.comprobar_eventos(ev, s["VHM0"], 2.7, 48.0)
    assert g["cumple"] and g["eventos"] == 3 and g["separacion_minima_entre_picos_h"] >= 48.0
    assert F.comprobar_eventos(ev, s["VHM0"], 2.7, 1e4)["picos_a_iet_o_mas"] is False
    assert F.comprobar_eventos(ev, s["VHM0"], 3.6, 48.0)["picos_sobre_u_id"] is False
    solape = ev.copy()
    solape.loc[1, "t_inicio"] = solape.loc[0, "t_fin"]
    assert F.comprobar_eventos(solape, s["VHM0"], 2.7, 48.0)["cumple"] is False


def _gates(**cambios):
    ramas = ("E0_sin_corregir", "E0_cuantiles", "E1_sin_corregir", "E1_cuantiles")
    base = {
        "g1": {"cumple": True},
        "g2": {"cumple": True},
        "g3": dict.fromkeys(ramas, True),
        "largas": {k: {"fraccion": 0.0} for k in ("toda_la_serie", "E0", "E1")} | {"marca": False},
        "umb": {"ramas": {r: {"g4": True, "R1b": {"estado": "con_resultado", "n": 41}} for r in ramas},
                "g8": {"picos_bajo_p95_f3": []}},
        "n_picos": dict.fromkeys(ramas, 41),
        "j_iet": {"iet_sensible": False},
        "bajo": [],
        "cola": {"q_max": 0.99, **{r: {"cumple": True} for r in ramas}},
    }  # fmt: skip
    return F.evaluar_gates(**(base | cambios))


def test_gates_calculados_y_g9_g10_pendientes():
    g, m = _gates()
    assert m == [] and g["G9"]["cumple"] == "pendiente" and g["G10"]["cumple"] == "pendiente"
    assert all(v["cumple"] is True for k, v in g.items() if k not in ("G9", "G10"))


def test_iet_sensible_es_marca_con_aviso_y_no_falla_g7():
    """F07-R-01: el contrato acepta el IET* con la marca «IET sensible» y el aviso."""
    g, m = _gates(j_iet={"iet_sensible": True})
    assert m == [] and g["G7"]["cumple"] is True and g["G7"]["aviso"]


@pytest.mark.parametrize(
    ("cambio", "gate"),
    [
        ({"g1": {"cumple": False}}, "G1"),
        ({"g2": {"cumple": False}}, "G2"),
        ({"g3": {"E0_sin_corregir": False}}, "G3"),
        ({"n_picos": {"E0_sin_corregir": 29}}, "G5"),
        ({"n_picos": {"E0_sin_corregir": 40}}, "G5"),  # no coincide con umbral.json
        ({"bajo": [{"rama": "E0_cuantiles"}]}, "G8"),
        ({"cola": {"E1_sin_corregir": {"cumple": False}}}, "premisa_factor_hs"),
        ({"cola": {"q_max": 0.99}}, "premisa_factor_hs"),  # sin ramas comprobadas
    ],
)
def test_gate_forzado_a_fallar_da_su_motivo(cambio, gate):
    """F07-R-08: cada gate que falla deja su motivo, y con motivos la etapa no publica (etapa.Gate)."""
    _, m = _gates(**cambio)
    assert m == [f"{gate} no se cumple"]


def test_verificar_insumos_contra_el_manifiesto(tmp_path):
    d = tmp_path / "processed" / "x"
    d.mkdir(parents=True)
    (d / "a.csv").write_text("1\n", encoding="utf-8")
    (d / "b.csv").write_text("2\n", encoding="utf-8")
    json_estricto.escribir(d / etapa.MANIFIESTO, {"salidas": {"a.csv": manifiesto.sha256(d / "a.csv"),
                                                              "b.csv": "otro"}})  # fmt: skip
    (tmp_path / "suelto.csv").write_text("3\n", encoding="utf-8")
    ins = {"a": d / "a.csv", "b": d / "b.csv", "c": tmp_path / "suelto.csv"}
    g = F.verificar_insumos(ins, etapa.exigir_insumos(ins), tmp_path)
    assert g["cumple"] is False and g["verificados_contra_manifiesto"] == 1
    assert g["hash_distinto"] == ["b"] and g["sin_manifiesto"] == ["c"]
    ok = {"a": d / "a.csv"}
    assert F.verificar_insumos(ok, etapa.exigir_insumos(ok), tmp_path)["cumple"] is True


def test_iet_siguiente_none_si_no_hay_vecino():
    """F07-R-04: con IET* = 120 h, iet.json deja el siguiente en None."""
    assert F._o_none(None) is None and F._o_none(54) == 54.0


def test_figura_rl100_y_picos(tmp_path):
    u = np.linspace(2.75, 3.65, 6)
    b = pd.concat([pd.DataFrame({"rama": r, "u_m": u, "n": [400, 300, 150, 70, 40, 30], "rl100": 4.8,
                                 "rl100_lo": 4.45, "rl100_hi": [5.0, 5.3, 5.6, 6.1, 7.6, 8.6]})
                   for r in F.RAMAS_F8])  # fmt: skip
    umb = {"ramas": {r: {"R1b": {"u_m": 3.395}, "R2": {"u_m": 2.755}, "R3": {"u_m": 2.885}} for r in F.RAMAS_F8}}
    ruta = tmp_path / "figuras" / F.FIGURA_RL100_PICOS
    cubre = F.figura_rl100_y_picos(b, umb, ruta)
    assert ruta.is_file() and ruta.stat().st_size > 0
    assert cubre == dict.fromkeys(F.RAMAS_F8, True)
