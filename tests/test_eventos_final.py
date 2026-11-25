"""F7: salidas finales (``eventos-final``; contrato §2, §3.2 punto 4, §3.6 y §4; ADR-0038)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
