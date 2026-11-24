"""F7: u* por rama en los datos (contrato §3.4, §3.6; ADR-0037; §8 «Rejilla de picos: misma función en la simulación
y en los datos»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from peruvian_wave_analysis.eventos import sesgo_regla as SR
from peruvian_wave_analysis.eventos import simulacion_previa as SP
from peruvian_wave_analysis.eventos import umbral as U
from peruvian_wave_analysis.eventos import umbral_datos as UD

REF = {"comparadas": {"R2": {"d_rl": [-0.096, 0.164], "delta_u_m": [-0.98, 0.19]},
                      "R3": {"d_rl": [-0.055, 0.142], "delta_u_m": [-0.87, 0.51]}}}  # fmt: skip
NULO = [-0.274, 0.463]


def _picos(n=480, xi=-0.1, sigma=0.35, u0=2.7, seed=5):
    x = stats.genpareto.rvs(xi, scale=sigma, size=n, random_state=np.random.default_rng(seed))
    return np.round(u0 + x, 2)


def test_mismas_reglas_que_la_simulacion_previa():
    picos = _picos()
    tabla, res = UD.analizar_rama("E0_sin_corregir", picos, 46.0, REF, NULO, con_ic=False)
    rej = U.rejilla(picos)
    aj = U.barrido(picos, rej, 46.0)
    ref = SP.aplicar_reglas(picos, rej, aj, ("R1b", "R2", "R3"), UD.semilla_r3("E0_sin_corregir"))
    for regla in ("R1b", "R2", "R3"):
        assert res[regla]["posicion"] == ref[regla].indice and res[regla]["estado"] == ref[regla].estado
    assert len(tabla) == rej.u.size and tabla["u_m"].is_monotonic_increasing
    assert tabla["n"].is_monotonic_decreasing  # G3: picos anidados


def test_marcas_y_sensibilidad_en_una_gpd():
    picos = _picos(seed=9)
    tabla, res = UD.analizar_rama("E1_sin_corregir", picos, 33.0, REF, NULO, con_ic=False)
    assert res["R1b"]["estado"] in (U.CON_RESULTADO, U.NO_DISCRIMINO)
    assert res["g5"] and res["g4"]
    for r in ("R2", "R3"):
        s = res["sensibilidad"][r]
        if s["comparable"]:
            assert s["lectura_d_rl"] in ("dentro", "bajo", "sobre")
            i = res["R1b"]["posicion"]
            assert s["d_rl"] == pytest.approx(res[r]["rl100"] / tabla.loc[i, "rl100"] - 1)
        else:
            assert s["lectura"] == SR.NO_COMPARABLE
    assert len(res["r3_semillas_descriptivas"]) == UD.SEMILLAS_R3_DESCRIPTIVAS
    assert {"pendiente", "intervalo_nulo_90", "marca", "nota", "n_en_u"} <= set(res["deriva_xi"])
    assert (res["deriva_xi"]["nota"] is None) == (not res["deriva_xi"]["marca"])
    tramos = res["tramos_estables"]
    assert sum(t["es_el_de_r1b"] for t in tramos) == 1
    assert all(t["rl100_min"] <= t["rl100_max"] for t in tramos)


def test_nota_de_deriva_con_los_numeros():
    t = UD.NOTA_DERIVA.format(n=41, a=73, b=176, lo=-0.274, hi=0.463)
    assert "41 picos" in t and "de 73 a 176" in t and "[-0.274, 0.463]" in t and "no cambia u*" in t


def test_relevancia_a_mano():
    rl = np.array([4.60, 4.70, 4.80, 4.90])
    r = UD.relevancia(rl, 1)
    assert r["maximo"] == pytest.approx(0.1 / 4.7) and r["punto_sensible"]
    assert not UD.relevancia(np.array([4.70, 4.71, 4.72]), 1)["punto_sensible"]


def test_tabla_de_picos_solo_sobre_u_y_con_exceso():
    p = pd.DataFrame({c: [0, 0, 0] for c in ("id_evento", "id_racha", "t_pico", "t_inicio", "t_fin", "n_excedencias",
                                             "tp", "vmdr", "vped", "particion_dominante", "estrato", "toca_hueco",
                                             "dividido_por_bajada", "pico_movido", "empate_resuelto", "dudoso_f3",
                                             "revision_visual_f6")} | {"hs": [3.0, 3.2, 3.5]})  # fmt: skip
    t = UD.tabla_picos(p, 3.1)
    assert t["hs"].tolist() == [3.2, 3.5] and t["exceso"].tolist() == pytest.approx([0.1, 0.4])


def test_ic_de_perfil_en_el_barrido():
    picos = _picos(n=300, seed=3)
    tabla, res = UD.analizar_rama("E0_cuantiles", picos, 46.0, REF, NULO, con_ic=True)
    i = res["R1b"]["posicion"]
    assert res["R1b"]["rl100_lo"] == pytest.approx(tabla.loc[i, "rl100_lo"])  # IC del RL100 en u* publicado
    assert res["R1b"]["rl100_lo"] < res["R1b"]["rl100"] < res["R1b"]["rl100_hi"]
    ok = tabla["valido"] & tabla["xi_lo"].notna() & tabla["rl100_lo"].notna()
    assert ok.sum() >= 15
    assert (tabla.loc[ok, "xi_lo"] < tabla.loc[ok, "xi"]).all()
    assert (tabla.loc[ok, "rl100_lo"] < tabla.loc[ok, "rl100"]).all() and (
        tabla.loc[ok, "rl100"] < tabla.loc[ok, "rl100_hi"]
    ).all()
