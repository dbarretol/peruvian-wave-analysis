"""F7: etapa de ramas y picos por IET (contrato §3.1, §3.2, §3.4 E-29, §3.5 G8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peruvian_wave_analysis.clima import insumos as CI
from peruvian_wave_analysis.eventos import ramas as R
from peruvian_wave_analysis.eventos import ramas_picos as RP

INI_E1 = pd.Timestamp("1993-02-01")


def _datos(factor_var=1.05, mover=None):
    idx = pd.date_range("1992-06-01", "1994-12-31 21:00", freq="3h")
    rng = np.random.default_rng(3)
    hs = np.round(1.2 + 0.4 * rng.random(idx.size), 2)
    for t in ("1992-08-10", "1992-12-01", "1993-06-15", "1993-06-25", "1994-03-03", "1994-07-07"):
        i = idx.get_loc(pd.Timestamp(t))
        hs[i - 2 : i + 3] = [3.0, 3.4, 3.9, 3.3, 2.9]
    serie = pd.DataFrame({"VHM0": hs, "VTPK": 12.0, "VMDR": 200.0, "VPED": 210.0, "VHM0_WW": 0.5, "VHM0_SW1": 1.0,
                          "VHM0_SW2": 0.2}, index=idx)  # fmt: skip
    corr = pd.DataFrame({"VHM0_cuantiles_toda": hs * factor_var, "VHM0_cuantiles_asimilado": hs * factor_var},
                        index=idx)  # fmt: skip
    if mover is not None:  # la variante pone su máximo en un instante con Hs de E0 bajo el P95
        i = idx.get_loc(pd.Timestamp(mover))
        serie.iloc[i + 1, 0] = 2.5  # valle dentro de la ventana (entre dos excedencias), bajo el P95
        corr.iloc[i + 1, 0] = 5.0
        corr.iloc[i + 1, 1] = 5.0
    ramas = CI.armar_ramas(serie, corr, INI_E1)
    return CI.Datos(serie, ramas, INI_E1, pd.DatetimeIndex([]), {}, {})


def test_conteos_por_iet_y_rama_sin_g8():
    d = _datos()
    cand = pd.DataFrame(columns=["tiempo", "decision", "motivo"])
    rev = pd.DataFrame(columns=["tipo", "t_pico", "t_pico_siguiente", "clasificacion_visual"])
    con, picos, var, bajo = RP.ramas_picos(d, 2.7, 2.7, cand, rev, iets=(48.0, 120.0))
    assert set(con["rama"]) == set(R.RAMAS) and set(con["iet_h"]) == {48.0, 120.0}
    c48 = con[con["iet_h"] == 48.0].set_index("rama")
    assert c48.loc["E0_sin_corregir", "eventos"] == 6 and c48.loc["E1_sin_corregir", "eventos"] == 4
    assert c48.loc["E0_cuantiles", "picos"] == 6 and c48.loc["E0_cuantiles", "pico_movido"] == 0
    assert c48.loc["E0_sin_corregir", "nota_ic"] == R.NOTA_GARWOOD_E0 and c48.loc["E1_cuantiles", "nota_ic"] == ""
    assert not bajo and not con["e29_activo"].any()
    assert var["E0_cuantiles"]["estados_discordantes"] == 0
    assert len(picos) == con["picos"].sum()


def test_pico_de_variante_en_un_valle_activa_e29_y_no_llega_a_g8():
    """Un pico de variante en un instante con Hs de E0 ≤ u_id es un estado discordante: E-29 se activa y el pico queda
    bajo u_id^max, así que se descarta antes de G8 (G8 queda como segunda barrera; su lista se prueba en
    ``test_eventos_ramas.py``)."""
    d = _datos(mover="1993-06-15")
    cand = pd.DataFrame(columns=["tiempo", "decision", "motivo"])
    rev = pd.DataFrame(columns=["tipo", "t_pico", "t_pico_siguiente", "clasificacion_visual"])
    con, picos, var, bajo = RP.ramas_picos(d, 2.7, 2.7, cand, rev, iets=(48.0,))
    c = con.set_index("rama")
    assert var["E0_cuantiles"]["estados_discordantes"] >= 1 and var["E0_cuantiles"]["u_id_max_m"] == 5.0
    assert (
        c.loc["E0_cuantiles", "e29_activo"]
        and c.loc["E0_cuantiles", "picos"] < c.loc["E0_cuantiles", "picos_antes_de_e29"]
    )
    assert not c.loc["E0_sin_corregir", "e29_activo"]
    assert not bajo
    assert (picos.loc[picos["rama"] == "E0_cuantiles", "hs"] > 5.0).all()
