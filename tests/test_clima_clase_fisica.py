"""Pruebas de la clasificación física por particiones (contrato de la F6, §3.2 y §8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.clima import clase_fisica as CF


def _df(ww, sw1, sw2):
    return pd.DataFrame({"VHM0_WW": ww, "VHM0_SW1": sw1, "VHM0_SW2": sw2})


def test_r_ww_conocidos_y_bordes():
    # r_WW = 0, 1/3, 2/3, 1 (con SW1² = 2 WW² para 1/3; WW² = 2 SW1² para 2/3)
    df = _df([0.0, 1.0, np.sqrt(2), 1.0], [1.0, np.sqrt(2), 1.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    cf = CF.clase_fisica(df)
    assert cf["r_WW"].to_numpy() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert cf["clase"].tolist() == [CF.SWELL, CF.MIXTO, CF.MIXTO, CF.VIENTO]  # los bordes son «mixto»
    assert cf["clase_sens"].tolist() == [CF.SWELL, CF.MIXTO, CF.MIXTO, CF.VIENTO]


def test_cortes_de_sensibilidad():
    r = pd.Series([0.1, 0.25, 0.5, 0.75, 0.9])
    assert CF.clasificar(r).tolist() == [CF.SWELL, CF.SWELL, CF.MIXTO, CF.VIENTO, CF.VIENTO]
    assert CF.clasificar(r, CF.CORTES_SENSIBILIDAD).tolist() == [CF.SWELL, CF.MIXTO, CF.MIXTO, CF.MIXTO, CF.VIENTO]


def test_particion_ausente_aporta_cero_y_sin_particiones():
    df = _df([1.0, 0.0, np.nan], [1.0, 0.0, np.nan], [5.0, 0.0, np.nan])
    ausente = {"SW2": pd.Series([True, False, False])}  # la F3 la marca ausente: no aporta aunque tenga altura
    cf = CF.clase_fisica(df, ausente)
    assert cf.loc[0, "r_WW"] == pytest.approx(0.5) and cf.loc[0, "r_SW2"] == 0.0
    assert cf.loc[1, "clase"] == CF.SIN_CLASE and cf.loc[2, "clase"] == CF.SIN_CLASE
    g2 = CF.comprobar_g2(cf)
    assert g2["cumple"] and g2["conteo_clases"][CF.SIN_CLASE] == 2


def test_denominador_es_la_suma_de_particiones_no_hs_total():
    # con discrepancia de energía (Hs total distinto de la suma), la fracción usa la suma de las particiones
    df = _df([1.0], [1.0], [0.0]).assign(VHM0=[3.0])
    assert CF.fracciones(df)["r_WW"].iloc[0] == pytest.approx(0.5)


def test_incoherencia_fisica():
    assert CF.incoherencia_fisica({CF.VIENTO: 11.0, CF.SWELL: 12.0}) == {CF.VIENTO: True, CF.SWELL: False}
    assert CF.incoherencia_fisica({CF.VIENTO: 6.0, CF.SWELL: 7.5}) == {CF.VIENTO: False, CF.SWELL: True}
