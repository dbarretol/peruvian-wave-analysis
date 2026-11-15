"""Etapa de verificación: bandas, conteo de la cobertura y determinismo con 1 y 2 procesos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa
from peruvian_wave_analysis.verificacion import escenarios as esc
from peruvian_wave_analysis.verificacion import run


def test_bandas_del_contrato():
    assert run.banda(0.95, 400) == pytest.approx((0.9173, 0.9827), abs=1e-4)
    assert run.banda(0.025, 400) == pytest.approx((0.0016, 0.0484), abs=1e-4)
    assert run.banda(0.05, 400) == pytest.approx((0.0173, 0.0827), abs=1e-4)
    assert run.banda(0.95, 1600) == pytest.approx((0.9337, 0.9663), abs=1e-4)
    assert run.banda(0.025, 1600) == pytest.approx((0.0133, 0.0367), abs=1e-4)
    assert [e.id for e in esc.ESCENARIOS if e.con_banda] == ["A1", "A2", "A3", "A4", "B", "D1", "D2"]


def test_conteo_de_cobertura_por_causa():
    assert esc._cubre(5.0, 4.0, np.nan, None, "tope") == (True, True)
    assert esc._cubre(5.0, np.nan, 6.0, "frontera", None) == (True, True)
    assert esc._cubre(5.0, 4.0, np.nan, None, "sin_convergencia") == (True, False)
    assert esc._cubre(5.0, np.nan, 6.0, "borde_de_factibilidad", None) == (False, True)
    assert esc._cubre(7.0, 4.0, 6.0, None, None) == (True, False)


def test_evaluar_cuenta_ajustes_fallidos_como_no_cobertura():
    df = pd.DataFrame(
        {
            "replica": range(4),
            "ajuste_fallido": [False, False, False, True],
            "rl_cubre_abajo": [True, True, False, None],
            "rl_cubre_arriba": [True, False, True, None],
            "rl_causa_lo": [None] * 4,
            "rl_causa_hi": [None, None, None, None],
            "xi_cubre_abajo": [True] * 3 + [None],
            "xi_cubre_arriba": [True] * 3 + [None],
            "xi_no_regular": [False] * 4,
        }
    )
    ev = run.evaluar(esc.POR_ID["A3"], df)
    assert ev["metricas"]["cobertura_rl"] == 0.25
    assert ev["metricas"]["falla_rl_arriba"] == 0.5 and ev["metricas"]["falla_rl_abajo"] == 0.5
    assert ev["metricas"]["ajustes_fallidos"] == 1 and len(ev["bandas"]) == 4


def test_serie_continua_sin_bajadas_internas():
    df = esc.serie_continua({"tasas_anio": (30.0, 12.0), "frac_replica": 0.0}, np.random.default_rng(0))
    assert df["Hs"].isna().sum() == 0 and set(np.unique(df["Hs"] > 0.5)) <= {True, False}
    assert ((df["Hs"] == 0) | (df["Hs"] >= 1.0)).all()


def test_determinismo_con_uno_y_dos_procesos(tmp_path):
    datos = tmp_path / "WAVE-DATA"
    datos.mkdir()
    hashes = []
    for procesos in (1, 2):
        salida = datos / "processed" / f"v{procesos}"
        run.ejecutar(datos, procesos, r=2, solo=["A2", "D2", "BS2"], salida=salida)
        etapa.exigir_publicada(salida)
        hashes.append((salida / etapa.MANIFIESTO).read_bytes())
    assert hashes[0] == hashes[1]
