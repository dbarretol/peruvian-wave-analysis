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


def test_tormenta_en_el_borde_de_la_serie():
    hs = np.zeros(10)
    esc.poner_tormenta(hs, 7, 12, np.random.default_rng(0))  # se corta en la última celda
    assert (hs[:7] == 0).all() and (hs[7:] >= 1.0).all()
    esc.poner_tormenta(hs, 12, 5, np.random.default_rng(0))  # empieza fuera: no hace nada
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    esc.poner_tormenta(np.zeros(3), 0, 12, a)
    esc.poner_tormenta(np.zeros(30), 0, 12, b)
    assert a.random() == b.random()  # el corte no cambia cuántos números consume


def test_corrida_no_canonica_no_toca_la_carpeta_canonica(tmp_path):
    datos = tmp_path / "WAVE-DATA"
    datos.mkdir()
    canon = run.ruta_canonica(datos)
    canon.mkdir(parents=True)
    (canon / "MANIFEST.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exige --salida"):
        run.ejecutar(datos, 1, r=2, solo=["A2"])
    with pytest.raises(ValueError, match="carpeta canónica"):
        run.ejecutar(datos, 1, r=2, solo=["A2"], salida=canon)
    with pytest.raises(ValueError, match="carpeta canónica"):
        run.ejecutar(datos, 1, solo=["A2"], salida=canon)
    assert sorted(p.name for p in canon.iterdir()) == ["MANIFEST.json"]  # intacta: no se abrió el staging
    assert run.comprobar_salida(datos, True, None) == canon


def test_la_corrida_canonica_exige_las_21_bandas():
    assert run.motivos_bandas(21, canonica=True) == []
    assert run.motivos_bandas(20, canonica=True)
    assert run.motivos_bandas(3, canonica=False) == []
    n = sum(3 + (e.tipo == "rl") if e.tipo.startswith("rl") else 1 for e in esc.ESCENARIOS if e.con_banda)
    assert n == run.N_BANDAS


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
