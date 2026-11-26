"""F7: etapa de diagnósticos de §3.5 sobre datos sintéticos (contrato §3.3 P11, §3.5, §3.7 punto 6; G9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.clima import insumos as CI
from peruvian_wave_analysis.eventos import diagnosticos as D
from peruvian_wave_analysis.eventos import diagnosticos_datos as DD
from peruvian_wave_analysis.eventos import ramas as R

INI_E1 = pd.Timestamp("1993-02-01")
B = {"poisson": 9, "a2": 9, "perm": 9, "disp": 9, "tamano": 9, "replicas_tamano": 2}
U = {"E0_sin_corregir": 3.205, "E1_sin_corregir": 3.205, "E0_cuantiles": 3.205 * 1.05, "E1_cuantiles": 3.205 * 1.05}


def _datos(seed=0):
    idx = pd.date_range("1988-01-01", "1997-12-31 21:00", freq="3h")
    rng = np.random.default_rng(seed)
    hs = np.round(1.2 + 0.4 * rng.random(idx.size), 2)
    pos = np.sort(rng.choice(np.arange(10, idx.size - 10, 40), 260, replace=False))
    for i in pos:
        pico = np.round(2.9 + rng.exponential(0.45), 2)
        hs[i - 2 : i + 3] = [2.8, pico - 0.2, pico, pico - 0.25, 2.75]
    serie = pd.DataFrame(
        {"VHM0": hs, "VTPK": 12.0, "VMDR": 200.0, "VPED": 210.0, "VHM0_WW": 0.5, "VHM0_SW1": 1.0, "VHM0_SW2": 0.2},
        index=idx,
    )
    corr = pd.DataFrame({"VHM0_cuantiles_toda": hs * 1.05, "VHM0_cuantiles_asimilado": hs * 1.05}, index=idx)
    return CI.Datos(serie, CI.armar_ramas(serie, corr, INI_E1), INI_E1, pd.DatetimeIndex([]), {}, {})


def _correr(tmp_path, procesos, nombre):
    cand = pd.DataFrame(columns=["tiempo", "decision", "motivo"])
    rev = pd.DataFrame(columns=["tipo", "t_pico", "t_pico_siguiente", "clasificacion_visual"])
    return DD.diagnosticar(_datos(), 2.7, 2.7, U, 48.0, cand, rev, tmp_path / nombre, procesos, B, iets=(48.0, 54.0))


@pytest.fixture(scope="module")
def corrida(tmp_path_factory):
    return _correr(tmp_path_factory.mktemp("diag"), 0, "uno")


def test_tablas_y_tareas(corrida):
    tabla, seps, tam, sd, mk, ver = corrida
    # P11: cada IET en las dos eras sin corregir; las ramas por cuantiles solo en IET*
    claves = set(zip(tabla["rama"], tabla["iet_h"], strict=True))
    assert claves == {(r, i) for r in DD.ERAS for i in (48.0, 54.0)} | {("E0_cuantiles", 48.0), ("E1_cuantiles", 48.0)}
    t48 = tabla[tabla["iet_h"] == 48.0].set_index("rama")
    # §3.7, punto 6: 2 y 4 estaciones en IET* de cada rama; en los otros IET solo la armónica
    assert t48["poisson_2_estaciones_p_ad"].notna().all() and t48["poisson_4_estaciones_p_ad"].notna().all()
    assert tabla.loc[tabla["iet_h"] == 54.0, "poisson_2_estaciones_p_ad"].isna().all()
    # la variante por cuantiles es la sin corregir × 1.05 con u* × 1.05: mismas rachas marcadas y mismos picos
    for e in ("E0", "E1"):
        assert t48.loc[f"{e}_cuantiles", "picos"] == t48.loc[f"{e}_sin_corregir", "picos"]
        assert t48.loc[f"{e}_cuantiles", "poisson_armonico_ad"] == pytest.approx(
            t48.loc[f"{e}_sin_corregir", "poisson_armonico_ad"]
        )
    assert (tabla["rachas_marcadas"] <= tabla["picos"]).all()
    assert t48.loc["E1_sin_corregir", "picos"] < t48.loc["E0_sin_corregir", "picos"]
    assert set(tam["modelo"]) == set(DD.MODELOS) and (tam["replicas"] == 2).all()
    assert set(seps["conjunto"]) == {"picos_sobre_u", "eventos_sobre_u_id"}
    assert len(sd) == 2 * len(D.celdas_sd((48.0, 54.0)))
    assert mk["se_evalua_en"] == "E1_sin_corregir" and mk["E0_sin_corregir"]["lectura"] == DD.NOTA_E0
    assert set(ver["picos_nov_abr_sobre_u"]) == set(R.RAMAS)


def test_determinismo_con_uno_y_dos_procesos(tmp_path, corrida):
    """G9: mismo resultado con 0 (en el proceso) y con 2 procesos hijos; semillas por nombre de tarea."""
    tabla2, seps2, tam2, sd2, _, _ = _correr(tmp_path, 2, "dos")
    pd.testing.assert_frame_equal(corrida[0], tabla2, check_exact=True)
    pd.testing.assert_frame_equal(corrida[2], tam2, check_exact=True)
    pd.testing.assert_frame_equal(corrida[3], sd2, check_exact=True)


def test_reanuda_sin_recalcular(tmp_path):
    a = _correr(tmp_path, 0, "x")
    archivos = sorted((tmp_path / "x").glob("*.pkl"))
    marcas = {p: p.stat().st_mtime_ns for p in archivos}
    b = _correr(tmp_path, 0, "x")
    assert all(p.stat().st_mtime_ns == m for p, m in marcas.items())
    pd.testing.assert_frame_equal(a[0], b[0])


def test_figuras(tmp_path, corrida):
    tabla, seps, _, sd, _, _ = corrida
    hechas = DD.figuras(tabla, seps, sd, 48.0, tmp_path)
    assert hechas and all((tmp_path / h).stat().st_size > 0 for h in hechas)


def test_p_en_el_piso_y_tabla_sd(corrida):
    tabla, _, _, sd, _, _ = corrida
    piso = tabla["poisson_armonico_p_ad_en_el_piso"].astype(bool)
    assert (tabla.loc[piso, "poisson_armonico_p_ad"] == 1 / (B["poisson"] + 1)).all()
    assert (tabla.loc[piso, "poisson_armonico_p_ad_texto"] == f"≤ 1/{B['poisson'] + 1}").all()
    filas = DD.tabla_sd(sd)
    assert {(f["periodo"], f["q"]) for f in filas} == {
        (p, q) for p in ("todo_el_anio", "mayo_octubre") for q in (0.99, 0.995)
    }
