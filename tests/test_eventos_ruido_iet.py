"""F7: regla del IET y su calibración bajo el nulo (contrato §3.3, §5.3; §8, «Regla del IET»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import ruido_iet as RI

IETS = RI.IETS_H  # 42, 48, …, 120


def _rl(base=4.6, filas=None):
    rl = np.full((len(IETS), 3), base)
    for j, v in (filas or {}).items():
        rl[j] = v
    return rl


def test_cambios_usan_las_6_comparaciones():
    rl = _rl()
    rl[0, 2] = 4.6 * 1.03  # 42 h, solo en el tercer umbral
    m = RI.cambios(rl)
    assert m[1] == pytest.approx(0.03 / 1.0, rel=1e-12)  # 48 h ve el cambio de su vecino de abajo
    assert m[2] == 0.0 and m[0] == pytest.approx(1 - 1 / 1.03)
    rl2 = _rl()
    rl2[8, 0] = 4.6 * 1.02  # 120 h: solo vecino de abajo
    assert RI.cambios(rl2)[7] == pytest.approx(0.02)


def test_regla_del_iet_a_mano():
    # 48 h estable → 48 h
    assert RI.regla_iet(RI.cambios(_rl()), IETS, 0.01) == (48.0, False)
    # 48 h y 54 h cambian 3 % frente a 42 h y entre sí; desde 60 h estable → 60 h
    rl = _rl(filas={0: 4.0, 1: 4.3})
    assert RI.regla_iet(RI.cambios(rl), IETS, 0.01) == (60.0, False)
    # 42 h nunca se elige, aunque sea estable
    m = np.array([0.0, 0.05, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    assert RI.regla_iet(m, IETS, 0.01) == (54.0, False)


def test_sin_iet_valido_marca_sensible():
    m = np.array([0.0, 0.05, 0.04, 0.03, 0.06, 0.07, 0.08, 0.09, 0.10])
    assert RI.regla_iet(m, IETS, 0.01) == (60.0, True)  # el de menor cambio, con la marca
    rl = _rl()
    rl[3, 1] = np.nan  # un RL100 nulo: sus vecinos y él no cumplen
    c = RI.cambios(rl)
    assert np.isinf(c[3]) and np.isinf(c[2]) and np.isinf(c[4])


def test_escala_hs_afin():
    x = pd.Series(np.random.default_rng(0).gumbel(size=50_000))
    y = RI.a_escala_hs(x)
    q95, q99 = np.quantile(y, [0.95, 0.99])
    assert q95 == pytest.approx(2.70, abs=1e-9) and q99 == pytest.approx(3.15, abs=1e-9)
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(1.0)


def test_calibracion_con_maximo_en_48():
    filas = []
    for p, a in RI.PROCESOS:
        for i in range(100):
            m = [0.0] * len(IETS)
            m[1] = 0.02 if (p == "fs" and i < 30) else 0.005  # FS: 30 % de fallos en 48 h
            filas.append({"proceso": p, "parametro": a, "replica": i, "maximo_cambio": m})
    df = pd.DataFrame(filas)
    cal = RI.calibrar(df)
    assert cal["juntos"]["fallo_48_con_1"] == pytest.approx(0.15)  # juntos: 15 % → sigue el 1 %
    assert cal["juntos"]["tolerancia"] == RI.TOL_BASE
    fs = cal["por_proceso"]["fs_0.25"]
    assert fs["fallo_48_con_1"] == pytest.approx(0.30) and fs["tolerancia"] == pytest.approx(fs["p80_maximo_48"])
    el = RI.eleccion_bajo_el_nulo(df, 0.01)
    assert el["juntos"]["48"] == pytest.approx(0.85) and el["juntos"]["54"] == pytest.approx(0.15)


def test_replica_determinista_y_escala():
    a = RI.replica("ar1", 48.0, 0, n=20_000)
    b = RI.replica("ar1", 48.0, 0, n=20_000)
    assert a == b
    assert len(a["maximo_cambio"]) == len(IETS) and a["bajada"] == 0.5
    assert all(np.diff(a["eventos"]) <= 0)  # más IET, menos eventos


def test_ruido_iet_de_extremo_a_extremo(tmp_path, monkeypatch):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    monkeypatch.setattr(RI, "PROCESOS", (("ar1", 48.0), ("fs", 0.5)))
    monkeypatch.setattr(RI, "TROZO", 1)
    orig = RI.replica
    monkeypatch.setattr(RI, "replica", lambda p, a, i: orig(p, a, i, n=20_000))
    salida = RI.ejecutar(raiz, procesos_=0, replicas=2)
    etapa.exigir_publicada(salida)
    r = json_estricto.leer(salida / "ruido_iet.json")
    assert set(r["calibracion"]["por_proceso"]) == {"ar1_48", "fs_0.5"}
    assert len(pd.read_csv(salida / "replicas.csv")) == 2 * 2 * len(IETS)
