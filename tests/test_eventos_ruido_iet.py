"""F7: regla del IET y su calibración bajo el nulo (contrato §3.3, §5.3; §8, «Regla del IET»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import ruido_iet as RI
from peruvian_wave_analysis.validacion import reanudable

IETS = RI.IETS_H  # 42, 48, …, 120
N_CORTA = 20_000


def _rl(base=4.6, filas=None):
    rl = np.full((len(IETS), 3), base)
    for j, v in (filas or {}).items():
        rl[j] = v
    return rl


def test_cambios_usan_las_6_comparaciones():
    rl = _rl()
    rl[0, 2] = 4.6 * 1.03  # 42 h, solo en el tercer umbral
    m = RI.cambios(rl)
    assert m[1] == pytest.approx(0.03, rel=1e-12)  # 48 h ve el cambio de su vecino de abajo
    assert m[2] == 0.0 and m[0] == pytest.approx(1 - 1 / 1.03)
    rl2 = _rl()
    rl2[8, 0] = 4.6 * 1.02  # 120 h: solo vecino de abajo
    assert RI.cambios(rl2)[7] == pytest.approx(0.02)


def test_regla_del_iet_a_mano():
    assert RI.regla_iet(RI.cambios(_rl()), IETS, 0.01) == (48.0, False)
    rl = _rl(filas={0: 4.0, 1: 4.3})  # 48 y 54 h inestables; desde 60 h, estable
    assert RI.regla_iet(RI.cambios(rl), IETS, 0.01) == (60.0, False)
    m = np.array([0.0, 0.05, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])  # 42 h nunca se elige
    assert RI.regla_iet(m, IETS, 0.01) == (54.0, False)


def test_sin_iet_valido_marca_sensible():
    m = np.array([0.0, 0.05, 0.04, 0.03, 0.06, 0.07, 0.08, 0.09, 0.10])
    assert RI.regla_iet(m, IETS, 0.01) == (60.0, True)
    rl = _rl()
    rl[3, 1] = np.nan  # un RL100 nulo: ni él ni sus vecinos cumplen
    c = RI.cambios(rl)
    assert np.isinf(c[3]) and np.isinf(c[2]) and np.isinf(c[4])


def test_escala_hs_afin():
    x = pd.Series(np.random.default_rng(0).gumbel(size=50_000))
    y = RI.a_escala_hs(x, 2.70, 3.15)
    q95, q99 = np.quantile(y, [0.95, 0.99])
    assert q95 == pytest.approx(2.70, abs=1e-9) and q99 == pytest.approx(3.15, abs=1e-9)
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(1.0)


def _df_maximos(fallos: dict[str, tuple[float, float]], n=100):
    """fallos[proceso] = (fracción de réplicas con 48 h > 1 %, valor del máximo en esas réplicas)."""
    filas = []
    for p, (frac, valor) in fallos.items():
        for i in range(n):
            m = [0.0] * len(IETS)
            m[1] = valor if i < frac * n else 0.005
            filas.append({"proceso": p, "replica": i, "maximo_cambio": m})
    return pd.DataFrame(filas)


def test_criterio_por_proceso_toma_el_mayor_y_dice_quien_decide():
    df = _df_maximos({"ar1_central": (0.10, 0.02), "fs_central": (0.30, 0.02), "fs_alta": (0.50, 0.03)})
    cal = RI.calibrar(df, ["ar1_central", "fs_central", "fs_alta"])
    assert cal["proceso_que_decide"] == "fs_alta"
    assert cal["tolerancia"] == pytest.approx(cal["por_proceso"]["fs_alta"]["p80_maximo_48"])
    # un proceso fuera del criterio no decide
    cal2 = RI.calibrar(df, ["ar1_central", "fs_central"])
    assert cal2["proceso_que_decide"] == "fs_central"
    # si ninguno supera el 20 %, rige el 1 %
    cal3 = RI.calibrar(_df_maximos({"ar1_central": (0.10, 0.02)}), ["ar1_central"])
    assert cal3["tolerancia"] == RI.TOL_BASE and cal3["proceso_que_decide"] is None
    el = RI.eleccion_bajo_el_nulo(df, 0.01)
    assert el["por_proceso"]["fs_central"]["48"] == pytest.approx(0.70)
    assert el["por_proceso"]["fs_central"]["54"] == pytest.approx(0.30)


def test_comprobacion_de_la_duracion():
    cal = [{"duracion_media_h": 40.0, "en_el_criterio": True}, {"duracion_media_h": 55.0, "en_el_criterio": True},
           {"duracion_media_h": 90.0, "en_el_criterio": False}]  # fmt: skip
    assert RI.comprobacion_duracion(45.0, cal)["sitio_dentro_del_rango"]
    assert not RI.comprobacion_duracion(70.0, cal)["sitio_dentro_del_rango"]  # el de 90 h no cuenta


def test_tasa_baja_con_la_persistencia():
    t_corta = RI.tasa_media("ar1", 24.0, 2.70, 3.15, N_CORTA)[0]
    t_larga = RI.tasa_media("ar1", 150.0, 2.70, 3.15, N_CORTA)[0]
    assert t_larga < t_corta
    f_alta = RI.tasa_media("fs", 0.5, 2.70, 3.15, N_CORTA)[0]
    f_baja = RI.tasa_media("fs", 0.1, 2.70, 3.15, N_CORTA)[0]
    assert f_baja < f_alta


def test_calibrar_parametro_llega_a_la_tasa():
    objetivo = RI.tasa_media("ar1", 100.0, 2.70, 3.15, N_CORTA)[0]
    c = RI.calibrar_parametro("ar1", objetivo, 2.70, 3.15, N_CORTA)
    assert c["en_el_criterio"] and c["tasa_lograda"] == pytest.approx(objetivo, rel=0.02)
    imposible = RI.calibrar_parametro("fs", 1e4, 2.70, 3.15, N_CORTA)  # fuera del rango del parámetro
    assert not imposible["en_el_criterio"]


def test_replica_determinista():
    p = RI.Proceso("ar1", 120.0, "ar1_central")
    a = RI.replica(p, 0, 2.70, 3.15, n=N_CORTA)
    assert a == RI.replica(p, 0, 2.70, 3.15, n=N_CORTA)
    assert len(a["maximo_cambio"]) == len(IETS)
    assert all(np.diff(a["eventos"]) <= 0)  # más IET, menos eventos


def test_ruido_iet_de_extremo_a_extremo(tmp_path, monkeypatch):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    sitio = {"p95_hs_m": 2.70, "p99_hs_m": 3.15, "tasas": {"central": 12.4}, "duracion_media_sitio_h": 45.0,
             "eventos_sitio_e0_48": 571}  # fmt: skip
    monkeypatch.setattr(RI, "datos_del_sitio", lambda r: ({}, sitio))
    calib = [{"familia": "ar1", "parametro": 120.0, "etiqueta": "ar1_central", "tasa_objetivo": 12.4,
              "tasa_lograda": 12.4, "duracion_media_h": 50.0, "en_el_criterio": True},
             {"familia": "fs", "parametro": 0.1, "etiqueta": "fs_central", "tasa_objetivo": 12.4,
              "tasa_lograda": 12.4, "duracion_media_h": 40.0, "en_el_criterio": True}]  # fmt: skip
    monkeypatch.setattr(RI, "calibrar_procesos", lambda t, a, b: calib)
    monkeypatch.setattr(RI, "TROZO", 1)
    orig = RI.replica
    monkeypatch.setattr(RI, "replica", lambda p, i, a, b: orig(p, i, a, b, n=N_CORTA))
    salida = RI.ejecutar(raiz, procesos_=0, replicas=2)
    etapa.exigir_publicada(salida)
    r = json_estricto.leer(salida / "ruido_iet.json")
    assert set(r["tolerancia"]["por_proceso"]) == {"ar1_central", "fs_central"}
    assert r["comprobacion_duracion"]["sitio_dentro_del_rango"] and r["valores_de_picos_leidos"] is False
    assert len(pd.read_csv(salida / "replicas.csv")) == 2 * 2 * len(IETS)


def test_ruido_iet_igual_con_0_y_2_procesos(tmp_path, monkeypatch):
    """G9 (F07-R-03): el ruido del IET da lo mismo en el proceso y con 2 procesos hijos (semilla por réplica).
    Réplicas de largo completo: los hijos importan ``replica`` sin el parche de largo corto."""
    monkeypatch.setattr(RI, "TROZO", 1)
    t = RI.tareas([RI.Proceso("ar1", 120.0, "ar1_central")], 2, 2.70, 3.15)
    assert len(t) == 2
    a = reanudable.ejecutar(RI._trabajo, t, tmp_path / "cero", 0)
    b = reanudable.ejecutar(RI._trabajo, t, tmp_path / "dos", 2)
    a = [r for parte in a for r in parte]
    b = [r for parte in b for r in parte]
    assert len(a) == len(b) == 2
    for x, y in zip(a, b, strict=True):
        assert x.keys() == y.keys()
        for k in x:
            np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(y[k]), err_msg=k)
