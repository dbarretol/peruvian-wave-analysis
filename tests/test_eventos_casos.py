"""F7: casos sintéticos de la simulación previa (contrato §5.1 y §8, «Caso B», «Caso C», «Oráculo y control»)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import casos as CS
from peruvian_wave_analysis.eventos import piloto_r3 as PR
from peruvian_wave_analysis.eventos import salidas as SA

from . import test_clima_etapa1 as T1

U_ID = 2.70


def _celda(caso, forma, N0=685, anios=46.0):
    return CS.Celda(caso, forma, N0, anios, U_ID)


def test_celdas_de_la_metrica_y_del_control():
    c = CS.celdas((334, 685), (33.0, 46.0), U_ID)
    assert len(c) == 18 and len({x.nombre for x in c}) == 18
    assert {(x.N0, x.anios) for x in c} == {(334, 33.0), (685, 46.0)}
    k = CS.celdas((334, 685), (33.0, 46.0), U_ID, control=True)
    assert len(k) == 4 and all(x.caso == "control" for x in k)
    assert {round(x.xi_abajo, 10) for x in k} == {-0.4, -0.3}  # ξ0 − 0.30


@pytest.mark.parametrize("caso", ["B+", "B-", "control"])
@pytest.mark.parametrize("xi0", [-0.1, 0.0])
def test_caso_b_continuo_y_masa_bajo_u0(caso, xi0):
    c = _celda(caso, xi0)
    for k0 in (4.0, 9.3, 14.0):
        d0 = CS.d0_de(c, k0)
        s = CS.supervivencia(c, np.array([d0 - 1e-7, d0, d0 + 1e-7]), k0)
        assert s[0] == pytest.approx(s[2], rel=1e-5)  # S continua
        # densidad continua: −dS/dx a cada lado de u0
        h = 1e-6
        izq = (CS.supervivencia(c, np.array([d0 - 2 * h]), k0) - CS.supervivencia(c, np.array([d0 - h]), k0)) / h
        der = (CS.supervivencia(c, np.array([d0 + h]), k0) - CS.supervivencia(c, np.array([d0 + 2 * h]), k0)) / h
        assert izq[0] == pytest.approx(der[0], rel=1e-3)
        n_k0 = c.N0 * CS.r_de(c.N0) ** (-k0 / 20)
        assert 1 - s[1] == pytest.approx(1 - n_k0 / c.N0, rel=1e-9)  # masa bajo u0


def test_caso_b_muestreo_y_k0():
    c = _celda("B-", 0.0)
    rng = np.random.default_rng(1)
    k0s = [CS.replica(c, rng).k0 for _ in range(200)]
    assert min(k0s) >= 4.0 and max(k0s) <= 14.0 and np.std(k0s) > 2.0
    x = CS.excesos(c, 200_000, np.random.default_rng(2), 9.0)
    grid = np.array([0.1, CS.d0_de(c, 9.0), 0.8, 1.5])
    emp = np.array([(x > g).mean() for g in grid])
    np.testing.assert_allclose(emp, CS.supervivencia(c, grid, 9.0), atol=3e-3)


def test_caso_c_weibull():
    for k in (1.5, 2.0):
        c = _celda("C", k)
        lam = c.escala_weibull()
        assert stats.weibull_min(k, scale=lam).ppf(0.95) == pytest.approx(U_ID, rel=1e-12)  # P95 sin truncar
        x = CS.excesos(c, 200_000, np.random.default_rng(3), None)
        assert x.min() >= 0
        cond = stats.weibull_min(k, scale=lam).sf(U_ID + 0.5) / stats.weibull_min(k, scale=lam).sf(U_ID)
        assert (x > 0.5).mean() == pytest.approx(cond, abs=3e-3)  # truncada en u_id


def test_picos_redondeados_y_rl_verdadero():
    c = _celda("A", 0.0)
    r = CS.replica(c, np.random.default_rng(4))
    assert np.allclose(np.round(r.picos * 100), r.picos * 100) and r.picos.min() > U_ID
    # GPD exacta: RL = u + σ·log(λT) con ξ = 0
    assert r.rl100_verdadero == pytest.approx(U_ID + 0.35 * np.log(685 / 46 * 100), rel=1e-9)
    assert abs(r.picos.size - 685) < 5 * np.sqrt(685)


def test_piloto_r3_de_extremo_a_extremo(tmp_path, monkeypatch):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    codigo = tmp_path / "codigo"
    codigo.mkdir()
    (codigo / "m.py").write_text("X = 1\n", encoding="utf-8")
    conteo = {"u_id_m": U_ID, "n0_simulacion": {"n0": [60, 80], "anios_simulados": [33, 46]}}
    T1._publicar(raiz, SA.ruta_conteo_previo(raiz),
                 {"conteo_previo.json": lambda p: json_estricto.escribir(p, conteo)}, codigo)  # fmt: skip
    monkeypatch.setattr(PR, "CELDAS", (("A", 0.0, 0),))
    monkeypatch.setattr(PR, "TROZO", 1)
    salida = PR.ejecutar(raiz, procesos_=0, replicas=2)
    man = etapa.exigir_publicada(salida)
    assert set(man["insumos"]) == {"conteo_previo"}
    r = json_estricto.leer(salida / "piloto_r3.json")
    assert r["replicas"] == 2 and "atajo_aceptado" in r and r["celdas"] == ["A_0_N60"]
    etapa.exigir_publicada(SA.ruta_conteo_previo(raiz))  # el piloto no borra el conteo
