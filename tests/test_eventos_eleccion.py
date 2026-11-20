"""F7: oráculo, métrica y elección de la regla de umbral (§5.4; §8: «Oráculo y control», «Simulación previa»)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import casos as CS
from peruvian_wave_analysis.eventos import eleccion as EL
from peruvian_wave_analysis.eventos import salidas as SA
from peruvian_wave_analysis.eventos import simulacion_previa as SP
from peruvian_wave_analysis.eventos import umbral as U

from . import test_clima_etapa1 as T1

U_ID = 2.70
N = 200  # réplicas por celda en los casos a mano
CELDAS_A_MANO = (
    [("A", f, n) for f in (-0.1, 0.0, 0.1) for n in (334, 685)]
    + [(c, f, n) for c in ("B+", "B-", "control") for f in (-0.1, 0.0) for n in (334, 685)]
    + [("C", f, n) for f in (1.5, 2.0) for n in (334, 685)]
)


def _tabla(errores: dict, estados: dict | None = None, sigma=0.35, seed=0) -> pd.DataFrame:
    """Tabla larga de réplicas dada a mano. ``errores[regla]`` es una función (caso, rng) → error_log por réplica."""
    rng = np.random.default_rng(seed)
    filas = []
    for caso, forma, n0 in CELDAS_A_MANO:
        celda = f"{caso}_{forma:g}_N{n0}"
        ruido = rng.normal(0, 0.05, N)  # error común: las reglas ven las mismas réplicas
        k0 = rng.uniform(4, 14, N)
        for regla, f in errores.items():
            if regla == "oraculo" and caso in ("A", "C"):
                continue
            e = f(caso, ruido)
            est = (estados or {}).get(regla, lambda c, r: np.full(N, U.CON_RESULTADO))(caso, rng)
            for i in range(N):
                filas.append({
                    "sigma": sigma, "celda": celda, "caso": caso, "forma": forma, "N0": n0, "replica": i,
                    "regla": regla, "estado": est[i], "sustituto": "", "error_log": e[i],
                    "u_usado": 3.0, "u0": 3.1 if caso in ("B+", "B-", "control") else np.nan, "k0": k0[i],
                    "pendiente_xi": ruido[i],
                })  # fmt: skip
    return pd.DataFrame(filas)


def _o(caso):
    """Sesgo del oráculo (−1.5 %), solo donde hay oráculo."""
    return -0.015 if caso in ("B+", "B-", "control") else 0.0


def _base():
    """Referencias: k10 con error 0.05 de ruido, k1 sesgado abajo en B− y control, oráculo con −1.5 %."""
    return {
        "k10": lambda c, r: r,
        "k1": lambda c, r: r + (-0.06 if c in ("B-", "control") else 0.0),
        "oraculo": lambda c, r: r + _o(c),
    }


def test_restriccion_b_menos_frente_al_oraculo_y_marca_de_potencia():
    e = _base() | {
        "R2": lambda c, r: r + _o(c) + (-0.03 if c == "B-" else 0.0),  # −3 % frente al oráculo en B−
        "R3": lambda c, r: r + _o(c),  # igual que el oráculo
    }
    res = EL.resumen_celdas(_tabla(e))
    assert not EL.restricciones(res, "R2")["cumple_restricciones"]
    assert len(EL.restricciones(res, "R2")["celdas_b_menos_fallan"]) == 4
    r3 = EL.restricciones(res, "R3")
    assert r3["elegible"] and r3["celdas_b_menos_fallan"] == []
    assert EL.marcas_potencia(res) == {"b_menos_no_discrimina": [], "control_no_discrimina": []}
    # k = 1 sin sesgo → B− y control no discriminan
    e2 = e | {"k1": lambda c, r: r + _o(c)}
    m = EL.marcas_potencia(EL.resumen_celdas(_tabla(e2)))
    assert len(m["b_menos_no_discrimina"]) == 4 and len(m["control_no_discrimina"]) == 4


def test_restriccion_c_absoluta_y_relativa():
    e = _base() | {"R3": lambda c, r: r + (-0.03 if c == "C" else 0.0)}
    assert len(EL.restricciones(EL.resumen_celdas(_tabla(e)), "R3")["celdas_c_fallan"]) == 4
    # si k = 10 incumple en C (−4 %), rige la relativa: R3 a −3 % cumple (−3 % ≥ −4 % + log 0.98)
    e2 = e | {"k10": lambda c, r: r + (-0.04 if c == "C" else 0.0)}
    r = EL.restricciones(EL.resumen_celdas(_tabla(e2)), "R3")
    assert r["celdas_c_fallan"] == [] and len(r["celdas_c_restriccion_relativa"]) == 4


def test_sin_resultado_mayor_al_10_por_ciento_en_una_celda():
    e = _base() | {"R2": lambda c, r: r}

    def est(caso, rng):
        p = 0.2 if caso == "A" else 0.0
        return np.where(rng.random(N) < p, U.SIN_RESULTADO, U.CON_RESULTADO)

    r = EL.restricciones(EL.resumen_celdas(_tabla(e, {"R2": est})), "R2")
    assert not r["cumple_restricciones"] and len(r["celdas_sin_resultado_mayor_10"]) == 6


def test_filtro_del_control_y_eleccion_por_metrica():
    # R2 tiene la menor métrica pero falla el control; R3 cumple todo; R1a peor que R1b
    e = _base() | {
        "R1a": lambda c, r: 1.6 * r + _o(c),
        "R1b": lambda c, r: 1.4 * r + _o(c),
        "R2": lambda c, r: 0.6 * r + _o(c) + (-0.05 if c == "control" else 0.0),
        "R3": lambda c, r: 0.8 * r + _o(c),
    }
    el = EL.elegir(_tabla(e))
    assert el["variante_r1"] == "R1b"
    assert el["mejor_solo_con_restricciones"] == "R2" and el["filtro_cambio_la_eleccion"]
    assert not el["por_regla"]["R2"]["cumple_control"]
    assert el["elegida"] == "R3" and el["empatadas"] == ["R3"]


def test_empate_por_pares_prefiere_r1_luego_r3():
    # métricas casi iguales (diferencia ≪ 2 errores típicos): empatan y gana R1, luego R3
    e = _base() | {
        "R1a": lambda c, r: 1.002 * r + _o(c),
        "R1b": lambda c, r: 1.5 * r + _o(c),
        "R2": lambda c, r: 0.998 * r + _o(c),
        "R3": lambda c, r: 1.001 * r + _o(c),
    }
    df = _tabla(e)
    df.loc[df["regla"].isin(["R1a", "R2", "R3"]), "error_log"] += np.random.default_rng(9).normal(
        0, 0.01, int(df["regla"].isin(["R1a", "R2", "R3"]).sum())
    )
    el = EL.elegir(df)
    assert set(el["empatadas"]) == {"R1a", "R2", "R3"} and el["elegida"] == "R1a"
    sin_r1 = EL.elegir(df[~df["regla"].isin(["R1a", "R1b"])])
    assert sin_r1["elegida"] == "R3"
    # con una diferencia clara no hay empate
    e2 = e | {"R2": lambda c, r: 0.5 * r + _o(c)}
    el2 = EL.elegir(_tabla(e2))
    assert el2["empatadas"] == ["R2"] and el2["elegida"] == "R2"


def test_ninguna_cumple_detiene():
    e = _base() | {"R2": lambda c, r: r - 0.05, "R3": lambda c, r: r - 0.05}
    el = EL.elegir(_tabla(e))
    assert el["elegida"] is None and "detiene" in el["motivo"]


def test_metrica_normalizada_por_k10():
    e = _base() | {"R3": lambda c, r: 2.0 * r}
    res = EL.resumen_celdas(_tabla(e))
    assert EL.metrica(res, "R3") == pytest.approx(2.0, rel=1e-12)
    assert EL.metrica(res, "k10") == pytest.approx(1.0)


# --- Oráculo, k = 10 y réplicas ---------------------------------------------------------------------


def _rejilla():
    picos = np.round(np.linspace(2.71, 6.0, 700), 2)
    return U.rejilla(picos)


def test_oraculo_es_el_primer_u_sobre_u0():
    rej = _rejilla()
    for j in (3, 8, 15):
        u0 = (rej.u[j - 1] + rej.u[j]) / 2
        assert SP.indice_oraculo(rej, u0) == j
        assert SP.indice_oraculo(rej, float(rej.u[j])) == j + 1  # estrictamente por encima
    assert SP.indice_oraculo(rej, None) is None
    assert SP.indice_oraculo(rej, 99.0) is None


def test_k10_con_respaldo_11_y_9():
    rej = _rejilla()
    rl = np.full(rej.u.size, 5.0)
    p10, p11, p9 = (SP.posicion_de_k(rej, k) for k in (10, 11, 9))
    assert SP.referencia_k10(rej, rl) == (p10, "k10")
    rl[p10] = np.nan
    assert SP.referencia_k10(rej, rl) == (p11, "k11")
    rl[p11] = np.nan
    assert SP.referencia_k10(rej, rl) == (p9, "k9")
    rl[p9] = np.nan
    assert SP.referencia_k10(rej, rl) == (None, "ninguno")


def test_posicion_de_k_con_repetidos():
    rej = U.Rejilla(np.array([1, 2, 3, 5]), np.array([9, 8, 7, 5]), np.array([3.0, 3.1, 3.2, 3.3]),
                    np.array([9, 8, 7, 5]), True, (4,))  # fmt: skip
    assert SP.posicion_de_k(rej, 4) == 2 and SP.posicion_de_k(rej, 5) == 3 and SP.posicion_de_k(rej, 1) == 0


def _celdas(n0=(334, 685)):
    conteo = {"u_id_m": U_ID, "n0_simulacion": {"n0": list(n0), "anios_simulados": [33, 46]}}
    return SP.celdas_de(conteo)


def test_celdas_y_nombres():
    cs = _celdas()
    assert len(cs) == 22 and sum(c.caso == "control" for c in cs) == 4
    s = SP.celdas_de({"u_id_m": U_ID, "n0_simulacion": {"n0": [334, 685], "anios_simulados": [33, 46]}}, 0.26)
    assert all(SP.nombre_celda(c).endswith("_s0.26") for c in s)
    assert {SP.nombre_celda(c) for c in cs} == {c.nombre for c in cs}


def test_replica_misma_semilla_mismas_reglas_y_sustituto_k10():
    c = next(x for x in _celdas() if x.caso == "B-" and x.N0 == 334)
    a = pd.DataFrame(SP.evaluar_replica(c, 3))
    b = pd.DataFrame(SP.evaluar_replica(c, 3))
    pd.testing.assert_frame_equal(a, b)  # misma semilla → mismo resultado
    assert set(a["regla"]) == {"R1a", "R1b", "R2", "R3", "k1", "k10", "oraculo"}
    assert a["n_picos"].nunique() == 1 and a["rl100_verdadero"].nunique() == 1  # mismas réplicas
    io = int(a["indice_oraculo"].iloc[0])
    assert a.loc[a["regla"] == "oraculo", "indice_usado"].item() == io
    assert a.loc[a["regla"] == "oraculo", "u_usado"].item() > a["u0"].iloc[0]
    for _, f in a.iterrows():
        if f["estado"] == U.SIN_RESULTADO:  # entra con k = 10
            assert f["sustituto"].startswith("k")
            assert f["indice_usado"] == a.loc[a.regla == "k10", "indice_usado"].item()


def test_regla_que_elige_el_oraculo_difiere_cero_con_gpd_exacta_sobre_u0(monkeypatch):
    """Una regla que elige el u_k del oráculo tiene su mismo error: la diferencia regla − oráculo es exactamente 0."""
    c = next(x for x in _celdas() if x.caso == "B-" and x.N0 == 685)

    def oraculo_como_regla(picos, rej, aj, reglas, semilla_r3):
        r = SP.CS.replica(c, SP.semillas.generador("f07", "simulacion-previa", SP.nombre_celda(c), str(i)))
        j = SP.indice_oraculo(rej, r.u0)
        return {"R3": U.ResultadoRegla(U.CON_RESULTADO, j, (j, rej.u.size - 1), {})}

    monkeypatch.setattr(SP, "aplicar_reglas", oraculo_como_regla)
    for i in range(4):
        a = pd.DataFrame(SP.evaluar_replica(c, i, ("R3",))).set_index("regla")
        assert a.loc["R3", "indice_usado"] == a.loc["oraculo", "indice_usado"]
        assert a.loc["R3", "error_log"] - a.loc["oraculo", "error_log"] == 0.0
        assert np.isfinite(a.loc["oraculo", "error_log"])


def test_pendiente_xi():
    n = np.array([400, 300, 200, 100, 50])
    xi = 0.1 * np.log(n) + 1.0
    assert SP.pendiente_xi(xi, n, 0) == pytest.approx(0.1)
    assert np.isnan(SP.pendiente_xi(xi, n, 3))  # 2 puntos
    xi[1] = np.nan
    assert SP.pendiente_xi(xi, n, 0) == pytest.approx(0.1)


def test_simulacion_previa_de_extremo_a_extremo(tmp_path, monkeypatch):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    codigo = tmp_path / "codigo"
    codigo.mkdir()
    (codigo / "m.py").write_text("X = 1\n", encoding="utf-8")
    conteo = {"u_id_m": U_ID, "n0_simulacion": {"n0": [60, 80], "anios_simulados": [33, 46]}}
    T1._publicar(raiz, SA.ruta_conteo_previo(raiz),
                 {"conteo_previo.json": lambda p: json_estricto.escribir(p, conteo)}, codigo)  # fmt: skip
    monkeypatch.setattr(SP, "REGLAS", ("R1a", "R1b", "R2"))  # sin R3: la prueba es de la etapa, no de la regla
    monkeypatch.setattr(SP, "SIGMAS_SENSIBILIDAD", (0.5,))
    monkeypatch.setattr(SP, "TROZO", 2)
    salida = SP.ejecutar(raiz, procesos_=0, replicas=4)
    man = etapa.exigir_publicada(salida)
    assert set(man["insumos"]) == {"conteo_previo"}
    df = pd.read_csv(salida / "replicas.csv")
    assert df.groupby("sigma")["celda"].nunique().loc[0.35] == 22
    assert set(df.loc[df["sigma"] == 0.35, "regla"]) == {"R1a", "R1b", "R2", "k1", "k10", "oraculo"}
    el = json_estricto.leer(salida / "eleccion.json")
    assert "eleccion" in el and el["replicas_por_celda"] == 4
    if el["eleccion"]["elegida"] is not None:
        assert len(el["sensibilidad_sigma"]) == 1 and set(df["sigma"]) == {0.35, 0.5}
    etapa.exigir_publicada(SA.ruta_conteo_previo(raiz))  # la simulación no borra el conteo
    # reanudar: publicar otra vez deja el mismo resultado
    salida2 = SP.ejecutar(raiz, procesos_=0, replicas=4)
    pd.testing.assert_frame_equal(df, pd.read_csv(salida2 / "replicas.csv"))
    assert CS.SIGMA == 0.35
