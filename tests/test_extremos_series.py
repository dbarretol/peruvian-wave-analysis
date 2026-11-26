"""Estratos, declustering, tasa, reescalado y bootstrap: pruebas de respuesta conocida (contrato de la F2, §5.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from peruvian_wave_analysis.comun import semillas
from peruvian_wave_analysis.extremos import bootstrap, decluster, reescalado, retorno, tasa
from peruvian_wave_analysis.extremos.estratos import Estratificacion
from peruvian_wave_analysis.extremos.retorno import EstratoRetorno

# --- Estratos -----------------------------------------------------------------------------------


def test_estrato_que_cruza_el_anio():
    est = Estratificacion.desde({"dic-feb": [12, 1, 2], "resto": list(range(3, 12))}, mes_inicio=12)
    t = pd.DatetimeIndex(["1993-12-15", "1994-01-15", "1994-11-30"])
    assert est.anio_bloque(t).tolist() == [1994, 1994, 1994]
    assert est.etiqueta(t).tolist() == ["dic-feb", "dic-feb", "resto"]
    assert est.horas_ventana(1994, "dic-feb") == (31 + 31 + 28) * 24
    assert est.horas_ventana(1996, "dic-feb") == (31 + 31 + 29) * 24
    with pytest.raises(ValueError, match="partido"):
        Estratificacion.desde({"dic-feb": [12, 1, 2], "resto": list(range(3, 12))}, mes_inicio=1)


def test_estratos_invalidos():
    with pytest.raises(ValueError, match="12"):
        Estratificacion.desde({"a": [1, 2], "b": [3]})
    with pytest.raises(ValueError, match="sin meses"):  # F02-K-08
        Estratificacion.desde({"a": [], "b": list(range(1, 13))})
    with pytest.raises(ValueError, match="consecutivos"):
        Estratificacion.desde({"a": [1, 3], "b": [2] + list(range(4, 13))})
    assert Estratificacion.mensual().nombres[0] == "01"
    assert Estratificacion.anual().horas_ventana(2001, "anual") == 365 * 24


# --- Declustering -------------------------------------------------------------------------------


def _serie_decluster():
    v = np.full(45, 0.5)
    v[6], v[10] = 2.0, 1.8  # A: separadas por 12 h = IET exacto → misma tormenta
    v[15] = 1.5  # B: 15 h después del fin de A → tormenta nueva; NaN 6 h después
    v[17] = np.nan
    v[25], v[26] = 3.0, 3.0  # C: empate en el máximo → el primero; sin huecos cerca
    v[32], v[33] = np.nan, np.nan
    v[31] = 1.2  # D: 15 h después de C; huecos justo después
    v[36] = 1.1  # E: 15 h después de D, con los huecos entre ambas (cuentan como sin excedencia)
    idx = pd.date_range("2000-01-01", periods=v.size, freq="3h")
    return pd.DataFrame({"Hs": v, "Tp": np.arange(v.size, dtype=float)}, index=idx)


def test_decluster_respuesta_exacta():
    df = _serie_decluster()
    p = decluster.decluster_rachas(df, u=1.0, iet_horas=12.0)
    pos = [df.index.get_loc(t) for t in p.index]
    assert pos == [6, 15, 25, 31, 36]
    assert p["Tp"].tolist() == [6.0, 15.0, 25.0, 31.0, 36.0]
    assert [df.index.get_loc(t) for t in p["t_fin"]] == [10, 15, 26, 31, 36]
    assert p["n_excedencias"].tolist() == [2, 1, 2, 1, 1]
    assert p["toca_hueco"].tolist() == [False, True, False, True, True]
    np.testing.assert_allclose(p["exceso"], [1.0, 0.5, 2.0, 0.2, 0.1])


def test_decluster_borde_de_la_serie_y_rejilla_incompleta():
    df = _serie_decluster()
    df.iloc[1, 0] = 5.0  # cerca del inicio de la serie: no se sabe qué pasó antes
    p = decluster.decluster_rachas(df, u=1.0, iet_horas=12.0)
    assert bool(p["toca_hueco"].iloc[0])
    with pytest.raises(ValueError, match="rejilla"):
        decluster.decluster_rachas(df.drop(df.index[20]), u=1.0, iet_horas=12.0)
    assert decluster.decluster_rachas(df, u=99.0, iet_horas=12.0).empty


# --- Tasa y retorno de extremo a extremo -----------------------------------------------------------


def test_tasa_y_retorno_extremo_a_extremo():
    est = Estratificacion.desde({"a": [12, 1, 2], "b": list(range(3, 12))}, mes_inicio=12)  # 3 y 9 meses
    t = pd.date_range("2000-12-01", "2002-11-30 21:00", freq="3h")
    validos = np.ones(t.size, bool)
    validos[(t >= "2001-03-01") & (t < "2001-04-01")] = False  # hueco solo en el estrato b
    eventos = pd.DatetimeIndex(
        [
            "2000-12-20",
            "2001-01-10",
            "2002-02-01",  # a: 3
            "2001-05-01",
            "2001-08-01",
            "2002-04-01",
            "2002-06-01",
            "2002-10-01",
        ]
    )  # b: 5
    tabla = tasa.contar_eventos(eventos, tasa.anios_con_dato(t, validos, 3.0, est), est)
    r = tasa.tasas(tabla, est)
    ventana_b = (275) * 24.0  # mar–nov
    E_b = 1.0 + (ventana_b - 31 * 24) / ventana_b
    assert r.loc["a", "anios_con_dato"] == pytest.approx(2.0, abs=1e-12)  # el hueco de b no toca a
    assert r.loc["b", "anios_con_dato"] == pytest.approx(E_b, abs=1e-12)
    assert r.loc["a", "tasa_por_anio"] == pytest.approx(1.5, abs=1e-12)
    assert r.loc["b", "tasa_por_anio"] == pytest.approx(5 / E_b, abs=1e-12)
    # ξ = 0 y la misma σ en ambos estratos: z = σ·ln(T·Σλ) en forma cerrada
    sigma, u, T = 0.5, 2.0, 100.0
    lam = r["tasa_por_anio"].to_numpy()
    rl = retorno.nivel_retorno_multiestrato(u, [EstratoRetorno(float(l_), 0.0, sigma) for l_ in lam], T).valor
    assert abs(rl - (u + sigma * np.log(T * lam.sum()))) < 1e-6
    with pytest.raises(ValueError, match="fuera"):
        tasa.contar_eventos(pd.DatetimeIndex(["2005-01-01"]), tabla, est)


def test_ic_garwood_tabulado():
    # χ²_{0.025}(10) = 3.24697, χ²_{0.975}(12) = 23.3367, χ²_{0.975}(2) = 7.37776 (tablas de χ²)
    lo, hi = tasa.ic_garwood(5, 1.0)
    assert lo == pytest.approx(3.24697 / 2, abs=1e-4) and hi == pytest.approx(23.3367 / 2, abs=1e-4)
    lo, hi = tasa.ic_garwood(0, 2.0)
    assert lo == 0.0 and hi == pytest.approx(7.37776 / 4, abs=1e-4)


# --- Reescalado ---------------------------------------------------------------------------------


def test_reescalado_proceso_puntual_exacto():
    rejilla = pd.date_range("2000-01-01", periods=20, freq="3h")
    ini = rejilla[[2, 8, 15]]
    r = reescalado.intervalos_reescalados(rejilla, np.ones(20, bool), ini, ini, ["x"] * 20, iet_horas=6.0)
    inten = 3 / 42  # 3 inicios en 14 celdas en riesgo (20 − 3·2 muertas) de 3 h
    assert r.intensidad_por_hora_en_riesgo["x"] == pytest.approx(inten)
    np.testing.assert_allclose(r.z, [12 * inten, 15 * inten])


def test_reescalado_llegadas_iguales_a_las_tormentas_reproduce_lo_anterior():
    rng = np.random.default_rng(4)
    rejilla = pd.date_range("2000-01-01", periods=8 * 365 * 3, freq="3h")
    validos = rng.random(rejilla.size) > 0.01
    estr = np.where(rejilla.month <= 6, "s1", "s2")
    ini, fin = reescalado.simular_nula(
        validos, (estr == "s2").astype(int), np.array([0.01, 0.004]), [np.array([0, 2, 4])] * 2, 4, 3.0, rng
    )
    a = reescalado.intervalos_reescalados(rejilla, validos, rejilla[ini], rejilla[fin], estr, 12.0)
    b = reescalado.intervalos_reescalados(rejilla, validos, rejilla[ini], rejilla[fin], estr, 12.0,
                                          llegadas=rejilla[ini])  # fmt: skip
    np.testing.assert_array_equal(a.z, b.z)
    assert a.intensidad_por_hora_en_riesgo == b.intensidad_por_hora_en_riesgo
    # intensidad por celda igual a la constante por estrato → mismos intervalos
    celda = np.array([a.intensidad_por_hora_en_riesgo[e] for e in estr])
    c = reescalado.intervalos_reescalados(rejilla, validos, rejilla[ini], rejilla[fin], None, 12.0, intensidad=celda)
    np.testing.assert_allclose(c.z, a.z, rtol=1e-12)
    assert c.intensidad_por_hora_en_riesgo == {}
    np.testing.assert_array_equal(c.en_riesgo, a.en_riesgo)


def test_reescalado_llegadas_subconjunto_con_tiempo_muerto_de_todas():
    rejilla = pd.date_range("2000-01-01", periods=30, freq="3h")
    validos = np.ones(30, bool)
    ini, fin = rejilla[[2, 9, 20]], rejilla[[3, 10, 21]]  # tres tormentas de 2 celdas; IET 6 h = 2 celdas muertas
    r = reescalado.intervalos_reescalados(rejilla, validos, ini, fin, ["x"] * 30, 6.0, llegadas=rejilla[[2, 20]])
    # en riesgo: 30 celdas menos 3 tormentas × (1 + 2) celdas muertas = 21; 2 llegadas → λ = 2 / (21·3 h)
    lam = 2 / 63
    assert r.intensidad_por_hora_en_riesgo["x"] == pytest.approx(lam)
    assert int(r.en_riesgo.sum()) == 21
    # entre las celdas 2 y 20 hay 18 celdas; fuera de riesgo: 3 (tras la 1.ª), 3 (la 2.ª, que no es llegada)
    np.testing.assert_allclose(r.z, [(18 - 6) * 3 * lam])


def test_reescalado_llegada_que_no_es_tormenta_es_error():
    rejilla = pd.date_range("2000-01-01", periods=30, freq="3h")
    with pytest.raises(ValueError, match="llegada"):
        reescalado.intervalos_reescalados(rejilla, np.ones(30, bool), rejilla[[2, 20]], rejilla[[3, 21]], ["x"] * 30,
                                          6.0, llegadas=rejilla[[5]])  # fmt: skip


def test_reescalado_intensidad_por_celda_armonica():
    rejilla = pd.date_range("2000-01-01", periods=40, freq="3h")
    inten = 0.01 * (1.5 + np.sin(np.arange(40) / 5))
    ini = rejilla[[3, 15, 30]]
    r = reescalado.intervalos_reescalados(rejilla, np.ones(40, bool), ini, ini, None, 6.0, intensidad=inten)
    riesgo = r.en_riesgo
    esperado = [3.0 * inten[4:16][riesgo[4:16]].sum(), 3.0 * inten[16:31][riesgo[16:31]].sum()]
    np.testing.assert_allclose(r.z, esperado)
    with pytest.raises(ValueError, match="finita"):
        reescalado.intervalos_reescalados(rejilla, np.ones(40, bool), ini, ini, None, 6.0, intensidad=-inten)


def test_reescalado_tiempo_muerto_incoherente():
    rejilla = pd.date_range("2000-01-01", periods=20, freq="3h")
    with pytest.raises(ValueError, match="IET"):
        reescalado.intervalos_reescalados(rejilla, np.ones(20, bool), rejilla[[2, 4]], rejilla[[2, 4]], ["x"] * 20, 6.0)


def test_prueba_poisson_tope_de_replicas():
    rejilla = pd.date_range("2000-01-01", periods=400, freq="3h")
    ini = rejilla[[10, 80, 150, 220, 290, 360]]  # 6 tormentas: las réplicas casi nunca tienen 6 inicios
    with pytest.raises(ValueError, match="réplicas"):
        reescalado.prueba_poisson(
            rejilla, np.ones(400, bool), ini, ini, ["x"] * 400, 6.0, 3.0, semillas.generador("prueba", "tope")
        )


def test_prueba_poisson_reproducible_y_ad_decide():
    rng = np.random.default_rng(1)
    rejilla = pd.date_range("2000-01-01", periods=8 * 365 * 4, freq="3h")
    validos = np.ones(rejilla.size, bool)
    estr = np.where(rejilla.month <= 6, "s1", "s2")
    sim = reescalado.simular_nula(
        validos, (estr == "s2").astype(int), np.array([0.01, 0.004]), [np.array([0, 2, 4])] * 2, 4, 3.0, rng
    )
    a = reescalado.prueba_poisson(
        rejilla, validos, rejilla[sim[0]], rejilla[sim[1]], estr, 12.0, 3.0, semillas.generador("prueba", "p")
    )
    b = reescalado.prueba_poisson(
        rejilla, validos, rejilla[sim[0]], rejilla[sim[1]], estr, 12.0, 3.0, semillas.generador("prueba", "p")
    )
    assert a == b and a.B == 99 and 0 < a.p_ad <= 1 and a.rechaza == (a.p_ad <= 0.05)
    k = round(a.p_ad * 100) - 1
    assert a.p_ad == pytest.approx((1 + k) / 100)


# --- Bootstrap ----------------------------------------------------------------------------------


def _serie_bloques():
    est = Estratificacion.desde({"dic-feb": [12, 1, 2], "resto": list(range(3, 12))}, mes_inicio=12)
    t = pd.date_range("2000-12-01", "2003-11-30", freq="D")
    anio = est.anio_bloque(t)
    pos = np.concatenate([np.arange((anio == a).sum()) for a in np.unique(anio)])
    return est, pd.Series(anio * 1000.0 + pos, index=t)  # salto de ~1000 entre años de bloque


def test_bootstrap_sin_saltos_entre_bloques_y_estratos_del_mismo_anio():
    est, s = _serie_bloques()
    bloques = bootstrap.bloques_anuales(s, est)
    assert sorted(bloques) == [2001, 2002, 2003]
    vistos = []

    def estadistico(lista):
        for b in lista:
            assert len(set(est.anio_bloque(b.index))) == 1  # los estratos de un bloque son del mismo año
            assert set(est.etiqueta(b.index)) == {"dic-feb", "resto"}
            vistos.append(np.diff(b.to_numpy()).max())
        return float(sum(np.diff(b.to_numpy()).sum() for b in lista))

    r1 = bootstrap.bootstrap_bloques(bloques, estadistico, 50, semillas.generador("prueba", "bloques"))
    r2 = bootstrap.bootstrap_bloques(bloques, estadistico, 50, semillas.generador("prueba", "bloques"))
    assert max(vistos) == 1.0  # ningún salto entre bloques
    np.testing.assert_array_equal(r1, r2)


def test_bootstrap_por_estrato_exige_declaracion():
    est, s = _serie_bloques()
    por_estrato = bootstrap.bloques_por_estrato(s, est)
    with pytest.raises(ValueError, match="independientes"):
        bootstrap.bootstrap_bloques(por_estrato, lambda _: 0.0, 5, np.random.default_rng(0))
    r = bootstrap.bootstrap_bloques(
        por_estrato,
        lambda bloques_rep: float(len(bloques_rep)),
        5,
        np.random.default_rng(0),
        estratos_independientes=True,
    )
    assert (r == 6).all()


def test_ic_percentil():
    ic = bootstrap.ic_percentil(np.r_[np.arange(1, 101, dtype=float), np.nan], 50.0, no_regular=False)
    assert ic.n_validas == 100 and ic.n_no_finitas == 1 and ic.contiene_punto and not ic.no_regular
    assert not bootstrap.ic_percentil(np.arange(10.0), 20.0, no_regular=False).contiene_punto
    assert bootstrap.ic_percentil(np.arange(10.0), 5.0, no_regular=True).no_regular
    with pytest.raises(TypeError):  # F02-K-07: obligatorio y por nombre
        bootstrap.ic_percentil(np.arange(10.0), 5.0)


# --- Estadísticos del reescalado (F02-K-01) -----------------------------------------------------


def test_estadistico_ad_valor_a_mano_y_por_integral():
    # n = 1 con F(z) = 1/2: A² = −1 − [ln ½ + ln ½] = 2·ln 2 − 1
    assert abs(reescalado.estadistico_ad([np.log(2.0)]) - (2 * np.log(2.0) - 1)) < 1e-12
    # Muestra cualquiera: A² = n ∫ (F_n − u)² / [u(1 − u)] du, integrado por tramos entre los u_(i)
    z = np.random.default_rng(5).exponential(1.3, 25)
    u = np.sort(-np.expm1(-z))
    n = u.size
    bordes = np.r_[0.0, u, 1.0]
    total = sum(
        integrate.quad(lambda t, k=k: (k / n - t) ** 2 / (t * (1 - t)), bordes[k], bordes[k + 1], epsabs=1e-13)[0]
        for k in range(n + 1)
    )
    assert abs(reescalado.estadistico_ad(z) - n * total) < 1e-8


def test_estadistico_ks_frente_a_formula():
    z = np.random.default_rng(6).exponential(0.8, 40)
    F = np.sort(-np.expm1(-z))
    i = np.arange(1, F.size + 1)
    d = max(np.max(i / F.size - F), np.max(F - (i - 1) / F.size))
    assert abs(reescalado.estadistico_ks(z) - d) < 1e-14
