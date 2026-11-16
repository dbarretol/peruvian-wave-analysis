"""Pruebas sintéticas de la F4 (contrato de la F4, v7, §5). Cada prueba nombra la fila de §5 que cubre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.homogeneidad import altimetros as A
from peruvian_wave_analysis.homogeneidad import config as C
from peruvian_wave_analysis.homogeneidad import familias as F
from peruvian_wave_analysis.homogeneidad import permutacion as P
from peruvian_wave_analysis.homogeneidad import relevante as R
from peruvian_wave_analysis.homogeneidad import run
from peruvian_wave_analysis.homogeneidad import tendencia as T
from peruvian_wave_analysis.homogeneidad.cuantiles import (
    HistAnual,
    TablaCola,
    cuantil_ponderado,
    cuantiles_mensuales,
    pesos_mensuales,
)

# --- Datos sintéticos -----------------------------------------------------------------------


def serie_diaria(anios, rng, escala_mes=None, factor=None, freq="D") -> pd.Series:
    """Un valor por día (o por ``freq``), gamma con escala que depende del mes; ``factor``: año → multiplicador."""
    t = pd.date_range(f"{anios[0]}-01-01", f"{anios[-1]}-12-31", freq=freq)
    esc = escala_mes if escala_mes is not None else 1.0 + 0.5 * np.cos(2 * np.pi * (np.arange(12)) / 12)
    x = rng.gamma(3.0, esc[t.month.to_numpy() - 1] / 3.0)
    if factor:
        x = x * np.array([factor.get(y, 1.0) for y in t.year])
    return pd.Series(x, index=t)


def contraste_sintetico(rng, n1=10, n2=20, paso=0.0, mes_inicio=1, freq="D") -> F.Contraste:
    anios = list(range(2000, 2000 + n1 + n2))
    s = serie_diaria(anios, rng, factor={y: 1.0 + paso for y in anios[n1:]}, freq=freq)
    l1 = (f"{anios[0]}-01-01", f"{anios[n1 - 1]}-12-31")
    l2 = (f"{anios[n1]}-01-01", f"{anios[-1]}-12-31")
    return F.preparar_contraste("sint", s, l1, l2, mes_inicio)


# --- Cuantiles con peso mensual ---------------------------------------------------------------


def test_cuantil_ponderado_corrige_la_mezcla_de_meses():
    """§5, cuantiles con peso mensual: mismo clima y otra mezcla de meses."""
    rng = np.random.default_rng(1)
    esc = np.array([1.0] * 6 + [3.0] * 6)
    a = serie_diaria(range(2000, 2040), rng, esc)
    b = a[(a.index.month <= 6) | (a.index.day <= 5)]  # casi sin meses de olas altas
    qa, qb = cuantiles_mensuales(a, (0.9,))[0], cuantiles_mensuales(b, (0.9,))[0]
    assert abs(qb / qa - 1) < 0.03
    sa, sb = np.quantile(a, 0.9), np.quantile(b, 0.9)
    assert sb / sa < 0.8


def test_cuantil_por_inversion_y_tablas_coinciden():
    rng = np.random.default_rng(2)
    s = serie_diaria(range(2000, 2010), rng)
    w = pesos_mensuales(s.index)
    q = cuantil_ponderado(s.to_numpy(), w, (0.5, 0.99))
    tc = TablaCola.desde_serie(s, 1, float(np.quantile(s, 0.9)))
    assert np.allclose(tc.cuantiles(np.ones((1, 10)), (0.99,))[0], q[1])
    h = HistAnual.desde_serie(s.round(2))
    assert np.allclose(h.cuantiles(np.ones((1, 10)), (0.5, 0.99))[0], cuantiles_mensuales(s.round(2), (0.5, 0.99)))


def test_inyeccion_cambia_los_cuantiles_en_delta_exacto():
    """§5, inyección de δ: multiplicar un tramo por (1 + δ) cambia sus cuantiles en δ exacto."""
    rng = np.random.default_rng(3)
    c = contraste_sintetico(rng)
    f = P.factores_inyeccion(c.identidad, c.lado, 1, 0.07)
    t2 = c.tabla.con_factores(f)
    lado2 = c.lado[None, :]
    q0 = c.tabla.cuantiles(lado2, (0.99, 0.999))
    q1 = t2.cuantiles(lado2, (0.99, 0.999))
    assert np.allclose(q1 / q0, 1.07)
    assert np.isclose(t2.media_maximos(lado2)[0] / c.tabla.media_maximos(lado2)[0], 1.07)


# --- Permutación con max-T --------------------------------------------------------------------


def test_permutacion_sin_cambio_rechaza_al_nivel():
    """§5: años iid sin cambio, tasa de rechazo de la familia entre 3 % y 7 % en 1 000 réplicas."""
    c = contraste_sintetico(np.random.default_rng(4))
    p = F.simular_A(c, "anios", 199, 1000, 0.0, 1, False, 0.3, ("prueba", "nula"))
    assert 0.03 <= np.mean(p <= 0.05) <= 0.07


def test_permutacion_detecta_un_escalon_de_10pct():
    """§5: con un escalón de 10 % en la cola, rechazo casi siempre (13 frente a 34 años de 3 h, como C1)."""
    rng = np.random.default_rng(5)
    rech = 0
    for i in range(20):
        c = contraste_sintetico(rng, n1=13, n2=34, paso=0.10, freq="3h")
        r = P.prueba_maxT(c.tabla, c.lado, c.identidad, "anios", 199, np.random.default_rng(i))
        rech += r.p_min <= 0.05
    assert rech >= 19


def test_maxT_rechaza_mas_que_holm_con_estadisticos_correlacionados():
    """§5: con tres estadísticos muy correlacionados (ρ = 0.95), max-T rechaza más que Holm."""
    rng = np.random.default_rng(6)
    cov = np.full((3, 3), 0.95) + 0.05 * np.eye(3)
    n_maxT = n_holm = 0
    for _ in range(300):
        Tp = rng.multivariate_normal(np.zeros(3), cov, 999)
        T_obs = rng.multivariate_normal(np.full(3, 2.3), cov)
        aj = P.ajuste_maxT(T_obs, Tp)
        n_maxT += aj.p_ajustado.min() <= 0.05
        n_holm += P.holm(aj.p_crudo).min() <= 0.05
    assert n_maxT > n_holm


def test_bloques_parciales_se_permutan_y_quedan_fuera_de_la_media_de_maximos():
    """§5: corte que no cae el 1 de enero (años de marzo a febrero)."""
    rng = np.random.default_rng(7)
    s = serie_diaria(range(2000, 2012), rng)
    c = F.preparar_contraste("c2", s, ("2000-02-01", "2006-02-28"), ("2006-03-01", "2011-08-31"), 3)
    # 1999 (solo febrero de 2000) y 2011 (marzo–agosto) son parciales
    assert list(c.anios[~c.tabla.completo]) == [1999, 2011]
    assert len(P.unidades(c.lado, "anios")) == c.anios.size
    P2 = c.lado[None, :].astype(float)
    esperado = c.tabla.maximos[(c.lado == 1) & c.tabla.completo].mean()
    assert np.isclose(c.tabla.media_maximos(P2)[0], esperado)


def test_bloques_de_dos_anios_con_numero_impar():
    """§5: con un número impar de años, el último bloque de cada lado es de un año."""
    lado = np.array([0] * 13 + [1] * 34)
    u = P.unidades(lado, "bloques2")
    lado0 = [x for x in u if lado[x[0]] == 0]
    assert [len(x) for x in lado0] == [2] * 6 + [1]
    assert all(len(x) == 2 for x in u if lado[x[0]] == 1)
    assert all(len(set(lado[x])) == 1 for x in u)


# --- Dependencia entre años -------------------------------------------------------------------


@pytest.mark.lento
def test_ar1_publica_el_error_y_la_recalibracion_lo_lleva_al_5pct():
    """§5, dependencia entre años: error de tipo I con AR(1) y valor crítico recalibrado en [4 %, 6 %]."""
    c = contraste_sintetico(np.random.default_rng(8))

    def nula(esquema, etiqueta, rho=0.3):
        return F.simular_A(c, esquema, 99, 600, 0.0, 1, True, rho, ("prueba", "ar1", etiqueta, esquema))

    regla = F.aplicar_regla(nula, 0.05, 0.07)
    assert "anios" in regla.tipo_I and 0 <= regla.tipo_I["anios"] <= 1
    # recalibración directa con dependencia fuerte, verificada con réplicas independientes
    p_cal = F.simular_A(c, "anios", 999, 1500, 0.0, 1, True, 0.45, ("prueba", "rho045", "cal"))
    p_ver = F.simular_A(c, "anios", 999, 1500, 0.0, 1, True, 0.45, ("prueba", "rho045", "ver"))
    assert np.mean(p_cal <= 0.05) > 0.07
    crit = P.valor_critico_recalibrado(p_cal, 0.05)
    assert 0.04 <= np.mean(p_ver <= crit) <= 0.06


def test_orden_ar1_conserva_los_anios_y_da_autocorrelacion():
    rng = np.random.default_rng(9)
    clave = rng.normal(size=47)
    rs = []
    for _ in range(400):
        o = P.orden_ar1(clave, 0.3, rng)
        assert sorted(o) == list(range(47))
        y = clave[o]
        rs.append(np.corrcoef(y[:-1], y[1:])[0, 1])
    assert 0.2 < np.mean(rs) < 0.35


# --- Signo de E1 ------------------------------------------------------------------------------


def _res(cociente_mm_c1, resultado_c1=C.NO_CONCLUYENTE):
    def a(cmm, resu):
        return {
            "cociente": [1.0, 1.0, cmm],
            "resultado": resu,
            "cambia_periodo": resu != C.DESCARTADO,
            "cuerpo_p50": {"cuerpo_estable_y_cola_cambia": False},
        }

    res_A = {"C1": a(cociente_mm_c1, resultado_c1), "C2": a(1.0, C.DESCARTADO)}
    return res_A, {"resultado": C.DESCARTADO}, {}, {"resultado": C.DESCARTADO}, {"resultado": C.DESCARTADO}


def test_signo_de_E1_elige_la_forma():
    """§5, signo de E1: T0 por debajo → recorte; T0 por encima → escalón al nivel de T0."""
    r = run.decision(*_res(1.05))
    e1 = [e for e in r["escenarios_hs"] if e["nombre"] == "E1"][0]
    assert e1["forma"] == "recorte" and e1["periodo"][0] == "1993-02-01"
    r = run.decision(*_res(0.95))
    e1 = [e for e in r["escenarios_hs"] if e["nombre"] == "E1"][0]
    assert e1["forma"] == "escalon_al_nivel_T0" and e1["periodo"][0] == "1980-01-01"
    r = run.decision(*_res(1.05, C.DESCARTADO))
    assert [e["nombre"] for e in r["escenarios_hs"]] == ["E0"] and r["n_contornos"] == 1


def test_maximo_de_cinco_contornos():
    res_A, res_B, res_C, res_D, res_alt = _res(1.05)
    res_A["C2"]["resultado"], res_A["C2"]["cambia_periodo"] = C.NO_CONCLUYENTE, True
    res_D["resultado"] = C.NO_CONCLUYENTE
    r = run.decision(res_A, res_B, res_C, res_D, res_alt)
    assert r["n_escenarios_hs"] == 3 and r["n_contornos"] == 5


# --- Potencia con años reasignados ------------------------------------------------------------


@pytest.mark.lento
def test_potencia_casi_no_cambia_con_un_escalon_real():
    """§5, inyección: con un escalón real de hasta δ_rel, la potencia (años reasignados) cambia menos que el error
    de Monte Carlo (tres errores estándar). Años con variabilidad entre años del 10 %, como un registro real."""
    anios = list(range(2000, 2030))
    fy = dict(zip(anios, np.exp(np.random.default_rng(20).normal(0, 0.10, 30)), strict=True))
    l1, l2 = ("2000-01-01", "2009-12-31"), ("2010-01-01", "2029-12-31")
    base = serie_diaria(anios, np.random.default_rng(10), factor=fy)
    paso = pd.Series(np.where(base.index.year >= 2010, 1.03, 1.0), index=base.index)
    c0 = F.preparar_contraste("s0", base, l1, l2, 1)
    c1 = F.preparar_contraste("s1", base * paso, l1, l2, 1)
    n = 400
    p0 = np.mean(F.simular_A(c0, "anios", 99, n, 0.12, 1, False, 0.3, ("prueba", "pot")) <= 0.05)
    p1 = np.mean(F.simular_A(c1, "anios", 99, n, 0.12, 1, False, 0.3, ("prueba", "pot")) <= 0.05)
    se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
    assert 0.2 < p0 < 0.95
    assert abs(p1 - p0) < 3 * se


# --- δ_rel -------------------------------------------------------------------------------------


def test_delta_rel_en_la_rejilla_analitica():
    """§5, δ_rel: con RL(δ) = (1 + δ)·RL(0), la raíz es exactamente 0.02."""
    deltas = R.rejilla(-0.3, 0.3, 0.005)
    r = R.raiz_en_rejilla(lambda d: np.array([10.0 * (1 + d)]), deltas, 0.02)
    assert r is not None and abs(r[0]) == pytest.approx(0.02)


def test_delta_rel_con_gpd_conocida_y_lado_sin_cola():
    rng = np.random.default_rng(11)
    x = rng.pareto(1 / 0.1, 4000) * 0.9  # GPD con ξ = 0.1, σ = 0.9
    u = 0.0  # con u = 0, escalar los picos escala los excesos: RL(δ) = (1 + δ)·RL(0) y la raíz es 0.02
    t = pd.date_range("2000-01-01", periods=x.size, freq="5D")
    m = R.ModeloProvisional(u, u + x, t, anios=x.size * 5 / 365.25)
    lados = {"todos": np.ones(x.size, bool), "vacio": np.zeros(x.size, bool)}
    d = R.delta_relevante(m, lados, (10.0, 50.0, 100.0), 0.02, R.rejilla(-0.3, 0.3, 0.005))
    assert d.lado == "todos" and abs(abs(d.delta) - 0.02) <= 0.005 + 1e-9
    assert abs(abs(d.cambio) - abs(d.delta)) < 1e-4
    assert d.por_lado["vacio"] is None
    d2 = R.delta_relevante(m, {"vacio": lados["vacio"]}, (100.0,), 0.02, R.rejilla(-0.3, 0.3, 0.005))
    assert d2.sin_raiz


# --- Mann-Kendall con Hamed-Rao, Pettitt y SNHT ----------------------------------------------


@pytest.mark.lento
def test_mann_kendall_hamed_rao_n23():
    """§5: AR(1) sin tendencia con n = 23: rechazo en [3 %, 7 %]; la versión sin corrección rechaza más."""
    rng = np.random.default_rng(12)
    ys = [P.ar1(23, C.RHO_AR1, rng) for _ in range(2000)]
    corr = np.mean([T.mann_kendall(y).p <= 0.05 for y in ys])
    sin = np.mean([T.mann_kendall(y, corregir=False).p <= 0.05 for y in ys])
    assert 0.03 <= corr <= 0.07, f"Hamed-Rao rechaza {corr:.3f}"
    assert sin > corr


def test_mann_kendall_detecta_tendencia_y_sen_la_estima():
    y = 0.5 * np.arange(23) + np.random.default_rng(13).normal(0, 1, 23)
    r = T.mann_kendall(y)
    assert r.p < 0.01 and abs(r.pendiente - 0.5) < 0.1


@pytest.mark.parametrize("fn", [T.pettitt, T.snht])
def test_punto_de_cambio_en_su_posicion(fn):
    rng = np.random.default_rng(14)
    for pos in (10, 20, 33):
        y = rng.normal(0, 1, 46)
        y[pos:] += 3.0
        assert abs(fn(y).posicion - pos) <= 1


# --- Exposición a altímetros ----------------------------------------------------------------


def test_exposicion_ventanas_distancia_tierra_y_completitud():
    """§5, exposición: ventana correcta; > 200 km o sobre tierra no cuentan; asimilada sin datos locales →
    incompleta; nada después del último dato local cuenta como «sin pasada» (queda incompleto)."""
    lat0, lon0 = C.NODO
    med = pd.DataFrame(
        {
            "mision": ["JASON-1", "JASON-1", "JASON-1", "JASON-1", "HY-2"],
            "tiempo": pd.to_datetime(
                [
                    "2005-01-01 01:29:00",
                    "2005-01-01 01:29:30",
                    "2005-01-02 00:00:00",
                    "2005-01-03 00:00:00",
                    "2005-01-01 06:00:00",
                ]
            ),
            "lat": [lat0, lat0 + 0.1, lat0 + 2.5, lat0, lat0],  # la tercera, a unos 280 km
            "lon": [lon0, lon0, lon0, lon0, lon0],
            "prof_archivo_m": [-200.0, -200.0, -200.0, 50.0, -200.0],  # la cuarta, sobre tierra
        }
    )
    pas = A.pasadas(med, lat0, lon0, 200.0, 60.0)
    assert len(pas) == 2  # una de Jason-1 (dos mediciones a 30 s) y una de HY-2
    rejilla = pd.date_range("2005-01-01", "2005-01-03 21:00", freq="3h")
    expo = A.exposicion(rejilla, pas, run.VENTANAS)
    c = expo["c:JASON-1"]
    assert c[pd.Timestamp("2005-01-01 00:00")] and not c[pd.Timestamp("2005-01-01 03:00")]  # 01:29 en [−1.5, 1.5)
    assert not expo["r:JASON-1"][pd.Timestamp("2005-01-01 00:00")]
    assert expo["r:JASON-1"][pd.Timestamp("2005-01-01 03:00")]  # (00:00, 03:00]
    assert not c[pd.Timestamp("2005-01-02 00:00")] and not c[pd.Timestamp("2005-01-03 00:00")]
    asim = A.con_pasada_asimilada(expo, "c", C.ASIMILADAS, C.NO_VERIFICABLES)
    assert asim[0] and not expo["c:HY-2"].iloc[0] and expo["c:HY-2"].iloc[2]
    assert not asim[2]  # HY-2 a las 06:00 no es asimilada
    t = pd.date_range("2016-02-28", "2016-03-02", freq="3h")
    ok = A.completa(t, (), {"SAR": "2016-03-01"}, pd.Timestamp("2016-03-05"))
    assert ok[t < "2016-03-01"].all() and not ok[t >= "2016-03-01"].any()
    ok = A.completa(t, (), {}, pd.Timestamp("2016-02-29 12:00"))
    assert ok[t <= "2016-02-29 12:00"].all() and not ok[t > "2016-02-29 12:00"].any()
    ok = A.completa(t, (("2016-02-29", "2016-02-29", "x"),), {}, pd.Timestamp("2017-01-01"))
    assert not ok[(t >= "2016-02-29") & (t < "2016-03-01")].any() and ok[t >= "2016-03-01"].all()


# --- Mantel-Haenszel ----------------------------------------------------------------------------


def _mh_sintetico(rr, oculta=0.0, n=400_000, semilla=15):
    rng = np.random.default_rng(semilla)
    x = rng.random(n) < 0.2
    est = rng.integers(0, 8, n)
    base = 0.01 * (1 + est / 4)
    ev = rng.random(n) < np.where(x, base * rr, base)
    visto = x & (rng.random(n) >= oculta)
    return A.mantel_haenszel(ev, visto, est)


def test_mantel_haenszel_recupera_el_cociente_y_la_ocultacion_lo_baja():
    """§5, MH: tasa doble → ≈ 2; sin dependencia → ≈ 1; con un tercio oculto el cociente baja hacia 1."""
    r2 = _mh_sintetico(2.0)
    assert abs(r2.cociente - 2.0) < 0.1 and r2.lo > 1
    r1 = _mh_sintetico(1.0)
    assert abs(r1.cociente - 1.0) < 0.06 and r1.lo < 1 < r1.hi
    r_o = _mh_sintetico(2.0, oculta=1 / 3)
    # con un tercio oculto, la tasa «sin pasada» sube a (0.8 + 0.2/3·2)/(0.8 + 0.2/3) de la base: cociente ≈ 1.857
    assert 1.0 < r_o.cociente < r2.cociente and abs(r_o.cociente - 2 / (1 + (0.2 / 3) / (0.8 + 0.2 / 3))) < 0.1


def test_placebo_filtrado_no_hereda_el_efecto():
    """§5: efecto real de 3 y un placebo que coincide con pasadas asimiladas: filtrado ≈ 1, sin filtrar no."""
    rng = np.random.default_rng(16)
    n = 400_000
    asim = rng.random(n) < 0.2
    placebo = np.where(asim, rng.random(n) < 0.5, rng.random(n) < 0.1)
    ev = rng.random(n) < np.where(asim, 0.03, 0.01)
    est = np.zeros(n, int)
    sin_filtro = A.mantel_haenszel(ev, placebo, est)
    filtro = A.mantel_haenszel(ev, placebo, np.where(asim, -1, 0))
    assert sin_filtro.lo > 1.2
    assert filtro.lo < 1 < filtro.hi


# --- Umbral c_T0 ---------------------------------------------------------------------------------


def test_c_T0_da_1pct_en_T0_y_una_tormenta_suave_no_salta():
    """§5: c_T0 da una tasa de saltos del 1 % en T0; una tormenta suave no da saltos relativos y sí absolutos."""
    rng = np.random.default_rng(17)
    t = pd.date_range("1980-01-01", "1999-12-31 21:00", freq="3h")
    hs = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.03, t.size)) * 0.1) + 1.0, index=t)
    c = A.umbral_c_T0(hs, "1980-01-01", "1992-12-31", 0.99)
    s = A.saltos_relativos(hs.to_numpy(), c)
    en_t0 = np.asarray(t <= "1992-12-31 21:00")
    assert abs(s[en_t0][1:].mean() - 0.01) < 0.001
    tormenta = 2.0 * 1.05 ** np.arange(24)  # crece un 5 % cada 3 h, de 2 a 6.1 m
    assert not A.saltos_relativos(tormenta, max(c, 0.06)).any()
    assert A.saltos_absolutos(tormenta, 0.20).any()


# --- Familia D ------------------------------------------------------------------------------------


def _eventos(rng, desplazamiento, n_anios=(23, 10)):
    anios = np.arange(sum(n_anios))
    lado = np.array([0] * n_anios[0] + [1] * n_anios[1])
    pos = np.repeat(anios, 4)
    tp = rng.normal(14, 2, pos.size) + desplazamiento * lado[pos]
    return F.EventosTp(tp, pos, lado, anios)


def test_familia_D_recupera_el_desplazamiento_y_mantiene_el_nivel():
    """§5, familia D: recupera un desplazamiento conocido; sin desplazamiento, rechazo en [3 %, 7 %]."""
    rng = np.random.default_rng(18)
    ev = _eventos(rng, 3.0)
    r = F.prueba_D(ev, ev.tp, np.arange(ev.anios.size), "anios", 499, np.random.default_rng(0))
    assert abs(r.diferencia - 3.0) < 0.8 and r.p < 0.01
    ev0 = _eventos(rng, 0.0)
    p = F.simular_D(ev0, "anios", 99, 1000, 0.0, False, 0.3, ("prueba", "D"))
    assert 0.03 <= np.mean(p <= 0.05) <= 0.07


# --- Placebos --------------------------------------------------------------------------------------


def test_calendario_desplazado_conserva_pasadas_por_hora():
    """§5, placebo desplazado: conserva el número de pasadas por estrato (hora sinóptica)."""
    t = pd.date_range("1980-01-01", "2003-12-31 21:00", freq="3h")
    rng = np.random.default_rng(19)
    x = pd.Series(rng.random(t.size) < 0.1, index=t)
    d = A.calendario_desplazado(x, "1993-02-01", "2003-06-16", 13, t)
    orig = x.loc["1993-02-01":"2003-06-16 21:00"]
    dd = pd.Series(d, index=t).loc["1980-02-01":"1990-06-16 21:00"]
    o = orig.groupby(orig.index.hour).sum()
    n = dd.groupby(dd.index.hour).sum()
    # solo pueden faltar los 29 de febrero sin destino
    feb29 = orig[(orig.index.month == 2) & (orig.index.day == 29)]
    assert ((o - n) <= feb29.groupby(feb29.index.hour).sum().reindex(o.index, fill_value=0)).all()
    assert (o >= n).all() and (n > 0).all()


def test_clases_de_placebo_y_bonferroni():
    """§5, clases: sin pasos → «sin pasos»; poca potencia → «no informativo»; m bien contado."""
    pot_base = {"a": 0.95, "b": 0.85, "c": 0.3}
    niveles = []

    def potencia(n, m):
        niveles.append(m)
        return pot_base[n] - 0.02 * (m - 1)  # más placebos → nivel más estricto → menos potencia

    cl = A.clasificar_placebos(["a", "b", "c", "d"], ["a", "b", "c"], potencia, 0.05, 0.8)
    assert cl.clase == {"a": "cuenta", "b": "cuenta", "c": "no informativo", "d": "sin pasos"}
    assert cl.m == 2 and 3 in niveles and niveles[-1] == 2
    cl0 = A.clasificar_placebos(["d"], [], potencia, 0.05, 0.8)
    assert cl0.m == 0 and cl0.clase == {"d": "sin pasos"}


# --- Orden potencia → resultado -------------------------------------------------------------------


def test_orden_exige_potencia_antes_de_los_resultados(tmp_path):
    """§5: la etapa falla si se pide un estadístico antes de que potencia.csv exista."""
    o = run.OrdenEscritura(tmp_path)
    with pytest.raises(run.ErrorOrden):
        o.exigir_potencia()
    (tmp_path / "potencia.csv").write_text("x\n")
    with pytest.raises(run.ErrorOrden):  # existe pero no se registró en orden
        o.exigir_potencia()
    o.registrar("potencia.csv")
    o.exigir_potencia()
