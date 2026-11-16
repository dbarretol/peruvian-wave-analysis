"""F3: detector bilateral, resolución, topes, dipolo, máximos y calibración de k (contrato de la F3, §3.4–§3.10, §5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.datos import calibracion as K
from peruvian_wave_analysis.datos import detector as D


def _base(n=400, semilla=0):
    rng = np.random.default_rng(semilla)
    return 2.0 + np.cumsum(rng.normal(0, 0.005, n))  # fondo suave, s pequeño


# --- Detector -----------------------------------------------------------------------------------


@pytest.mark.parametrize("signo", [+1, -1])
def test_marca_el_pico_no_el_siguiente(signo):
    x = _base()
    s = D.escala_s(x)
    for p in (50, 200, 333):
        x[p] += signo * 1.0
    x[120:122] += signo * 0.8  # pico de dos pasos: el pico es el de mayor salto
    x[121] += signo * 0.3
    exc = D.detectar(x, k=10, s=s)
    assert [e.pico for e in exc] == [50, 121, 200, 333]
    assert [e.duracion for e in exc] == [1, 2, 1, 1]
    assert all(np.sign(e.salto) == signo for e in exc)


def test_tormenta_sostenida_y_llegada_de_swell_no_se_marcan():
    x = _base()
    s = D.escala_s(x)
    sub = 0.5 * (1 - np.cos(np.pi * np.arange(1, 9) / 8))
    x[100:108] += 2.0 * sub  # subida de 24 h
    x[108:200] += 2.0  # nivel sostenido
    # llegada de swell: escalón sostenido (Tp sube antes, Hs después, sin volver)
    x[300:] += 0.6
    assert D.detectar(x, k=10, s=s) == []


def test_excursiones_contiguas_se_unen_y_el_cierre_no_abre_otra():
    x = _base()
    s = D.escala_s(x)
    x[100] += 1.0
    x[101] += 1.5  # contiguo: una sola excursión de dos pasos
    exc = D.detectar(x, k=10, s=s)
    assert len(exc) == 1 and exc[0].inicio == 100 and exc[0].fin == 101 and exc[0].pico == 101


# --- Resolución ---------------------------------------------------------------------------------


def _vecinas(x, n=7, comparten_en=()):
    v = np.repeat(np.asarray(x)[:, None], n, axis=1) * 0 + 2.0  # vecinas planas
    for p, amp in comparten_en:
        v[p, :] += amp
    return v


def _caso(pico_hs=3.0, comparten=False, n_vec=7, tm02_salto=0.0):
    x = _base()
    x[200] = pico_hs
    tm = np.full(x.size, 8.0)
    tm[200] += tm02_salto
    v = _vecinas(x, n_vec, [(200, (pico_hs - 2.0) * 0.5)] if comparten else ())
    exc = D.detectar(x, k=10, s=D.escala_s(x))[0]
    return exc, x, tm, v


def test_resolucion_automatica():
    exc, x, tm, v = _caso()
    assert D.resolver(exc, x, tm, v, p95=10.0) == D.Resolucion(D.DESCARTAR, "reversion_rapida")
    exc, x, tm, v = _caso(tm02_salto=1.5)
    assert D.resolver(exc, x, tm, v, p95=10.0).motivo == "salto_espectral_conjunto"
    exc, x, tm, v = _caso(comparten=True)
    assert D.resolver(exc, x, tm, v, p95=10.0) == D.Resolucion(D.MANUAL, "vecinas_comparten")
    exc, x, tm, v = _caso(n_vec=4)
    assert D.resolver(exc, x, tm, v, p95=10.0) == D.Resolucion(D.MANUAL, "vecinas_insuficientes")


def test_resolucion_por_clase_en_la_cola():
    """v8, §16: mismas definiciones que bajo P95; compartido → mantener dudoso; vecinas insuficientes → mantener."""
    exc, x, tm, v = _caso(pico_hs=3.0)
    assert D.resolver_por_clase(exc, x, tm, v) == D.Resolucion(D.DESCARTAR, "reversion_rapida")
    exc, x, tm, v = _caso(pico_hs=3.0, comparten=True)
    assert D.resolver_por_clase(exc, x, tm, v) == D.Resolucion(D.MANTENER, D.DUDOSO)
    exc, x, tm, v = _caso(pico_hs=3.0, n_vec=4)
    assert D.resolver_por_clase(exc, x, tm, v) == D.Resolucion(D.MANTENER, "vecinas_insuficientes")
    # no depende del P95: el mismo caso da lo mismo que la resolución automática bajo P95
    exc, x, tm, v = _caso()
    assert D.resolver_por_clase(exc, x, tm, v) == D.resolver(exc, x, tm, v, p95=10.0)


def test_proteccion_de_cola_antes_que_el_descarte():
    exc, x, tm, v = _caso(pico_hs=3.0)
    assert D.resolver(exc, x, tm, v, p95=2.9) == D.Resolucion(D.MANUAL, "cola_p95")
    # el P95 es el de la serie antes de los descartes: sembrar valores fuera de límite no lo cambia
    hs = pd.Series(np.linspace(0.5, 5.0, 1000))
    p95 = D.percentil_cola(hs)
    hs2 = hs.copy()
    hs2.iloc[:5] = 25.0  # fuera de límite; la función recibe la serie decodificada, sin descartes
    assert D.percentil_cola(hs2) == pytest.approx(np.nanpercentile(hs2, 95))
    assert p95 == pytest.approx(np.nanpercentile(hs, 95))


def test_evento_sostenido_se_mantiene():
    x = _base()
    x[200:] += 4.0  # escalón de 4 m con un sobrepico de 2 m: revierte 2 m (> k·s), pero no vuelve al 50 %
    x[200] += 2.0
    s = D.escala_s(x)
    exc = D.detectar(x, k=10, s=s)[0]
    tm = np.full(x.size, 8.0)
    assert D.resolver(exc, x, tm, _vecinas(x), p95=50.0) == D.Resolucion(D.MANTENER, "evento_sostenido")


# --- Topes --------------------------------------------------------------------------------------


def _fechas(por_anio):
    out = []
    for anio, n in por_anio.items():
        out += list(pd.date_range(f"{anio}-01-01", periods=n, freq="7D"))
    return pd.DatetimeIndex(out)


def test_topes():
    assert D.topes(_fechas({1990 + i: 1 for i in range(47)}), 100).revision_requerida == []
    assert D.topes(_fechas({1990 + i: 1 for i in range(48)}), 0).revision_requerida
    assert D.topes(_fechas({2000: 5}), 0).revision_requerida == []
    assert D.topes(_fechas({2000: 6}), 0).revision_requerida
    assert D.topes(_fechas({}), 101).revision_requerida


# --- Dipolo -------------------------------------------------------------------------------------


def test_dipolo():
    x = _base()
    s = D.escala_s(x)
    v = np.full((x.size, 7), 2.0)
    x[100] += 1.0  # salto del nodo que las vecinas no tienen
    x[300] += 1.0
    v[300, :] += 1.0  # salto compartido
    assert D.dipolo(x, v, 10, s).tolist() == [100]
    assert D.dipolo(x, v[:, :4], 10, s).tolist() == [100, 300]  # < 5 vecinas: pasa a manual


# --- Máximos ------------------------------------------------------------------------------------


def test_maximos_independientes_antes_de_descartes():
    idx = pd.date_range("2000-01-01", periods=3000, freq="3h")
    hs = pd.Series(1.0, index=idx)
    sembrados = idx[[100, 101, 400, 900, 1500]]
    hs[sembrados] = [5.0, 4.9, 6.0, 25.0, 4.0]  # 101 es de la misma tormenta que 100; 900 viola el límite
    m = D.maximos_independientes(hs, n=4)
    assert list(m.index) == list(idx[[100, 400, 900, 1500]])


# --- Calibración --------------------------------------------------------------------------------


def test_cota_superior_y_eleccion_de_k():
    assert K.cota_superior(0, 300) == pytest.approx(-np.log(0.05) / 300, rel=1e-6)  # ≈ 3/300
    anios = 10_000
    conteos = np.zeros(K.REJILLA_K.size, dtype=int)
    conteos[:10] = 1000
    conteos[10] = 94  # cota al 95 % > 1/100
    conteos[11] = 80  # cota < 1/100
    k = K.elegir_k(conteos, anios)
    assert k == K.REJILLA_K[11]
    assert K.cota_superior(int(conteos[11]), anios) < K.TASA_MAX <= K.cota_superior(int(conteos[10]), anios)
    with pytest.raises(K.CalibracionImposible):
        K.elegir_k(np.full(K.REJILLA_K.size, 500), anios)


def test_escala_minima():
    assert K.exigir_escala(0.05) == pytest.approx(5.0)
    with pytest.raises(K.CalibracionImposible):
        K.exigir_escala(0.019)


def test_calibracion_reproducible_y_k_desc_k_rev():
    a = K.calibrar(cuanto_en_s=0.2, anios=400, con_tormentas=False)  # 0 falsas alarmas en 400 años: cota < 1/100
    b = K.calibrar(cuanto_en_s=0.2, anios=400, con_tormentas=False)
    assert a.conteos == b.conteos and a.k_por_generador == b.k_por_generador
    assert a.k_desc == max(a.k_por_generador.values())
    assert a.k_rev == min(a.k_por_generador[(phi, 10)] for phi in (0.90, 0.98))
    assert a.k_rev <= a.k_desc
    # la cola pesada (ν = 3) exige k mayor que la ligera (ν = 10) con el mismo φ
    assert a.k_por_generador[(0.90, 3)] >= a.k_por_generador[(0.90, 10)]
