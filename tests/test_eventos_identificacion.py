"""F7: identificación de tormentas (§3.2) y conteo previo (§5.0). Contrato de la F7, §8."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import conteo_previo as CP
from peruvian_wave_analysis.eventos import identificacion as ID
from peruvian_wave_analysis.extremos import decluster

from . import test_clima_etapa1 as T1


def _serie(valores, inicio="2000-01-01"):
    t = pd.date_range(inicio, periods=len(valores), freq="3h")
    return pd.Series(np.asarray(valores, dtype=float), index=t)


def _con_crestas(fondo: float, n: int, crestas: dict[int, float], ancho: int = 2) -> np.ndarray:
    """Serie con fondo constante y crestas triangulares (posición → altura) de ``ancho`` pasos por lado."""
    v = np.full(n, fondo)
    for p, h in crestas.items():
        for d in range(-ancho, ancho + 1):
            v[p + d] = max(v[p + d], h - abs(d) * (h - fondo) / (ancho + 1))
    return np.round(v, 2)


# --- Rachas y huecos ------------------------------------------------------------------------------


def test_hueco_dentro_de_una_tormenta_no_la_parte():
    v = np.full(60, 1.0)
    v[20:30] = 3.0
    v[24] = np.nan  # un paso sin dato dentro de la tormenta
    ev = ID.identificar(_serie(v), 2.0, 48.0)
    assert len(ev) == 1 and bool(ev["toca_hueco"].iloc[0])
    assert ev["n_excedencias"].iloc[0] == 9


def test_filas_eliminadas_es_error():
    s = _serie(np.full(20, 3.0)).drop(index=_serie(np.full(20, 3.0)).index[5])
    with pytest.raises(ValueError, match="rejilla completa"):
        ID.identificar(s, 2.0, 48.0)


def test_rachas_coinciden_con_decluster_sin_division():
    rng = np.random.default_rng(5)
    v = np.round(1.5 + rng.gamma(2.0, 0.4, 3000), 2)
    v[rng.choice(3000, 40, replace=False)] = np.nan
    s = _serie(v)
    ev = ID.identificar(s, 2.6, 48.0, bajada=None)
    ref = decluster.decluster_rachas(s.to_frame("Hs"), 2.6, 48.0)
    assert len(ev) == len(ref)
    assert (pd.DatetimeIndex(ev["t_pico"]) == ref.index).all()
    assert (ev["toca_hueco"].to_numpy() == ref["toca_hueco"].to_numpy()).all()
    assert (ev["n_excedencias"].to_numpy() == ref["n_excedencias"].to_numpy()).all()


def test_iet_estricto():
    v = np.full(80, 1.0)
    v[10] = v[26] = 3.0  # separadas 16 pasos = 48 h: no pasan «más de» 48 h → una racha
    v[60] = 3.0  # 34 pasos después → racha nueva
    r = ID.rachas(v, 2.0, 48.0)
    assert list(r.ini) == [10, 60] and list(r.fin) == [26, 60]


# --- Bajada intermedia ----------------------------------------------------------------------------


def test_dos_crestas_separadas_y_con_bajada_dan_dos_eventos():
    # crestas a 20 pasos (60 h ≥ 48 h); mínimo intermedio 2.5 m, crestas 3.5 y 3.2 m: bajadas 1.0 y 0.7 m
    v = _con_crestas(2.5, 80, {20: 3.5, 40: 3.2})
    ev = ID.identificar(_serie(v), 2.0, 48.0)
    assert len(ev) == 2 and ev["dividido_por_bajada"].all() and ev["id_racha"].nunique() == 1
    assert (ev["partes_racha"] == 2).all()
    # bordes: la primera termina en el mínimo y la segunda empieza en el paso siguiente
    assert ev["i_fin"].iloc[0] + 1 == ev["i_inicio"].iloc[1]
    assert v[ev["i_fin"].iloc[0]] == 2.5
    assert list(ev["i_pico"]) == [20, 40]


def test_crestas_cercanas_no_se_parten():
    v = _con_crestas(2.5, 80, {20: 3.5, 30: 3.4})  # 10 pasos = 30 h < 48 h
    assert len(ID.identificar(_serie(v), 2.0, 48.0)) == 1


def test_una_bajada_menor_a_medio_metro_no_parte():
    v = _con_crestas(2.5, 80, {20: 3.5, 40: 2.95})  # la segunda cresta supera el mínimo en 0.45 m
    assert len(ID.identificar(_serie(v), 2.0, 48.0)) == 1


def test_bajada_exacta_de_medio_metro_parte():
    v = _con_crestas(2.5, 80, {20: 3.5, 40: 3.0})  # 0.5 m exactos con la resolución de 0.01 m
    assert len(ID.identificar(_serie(v), 2.0, 48.0)) == 2


def test_tres_crestas_recursion():
    # tres crestas separadas 20 pasos; los mínimos entre ellas: 2.5 m (entre 1 y 2) y 2.2 m (entre 2 y 3)
    v = _con_crestas(2.5, 120, {20: 3.5, 40: 3.3, 60: 3.6})
    v[45:56] = np.minimum(v[45:56], 2.2)
    ev = ID.identificar(_serie(v), 2.0, 48.0)
    assert len(ev) == 3 and list(ev["i_pico"]) == [20, 40, 60]
    assert ev["id_racha"].nunique() == 1 and (ev["partes_racha"] == 3).all()


def test_tres_crestas_la_del_medio_cercana_queda_con_una():
    # la cresta del medio está a 10 pasos de la primera: solo se parte entre la segunda y la tercera
    v = _con_crestas(2.5, 120, {20: 3.5, 30: 3.3, 60: 3.6})
    ev = ID.identificar(_serie(v), 2.0, 48.0)
    assert len(ev) == 2 and list(ev["i_pico"]) == [20, 60]


def test_corte_comprobado_frente_a_la_vecina():
    # el mínimo más profundo está entre las crestas de 20 y 30 pasos (a 30 h): la primera división lo toma, porque
    # mide la cresta de 20 contra la de 60; después, la cresta de 30 no puede quedar sola junto a la de 20
    v = _con_crestas(2.5, 120, {20: 3.5, 30: 3.4, 60: 3.6})
    v[25] = 2.49
    ev = ID.identificar(_serie(v), 2.0, 48.0)
    assert list(ev["i_pico"]) == [20, 60]
    picos = ev["i_pico"].to_numpy()
    assert np.all(np.diff(picos) * 3.0 >= 48.0)  # ningún par de eventos de una racha a menos de IET


def test_eventos_de_una_racha_siempre_a_iet_o_mas():
    rng = np.random.default_rng(8)
    z = np.zeros(6000)
    e = rng.normal(0, 1, z.size)
    for i in range(1, z.size):
        z[i] = 0.95 * z[i - 1] + 0.31 * e[i]
    v = np.round(2.0 * np.exp(0.3 * z), 2)
    ev = ID.identificar(_serie(v), float(np.quantile(v, 0.9)), 48.0)
    assert ev["dividido_por_bajada"].any()
    for _, g in ev.groupby("id_racha"):
        if len(g) < 2:
            continue
        p = g["i_pico"].to_numpy()
        assert np.all(np.diff(p) * 3.0 >= 48.0)
        for k in range(len(g) - 1):
            corte = v[g["i_fin"].iloc[k]]
            assert v[p[k]] - corte >= 0.5 - 1e-9 and v[p[k + 1]] - corte >= 0.5 - 1e-9


def test_nan_no_es_el_minimo_de_un_corte():
    v = _con_crestas(3.1, 80, {20: 3.5, 40: 3.5})  # bajada de 0.4 m en el fondo
    v[30] = np.nan  # el único punto bajo sería el hueco
    assert len(ID.identificar(_serie(v), 2.0, 48.0)) == 1


def test_mejor_corte_toma_la_bajada_mas_profunda():
    x = np.array([3.6, 3.0, 2.0, 3.0, 3.5, 3.0, 2.6, 3.0, 3.6])
    assert ID.mejor_corte(x, sep_pasos=2) == 2


def test_periodo_por_pico():
    v = _con_crestas(1.0, 200, {20: 3.0, 150: 3.0})
    ev = ID.identificar(_serie(v), 2.0, 48.0)
    t = pd.DatetimeIndex(ev["t_pico"])
    assert len(ID.en_periodo(ev, t[1], t[1] + pd.Timedelta(days=1))) == 1


# --- Conteo previo (§5.0) -------------------------------------------------------------------------


def _hs_sintetica():
    return T1._serie()["VHM0"].astype(float)


def test_conteo_solo_conteos_y_tasas():
    hs = _hs_sintetica()
    u = float(np.nanquantile(hs, 0.95))
    c = CP.contar(hs, u, "1993-02-01")
    assert len(c) == 4
    claves = {"rama", "iet_h", "eventos", "rachas", "eventos_de_rachas_divididas", "anios_exposicion",
              "eventos_por_anio"}  # fmt: skip
    for fila in c:
        assert set(fila) == claves  # ninguna magnitud de los picos
    e0_48 = next(f for f in c if f["rama"] == "E0" and f["iet_h"] == 48.0)
    e0_120 = next(f for f in c if f["rama"] == "E0" and f["iet_h"] == 120.0)
    e1_48 = next(f for f in c if f["rama"] == "E1" and f["iet_h"] == 48.0)
    assert e0_120["eventos"] <= e0_48["eventos"] and e1_48["eventos"] < e0_48["eventos"]
    assert abs(e0_48["anios_exposicion"] - 46.0) < 0.05  # 1980–2025 con un hueco corto
    assert abs(e1_48["anios_exposicion"] - (11 / 12 + 32)) < 0.01  # desde el 1993-02-01


def test_regla_de_revision_de_u_id():
    hs = _hs_sintetica()
    u_bajo = float(np.nanquantile(hs, 0.80))  # da muchas más de 20 tormentas por año
    r = CP.conteo_previo(hs, u_bajo, "1993-02-01")
    assert r["tasa_e0_48h"] > CP.TASA_MAX
    assert r["u_id_revisado"] and r["revision"]["cuantil_elegido"] in CP.CUANTILES_REVISION
    assert r["u_id_m"] > u_bajo
    assert CP._tasa_e0_48(r["conteos"]) <= CP.TASA_MAX
    # el primer cuantil que cumple: los anteriores no cumplían
    assert all(i["tasa_e0_48h"] > CP.TASA_MAX for i in r["revision"]["intentos"][:-1])


def test_sin_revision_con_tasa_baja():
    hs = _hs_sintetica()
    r = CP.conteo_previo(hs, float(np.nanquantile(hs, 0.99)), "1993-02-01")
    assert not r["u_id_revisado"] and r["revision"] is None and r["conteos"] == r["conteos_con_u_id_f3"]


def test_n0_de_la_simulacion():
    c = [
        {"rama": "E0", "iet_h": 48.0, "eventos": 400, "anios_exposicion": 46.0},
        {"rama": "E1", "iet_h": 48.0, "eventos": 300, "anios_exposicion": 32.9},
        {"rama": "E0", "iet_h": 120.0, "eventos": 250, "anios_exposicion": 46.0},
        {"rama": "E1", "iet_h": 120.0, "eventos": 180, "anios_exposicion": 32.9},
    ]
    n = CP.n0_simulacion(c)
    assert n["n0"] == [180, 400] and n["anios_simulados"] == [33, 46] and not n["ampliado_por_razon_menor_a_1_5"]
    c[3]["eventos"] = 300
    c[2]["eventos"] = 300
    c[1]["eventos"] = 300
    n = CP.n0_simulacion(c)  # razón 400/300 < 1.5 → {0.8·N_mín, 1.2·N_máx}
    assert n["ampliado_por_razon_menor_a_1_5"] and n["n0"] == [240, 480]


def test_etapa_publica_conteo(tmp_path):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    T1._sembrar(raiz)
    hs = T1._serie()["VHM0"]
    codigo = raiz.parent / "codigo"
    resumen = {"p95_hs_m": float(np.nanquantile(hs, 0.95))}
    T1._publicar(
        raiz,
        CP.ruta_candidatos(raiz, CP.NOMBRE),
        {
            "candidatos.csv": lambda p: pd.DataFrame({"tiempo": ["2017-01-01 00:00"]}).to_csv(p, index=False),
            "resumen_candidatos.json": lambda p: json_estricto.escribir(p, resumen),
        },
        codigo,
    )
    salida = CP.ejecutar(raiz)
    man = etapa.exigir_publicada(salida)
    assert set(man["insumos"]) == {"serie_nodo", "resumen_candidatos", "resumen_f04"}
    r = json_estricto.leer(salida / "conteo_previo.json")
    assert r["valores_de_picos_leidos"] is False and len(r["conteos"]) == 4
    assert r["n0_simulacion"]["n0"][0] <= r["n0_simulacion"]["n0"][1]
