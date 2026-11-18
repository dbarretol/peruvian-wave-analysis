"""Etapa 1 de la F5, ``validacion-modelo`` (contrato v6 §1, §2 y §8), con datos sintéticos del modelo y sin red."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from peruvian_wave_analysis import manifiesto
from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.comun.errores import GateFallido
from peruvian_wave_analysis.datos import detector as D
from peruvian_wave_analysis.datos.run import ruta_candidatos, ruta_serie
from peruvian_wave_analysis.validacion import modelo, ndbc

NOMBRE = "sitio-prueba"
ANIOS = (1986, 1987)
T = pd.date_range("1986-01-01", "1987-12-31 21:00", freq="3h", name="tiempo")
LATS = np.round(np.arange(-15.4, -14.59, 0.2), 4)
LONS = np.round(np.arange(-76.2, -75.39, 0.2), 4)
TIERRA = (-14.6, -75.4)
BOYAS = (
    ndbc.Boya("90001", -18.03, -85.08, ANIOS, "boya sintética 1"),
    ndbc.Boya("90002", -19.43, -85.07, ANIOS, "boya sintética 2"),
)


def _hs(n, fase=0.0):
    t = np.arange(n)
    return np.round(1.8 + 0.6 * np.sin(2 * np.pi * t / 2920 + fase) + 0.2 * np.sin(2 * np.pi * t / 37), 2)


def _escribir_nc(ruta, datos, lats, lons, t):
    ds = xr.Dataset(
        {v: (("time", "latitude", "longitude"), a) for v, a in datos.items()},
        coords={"time": pd.DatetimeIndex(t).rename(None), "latitude": lats, "longitude": lons},
    )
    for v in datos:
        ds[v].attrs["units"] = "m" if v == "VHM0" else "s"
    enc = {v: {"dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0, "_FillValue": -32767} for v in datos}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(ruta, encoding=enc)


def _publicar(raiz, salida, archivos: dict, codigo):
    with etapa.staging_protegido(salida, [codigo], base_codigo=codigo.parent) as st:
        for nombre, escribir in archivos.items():
            escribir(st / nombre)
        etapa.publicar(st, salida, etapa.Gate(True, {}), [codigo], {}, raiz, None, base_codigo=codigo.parent)


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    """WAVE-DATA sintético: región, celdas de las boyas y salidas publicadas de la F3."""
    monkeypatch.setattr(ndbc, "BOYAS", BOYAS)
    raiz = tmp_path / "WAVE-DATA"
    raw = raiz / "raw"
    codigo = tmp_path / "codigo.py"
    codigo.write_text("x = 1\n")
    n = T.size
    reg = np.empty((n, LATS.size, LONS.size))
    for i in range(LATS.size):
        for j in range(LONS.size):
            reg[:, i, j] = _hs(n, fase=0.1 * i + 0.05 * j)
    i_t, j_t = int(np.argmin(abs(LATS - TIERRA[0]))), int(np.argmin(abs(LONS - TIERRA[1])))
    reg[:, i_t, j_t] = np.nan
    # celda (-15.4, -76.2): un registro fuera de límites, 60 registros acumulados en 3.33 m y un pico aislado
    reg[100, 0, 0] = 25.0
    reg[200:260, 0, 0] = 3.33
    reg[1000, 0, 0] = reg[999, 0, 0] + 6.0
    for anio in ANIOS:
        sel = T.year == anio
        r = raw / "waverys" / "regional" / f"waverys_regional_ds_202411_{anio}.nc"
        _escribir_nc(r, {"VHM0": reg[sel]}, LATS, LONS, T[sel])
        manifiesto.registrar(raw, r, {"fuente": "sintético"})
    for k, b in enumerate(BOYAS):
        la = np.round(np.arange(round(b.lat) - 0.4, round(b.lat) + 0.41, 0.2), 4)
        lo = np.round(np.arange(round(b.lon) - 0.4, round(b.lon) + 0.41, 0.2), 4)
        forma = (n, la.size, lo.size)
        hs = np.broadcast_to(_hs(n, fase=k)[:, None, None], forma).copy()
        tp = np.broadcast_to(np.round(9 + 3 * np.sin(np.arange(n) / 50.0), 2)[:, None, None], forma).copy()
        tm = np.round(tp * 0.7, 2)
        tp[[10, 11, 12]] = 28.6  # saturados (continuo: ≥ 1/0.035 − 0.05 s)
        for anio in ANIOS:
            sel = T.year == anio
            r = raw / "waverys" / "boyas" / f"waverys_boya_{b.estacion}_ds_202411_{anio}.nc"
            _escribir_nc(r, {"VHM0": hs[sel], "VTPK": tp[sel], "VTM02": tm[sel]}, la, lo, T[sel])
            manifiesto.registrar(raw, r, {"fuente": "sintético"})
    nodo = pd.DataFrame({"VHM0": reg[:, 2, 2], "valido": True}, index=T)
    s = D.escala_s(nodo["VHM0"].to_numpy())
    resumen = {"k_desc": 15.5, "k_rev": 4.5, "s_m": s, "acumulaciones": {}}
    cand = pd.DataFrame({"tiempo": [], "regla": [], "decision": []})
    _publicar(
        raiz,
        ruta_candidatos(raiz, NOMBRE),
        {
            "resumen_candidatos.json": lambda p: json_estricto.escribir(p, resumen),
            "candidatos.csv": lambda p: cand.to_csv(p, index=False),
        },
        codigo,
    )
    _publicar(raiz, ruta_serie(raiz, NOMBRE), {"serie_nodo.parquet": lambda p: nodo.to_parquet(p)}, codigo)
    return raiz, reg


def test_etapa_publica_y_aplica_el_control(mundo):
    raiz, reg = mundo
    salida = modelo.modelo(raiz, NOMBRE)
    man = etapa.exigir_publicada(salida)
    assert set(man["salidas"]) == {
        "modelo_region.parquet",
        "marcas_region.parquet",
        "celdas_region.csv",
        "modelo_boyas.parquet",
        "celdas_boyas.csv",
        "resumen_modelo.json",
    }
    res = json_estricto.leer(salida / "resumen_modelo.json")
    assert res["celdas_region"] == LATS.size * LONS.size - 1  # sin la celda de tierra
    assert res["coherencia_nodo_f3"]["max_dif_hs_m"] < 1e-6
    assert res["valores_observados_leidos"] is False
    assert res["reglas_no_aplicadas"]
    celdas = pd.read_csv(salida / "celdas_region.csv").set_index("celda")
    c = celdas.loc["-15.4_-76.2"]
    assert c["fuera_de_limites"] == 1 and c["acumulaciones"] == 60 and str(c["valores_acumulados"]) == "3.33"
    assert c["marcados_detector"] >= 1
    assert celdas.drop("-15.4_-76.2")["excluidos"].sum() == 0
    v = pd.read_parquet(salida / "modelo_region.parquet")
    m = pd.read_parquet(salida / "marcas_region.parquet")
    assert np.isnan(v["-15.4_-76.2"].iloc[100]) and v["-15.4_-76.2"].iloc[200:260].isna().all()
    assert m["-15.4_-76.2"].iloc[1000] and not np.isnan(v["-15.4_-76.2"].iloc[1000])  # marca, no excluye
    assert v["-15.0_-75.8"].to_numpy() == pytest.approx(reg[:, 2, 2], abs=1e-6)
    b = pd.read_parquet(salida / "modelo_boyas.parquet")
    assert set(b["boya"]) == {"90001", "90002"}
    assert b.groupby("boya")["VTPK"].apply(lambda x: x.iloc[10:13].isna().all()).all()
    cb = pd.read_csv(salida / "celdas_boyas.csv", dtype={"boya": str}).set_index("boya")
    assert cb.loc["90001", "celda"] == "-18.0_-85.0" and cb.loc["90001", "VTPK:saturados"] == 3
    assert (cb["dist_boya_km"] < 15).all()


def test_determinismo(mundo):
    raiz, _ = mundo
    s1 = etapa.exigir_publicada(modelo.modelo(raiz, NOMBRE))["salidas"]
    s2 = etapa.exigir_publicada(modelo.modelo(raiz, NOMBRE))["salidas"]
    assert s1 == s2


def test_gate_falla_si_el_nodo_no_coincide_con_la_f3(mundo, tmp_path):
    raiz, reg = mundo
    codigo = tmp_path / "codigo.py"
    nodo = pd.DataFrame({"VHM0": reg[:, 2, 2] + 0.01, "valido": True}, index=T)
    _publicar(raiz, ruta_serie(raiz, NOMBRE), {"serie_nodo.parquet": lambda p: nodo.to_parquet(p)}, codigo)
    with pytest.raises(GateFallido, match="celda del nodo difiere"):
        modelo.modelo(raiz, NOMBRE)
    assert (modelo.ruta_modelo(raiz, NOMBRE) / etapa.GATE_FALLIDO).exists()


def test_raw_alterado_falla(mundo):
    raiz, _ = mundo
    raw = raiz / "raw"
    r = next((raw / "waverys" / "regional").glob("*.nc"))
    r.write_bytes(r.read_bytes() + b"x")
    with pytest.raises(Exception, match="sha256"):
        modelo.modelo(raiz, NOMBRE)
