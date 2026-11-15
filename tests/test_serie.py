"""Serie de WAVERYS (contrato de la F1 §3.5 y gate §4.2) con datos sintéticos y sin red."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from peruvian_wave_analysis import manifiesto
from peruvian_wave_analysis.sitio import serie

LATS = np.round(np.arange(-15.4, -14.59, 0.2), 4)
LONS = np.round(np.arange(-76.2, -75.39, 0.2), 4)
TIERRA = {(-14.6, -75.6), (-14.6, -75.4), (-14.8, -75.4), (-14.8, -75.6)}  # sin dato (como en el bloque real)


def _cobertura(inicio="2019-11-01T00:00:00Z", fin="2021-02-28T21:00:00Z", variables=None):
    return serie.Cobertura(serie.DATASET, "202411", inicio, fin, 3.0, sorted(variables or serie.VARIABLES_ADR_0016),
                           "2026-11-14T00:00:00Z")


def _describe_falso(cob):
    def _f(dataset_id):
        t0 = pd.Timestamp(cob.inicio).value // 10**6
        t1 = pd.Timestamp(cob.fin).value // 10**6
        variables = [{"short_name": v, "coordinates": [{"coordinate_id": "time", "minimum_value": t0,
                                                         "maximum_value": t1, "step": 10800000.0}]}
                     for v in cob.variables]
        return {"products": [{"datasets": [{"dataset_id": dataset_id, "versions": [
            {"label": "202411", "parts": [{"services": [{"service_name": "arco-time-series", "variables": variables}]}]}]}]}]}
    return _f


def _subset_falso(llamadas, faltante_en=None, duplicar=False, sin_sw2=True):
    """Escribe un NetCDF con el formato de copernicusmarine para el rango pedido."""
    def _f(**p):
        llamadas.append(p)
        t = pd.date_range(p["start_datetime"], p["end_datetime"], freq="3h")
        if duplicar:
            t = t.append(t[:1])
        datos = {}
        for v in p["variables"]:
            a = np.full((t.size, LATS.size, LONS.size), 1.0)
            for la, lo in TIERRA:
                a[:, np.isclose(LATS, la), np.isclose(LONS, lo)] = np.nan
            if v == "VHM0_SW2" and sin_sw2:
                a[::2] = np.nan  # la segunda partición falta legítimamente a veces
            if faltante_en and v == faltante_en:
                a[5, np.isclose(LATS, -15.0), np.isclose(LONS, -75.8)] = np.nan
            datos[v] = (("time", "latitude", "longitude"), a.astype("float32"))
        ds = xr.Dataset(datos, coords={"time": t, "latitude": LATS, "longitude": LONS})
        ds.to_netcdf(f"{p['output_directory']}/{p['output_filename']}")
    return _f


@pytest.fixture
def mascara():
    v = np.ones((LATS.size, LONS.size), bool)
    for la, lo in TIERRA:
        v[np.isclose(LATS, la), np.isclose(LONS, lo)] = False
    return xr.DataArray(v, coords={"lat": LATS, "lon": LONS}, dims=("lat", "lon"))


def _descargar_todo(tmp_path, monkeypatch, cob, **kw):
    llamadas = []
    monkeypatch.setattr(serie, "_subset", _subset_falso(llamadas, **kw))
    cfg = serie.ConfigSerie()
    raiz_raw = tmp_path / "raw"
    rutas = [serie.descargar_anio(a, cfg, cob, raiz_raw)[0] for a in serie.anios(cob)]
    return cfg, raiz_raw, rutas, llamadas


def test_catalogo_coincide_y_detecta_cambio_de_variables(monkeypatch):
    cob = _cobertura()
    monkeypatch.setattr(serie, "_describir", _describe_falso(cob))
    leida = serie.consultar_catalogo(serie.ConfigSerie())
    assert leida.inicio == cob.inicio and leida.fin == cob.fin and leida.paso_h == 3.0
    otra = _cobertura(variables=serie.VARIABLES_ADR_0016[:-1])
    monkeypatch.setattr(serie, "_describir", _describe_falso(otra))
    with pytest.raises(serie.ErrorSerie, match="ADR"):
        serie.consultar_catalogo(serie.ConfigSerie())


def test_descarga_por_anio_con_version_y_reanudacion(tmp_path, monkeypatch):
    cob = _cobertura()
    cfg, raiz_raw, rutas, llamadas = _descargar_todo(tmp_path, monkeypatch, cob)
    assert [r.name.split("_")[2] for r in rutas] == ["2019", "2020", "2021"]
    assert llamadas[0]["start_datetime"] == "2019-11-01T00:00:00" and llamadas[-1]["end_datetime"] == "2021-02-28T21:00:00"
    assert all(p["dataset_version"] == "202411" and p["coordinates_selection_method"] == "outside" for p in llamadas)
    entrada = manifiesto.verificar(raiz_raw, rutas[1])
    assert entrada["version_dataset"] == "202411" and entrada["catalogo"]["fin"] == cob.fin
    # Reanudación: nada se vuelve a pedir.
    n = len(llamadas)
    for a in serie.anios(cob):
        serie.descargar_anio(a, cfg, cob, raiz_raw)
    assert len(llamadas) == n


def test_gate_cumple_y_no_falla_por_la_segunda_particion(tmp_path, monkeypatch, mascara):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    inf = serie.verificar(rutas, cob, cfg, raiz_raw, mascara)
    assert inf["cumple"] and inf["pasos"] == inf["pasos_esperados"] == len(pd.date_range(
        "2019-11-01", "2021-02-28T21:00", freq="3h"))
    assert inf["fraccion_sin_dato_en_el_nodo"]["VHM0_SW2"] == pytest.approx(0.5, abs=0.01)
    assert inf["celdas_con_dato"] == 25 - len(TIERRA)


def test_gate_falla_con_faltante_en_obligatoria(tmp_path, monkeypatch, mascara):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob, faltante_en="VTPK")
    with pytest.raises(serie.ErrorSerie, match="VTPK sin dato"):
        serie.verificar(rutas, cob, cfg, raiz_raw, mascara)


def test_gate_falla_con_paso_duplicado(tmp_path, monkeypatch, mascara):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob, duplicar=True)
    with pytest.raises(serie.ErrorSerie, match="duplicados"):
        serie.verificar(rutas, cob, cfg, raiz_raw, mascara)


def test_gate_falla_con_paso_faltante(tmp_path, monkeypatch, mascara):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    with pytest.raises(serie.ErrorSerie, match="faltantes"):
        serie.verificar(rutas[:1] + rutas[2:], cob, cfg, raiz_raw, mascara)


def test_gate_falla_con_mascara_distinta(tmp_path, monkeypatch, mascara):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    otra = mascara.copy()
    otra.loc[dict(lat=-15.4, lon=-76.2)] = False
    with pytest.raises(serie.ErrorSerie, match="máscara"):
        serie.verificar(rutas, cob, cfg, raiz_raw, otra)


def test_gate_falla_con_version_distinta_entre_anios(tmp_path, monkeypatch, mascara):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    entradas = manifiesto.leer(raiz_raw)
    clave = next(k for k in entradas if k.endswith(rutas[1].name))
    entradas[clave]["version_dataset"] = "202311"
    manifiesto._escribir(raiz_raw, entradas)
    with pytest.raises(serie.ErrorSerie, match="versiones"):
        serie.verificar(rutas, cob, cfg, raiz_raw, mascara)


def test_gate_falla_con_mascara_distinta_al_reves(tmp_path, monkeypatch, mascara):
    """F01-K-04: celda sin dato donde la máscara dice mar."""
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    otra = mascara.copy()
    otra.loc[dict(lat=-14.6, lon=-75.6)] = True
    with pytest.raises(serie.ErrorSerie, match="máscara"):
        serie.verificar(rutas, cob, cfg, raiz_raw, otra)


def test_reanudacion_detecta_archivo_alterado(tmp_path, monkeypatch):
    """F01-K-04: un año ya descargado cuyo hash no coincide no se reutiliza en silencio."""
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    with open(rutas[1], "ab") as f:
        f.write(b"\0")
    with pytest.raises(manifiesto.ErrorManifiesto, match="hash"):
        serie.descargar_anio(2020, cfg, cob, raiz_raw)


def test_cobertura_del_manifiesto_y_correspondencia(tmp_path, monkeypatch, mascara):
    """F01-K-03: el gate usa el catálogo del día de la descarga y su informe se ata a los hashes."""
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    c = serie.cobertura_de_manifiesto(rutas, raiz_raw)
    assert (c.inicio, c.fin, c.version) == (cob.inicio, cob.fin, cob.version)
    inf = serie.verificar(rutas, c, cfg, raiz_raw, mascara)
    assert len(inf["archivos_sha256"]) == 3 and serie.corresponde(inf, raiz_raw) == []
    # El catálogo de hoy creció (M−2): el gate con la cobertura anotada sigue cumpliendo.
    assert serie.verificar(rutas, serie.cobertura_de_manifiesto(rutas, raiz_raw), cfg, raiz_raw, mascara)["cumple"]
    # Un archivo cambiado después del gate rompe la correspondencia.
    entradas = manifiesto.leer(raiz_raw)
    clave = next(k for k in entradas if k.endswith(rutas[0].name))
    entradas[clave]["sha256"] = "0" * 64
    manifiesto._escribir(raiz_raw, entradas)
    assert any("hash cambió" in f for f in serie.corresponde(inf, raiz_raw))
    assert serie.corresponde({}, raiz_raw)


def test_cobertura_distinta_entre_anios_falla(tmp_path, monkeypatch):
    cob = _cobertura()
    cfg, raiz_raw, rutas, _ = _descargar_todo(tmp_path, monkeypatch, cob)
    entradas = manifiesto.leer(raiz_raw)
    clave = next(k for k in entradas if k.endswith(rutas[2].name))
    entradas[clave]["catalogo"]["fin"] = "2021-04-30T21:00:00Z"
    manifiesto._escribir(raiz_raw, entradas)
    with pytest.raises(serie.ErrorSerie, match="coberturas distintas"):
        serie.cobertura_de_manifiesto(rutas, raiz_raw)
