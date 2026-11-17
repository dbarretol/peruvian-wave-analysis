"""Descargas de la F5 (contrato v6 §1 y §8): boyas NDBC y WAVERYS regional y en las boyas. Sin red.

Los archivos de NDBC son **sintéticos**, uno por formato (contrato §8; G4). Ninguna prueba abre un archivo real.
Los valores sintéticos llevan cifras reconocibles (7.77, 13.31, 8.88) para comprobar que no aparecen en las salidas
de ``estructura``, ni en los mensajes de error, ni en lo que imprime la línea de comandos.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from peruvian_wave_analysis import manifiesto
from peruvian_wave_analysis.sitio import serie
from peruvian_wave_analysis.sitio.exposicion import distancia_km
from peruvian_wave_analysis.validacion import cli, ndbc, waverys

VALORES = ("7.77", "13.31", "8.88")

# --- Archivos sintéticos de NDBC, uno por formato ---------------------------------------------------

YY_SIN_MIN = """YY MM DD hh WD   WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  WTMP  DEWP  VIS
86 01 01 00 140  6.2  7.6  7.77 13.31  8.88 999 1012.3  21.9  22.4 999.0 99.0
86 01 01 01 140  6.2  7.6 99.00 99.00 99.00 999 1012.3  21.9  22.4 999.0 99.0
86 12 31 23 140  6.2  7.6  2.10 12.50  7.60 999 1012.3  21.9  22.4 999.0 99.0
"""

YYYY_SIN_MIN = """YYYY MM DD hh WD  WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  WTMP  DEWP  VIS  TIDE
2003 01 01 00 140  6.2  7.6  7.77 13.31  8.88 999 1012.3  21.9  22.4 999.0 99.0 99.00
2003 06 01 00 140  6.2  7.6  2.10 99.00  7.60 999 1012.3  21.9  22.4 999.0 99.0 99.00
"""

YYYY_CON_MIN = """YYYY MM DD hh mm  WD  WSPD GST  WVHT  DPD   APD  MWD  BAR    ATMP  WTMP  DEWP  VIS  TIDE
2006 01 01 00 50 140  6.2  7.6  7.77 13.31  8.88 999 1012.3  21.9  22.4 999.0 99.0 99.00
"""

ACTUAL = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi    ft
2016 01 01 00 50 999 99.0 99.0  7.77 13.31  8.88 999 9999.0 999.0  22.4 999.0 99.0 99.00
2016 01 01 01 50 999 99.0 99.0 99.00 99.00 99.00 999 9999.0 999.0  22.4 999.0 99.0 99.00
2016 12 31 23 50 999 99.0 99.0  2.10 12.50  7.60 999 9999.0 999.0  22.4 999.0 99.0 99.00
2017 01 01 00 50 999 99.0 99.0  2.10 12.50  7.60 999 9999.0 999.0  22.4 999.0 99.0 99.00
"""


def _gz(ruta, texto):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wt", encoding="ascii") as f:
        f.write(texto)
    return ruta


def _sin_valores(texto):
    return not any(v in str(texto) for v in VALORES)


@pytest.mark.parametrize(
    ("texto", "formato", "anio", "registros", "inicio", "fin"),
    [
        (YY_SIN_MIN, "YY sin minutos", 1986, 3, "1986-01-01 00:00:00", "1986-12-31 23:00:00"),
        (YYYY_SIN_MIN, "YYYY sin minutos", 2003, 2, "2003-01-01 00:00:00", "2003-06-01 00:00:00"),
        (YYYY_CON_MIN, "YYYY con minutos", 2006, 1, "2006-01-01 00:50:00", "2006-01-01 00:50:00"),
        (ACTUAL, "#YY con minutos y unidades", 2016, 4, "2016-01-01 00:50:00", "2017-01-01 00:50:00"),
    ],
)
def test_estructura_de_cada_formato_sin_valores(tmp_path, texto, formato, anio, registros, inicio, fin):
    if formato.startswith("#YY"):
        # el 1 de enero siguiente a las 00:50 no es «00 h»: se admite solo exactamente a las 00:00
        with pytest.raises(ndbc.ErrorNDBC, match="fuera del año"):
            ndbc.estructura(_gz(tmp_path / f"x{anio}.txt.gz", texto), anio_esperado=anio)
        texto = texto.replace("2017 01 01 00 50", "2017 01 01 00 00")
        fin = "2017-01-01 00:00:00"
    e = ndbc.estructura(_gz(tmp_path / f"b{anio}.txt.gz", texto), anio_esperado=anio)
    assert e["formato"] == formato
    assert e["registros"] == registros
    assert (e["inicio"], e["fin"]) == (inicio, fin)
    assert {"WVHT", "DPD", "APD", "WDIR", "PRES"} <= set(e["columnas"])
    assert not {"WD", "BAR", "YY", "YYYY", "MM", "mm"} & set(e["columnas"])
    assert _sin_valores(e)


def test_lector_interpreta_valores_y_centinelas(tmp_path):
    df, _ = ndbc._leer(_gz(tmp_path / "a.txt.gz", YY_SIN_MIN))
    assert df["tiempo"].iloc[0] == pd.Timestamp("1986-01-01 00:00")  # 86 → 1986
    assert df["WVHT"].iloc[0] == pytest.approx(7.77)
    assert df["DPD"].iloc[0] == pytest.approx(13.31)
    assert df["APD"].iloc[0] == pytest.approx(8.88)
    assert df.loc[1, ["WVHT", "DPD", "APD"]].isna().all()  # 99.00
    assert df["MWD"].isna().all()  # 999
    df, _ = ndbc._leer(_gz(tmp_path / "b.txt.gz", ACTUAL))
    assert df["PRES"].isna().all()  # 9999.0
    assert df["tiempo"].iloc[0] == pd.Timestamp("2016-01-01 00:50")


def test_presencia_devuelve_solo_mascara(tmp_path):
    p = ndbc.presencia(_gz(tmp_path / "a.txt.gz", YYYY_SIN_MIN))
    assert list(p.columns) == ["tiempo", "WVHT", "DPD", "APD"]
    assert all(p[v].dtype == bool for v in ("WVHT", "DPD", "APD"))
    assert p["WVHT"].tolist() == [True, True]
    assert p["DPD"].tolist() == [True, False]
    assert _sin_valores(p.to_string())


@pytest.mark.parametrize(
    ("texto", "mensaje"),
    [
        (YY_SIN_MIN.replace(" 7.77 ", " 7.77x "), "no numéricos"),
        (YY_SIN_MIN.replace(" 7.77 13.31", " 7.77"), "número de campos"),
        ("FOO BAR\n1 2\n", "encabezado no reconocido"),
        (YY_SIN_MIN.replace("WVHT", "WVHX"), "faltan las columnas"),
        ("", "vacío"),
    ],
)
def test_errores_sin_citar_valores(tmp_path, texto, mensaje):
    with pytest.raises(ndbc.ErrorNDBC, match=mensaje) as exc:
        ndbc.estructura(_gz(tmp_path / "e.txt.gz", texto))
    assert _sin_valores(str(exc.value))
    assert exc.value.__cause__ is None


def test_anio_ajeno_falla(tmp_path):
    with pytest.raises(ndbc.ErrorNDBC, match="fuera del año 1987"):
        ndbc.estructura(_gz(tmp_path / "a.txt.gz", YY_SIN_MIN), anio_esperado=1987)


# --- Descarga de NDBC -----------------------------------------------------------------------------


def _bajar_falso(llamadas, textos):
    def _f(url):
        llamadas.append(url)
        anio = int(url.split("h")[-1][:4])
        if anio not in textos:
            return None
        return gzip.compress(textos[anio].encode("ascii"))

    return _f


def test_descarga_ndbc_registra_reanuda_y_anota_faltantes(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(ndbc, "_bajar", _bajar_falso(llamadas, {1986: YY_SIN_MIN}))
    raiz_raw = tmp_path / "raw"
    boya = ndbc.BOYAS[0]
    rutas, faltan = ndbc.descargar(raiz_raw, boya, [1986, 1987])
    assert [r.name for r in rutas] == ["32302h1986.txt.gz"] and faltan == [1987]
    assert llamadas[0] == "https://www.ndbc.noaa.gov/data/historical/stdmet/32302h1986.txt.gz"
    e = manifiesto.verificar(raiz_raw, rutas[0])
    assert e["estacion"] == "32302" and e["anio"] == 1986 and e["formato"] == "YY sin minutos"
    assert e["servicio"] == llamadas[0]
    assert _sin_valores(e)
    ndbc.descargar(raiz_raw, boya, [1986])  # reanudación: verifica por hash, no pide
    assert len(llamadas) == 2


def test_descarga_ndbc_rechaza_archivo_de_otro_anio(tmp_path, monkeypatch):
    monkeypatch.setattr(ndbc, "_bajar", _bajar_falso([], {1987: YY_SIN_MIN}))
    raiz_raw = tmp_path / "raw"
    with pytest.raises(ndbc.ErrorNDBC):
        ndbc.descargar(raiz_raw, ndbc.BOYAS[0], [1987])
    assert not ndbc.ruta_archivo(raiz_raw, "32302", 1987).exists()
    assert manifiesto.leer(raiz_raw) == {}


def test_comparar_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(ndbc, "_bajar", _bajar_falso([], {1986: YY_SIN_MIN, 1988: YY_SIN_MIN.replace("86 ", "88 ")}))
    raiz_raw = tmp_path / "raw"
    rutas, _ = ndbc.descargar(raiz_raw, ndbc.BOYAS[0], [1986, 1988])
    otro = tmp_path / "v2"
    otro.mkdir()
    (otro / rutas[0].name).write_bytes(rutas[0].read_bytes())
    assert ndbc.comparar_hashes(rutas, otro) == {"32302h1986.txt.gz": "igual", "32302h1988.txt.gz": "ausente"}
    (otro / rutas[1].name).write_bytes(b"otro")
    assert ndbc.comparar_hashes(rutas, otro)["32302h1988.txt.gz"] == "distinto"


def test_cli_ndbc_no_imprime_valores(tmp_path, monkeypatch, capsys):
    textos = {1986: YY_SIN_MIN, 2016: ACTUAL.replace("2017 01 01 00 50", "2017 01 01 00 00")}
    monkeypatch.setattr(ndbc, "_bajar", _bajar_falso([], textos))
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    assert cli.main_descarga(["ndbc", "--anio", "1986", "2016", "--datos", str(raiz)]) == 0
    salida = capsys.readouterr()
    assert "32302h1986.txt.gz" in salida.out and "32012h2016.txt.gz" in salida.out
    assert _sin_valores(salida.out) and _sin_valores(salida.err)


# --- WAVERYS ----------------------------------------------------------------------------------------

DS = serie.DATASET
DS_INT = "cmems_mod_glo_wav_myint_0.2deg_PT3H-i"


def test_recuadro_regional_contiene_el_circulo():
    r = waverys.recuadro_regional()
    rumbos = np.radians(np.arange(0, 360, 5))
    # puntos a 200 km en cada rumbo (aproximación local, sobrada para el margen de una celda)
    lat = waverys.NODO[0] + 200 / 111.195 * np.cos(rumbos)
    lon = waverys.NODO[1] + 200 / (111.195 * np.cos(np.radians(lat))) * np.sin(rumbos)
    assert np.all(distancia_km(*waverys.NODO, lat, lon) <= 201.5)
    assert np.all((lat >= r.sur + 0.1) & (lat <= r.norte - 0.1) & (lon >= r.oeste + 0.1) & (lon <= r.este - 0.1))
    assert r.variables == ("VHM0",) and r.anios is None


def test_recuadros_de_las_boyas():
    b = {x.nombre: x for x in waverys.recuadros()}
    assert b["boya_32302"].variables == ("VHM0", "VTPK", "VTM02")
    assert b["boya_32302"].anios == tuple(range(1986, 1996))
    assert b["boya_32012"].anios == tuple(range(2007, 2019))
    assert b["boya_32012"].sur == pytest.approx(-20.425) and b["boya_32012"].oeste == pytest.approx(-86.078)


def _registrar_nodo(raiz_raw, catalogos):
    for i, c in enumerate(catalogos):
        r = raiz_raw / "waverys" / "serie" / f"nodo_{i}.nc"
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_bytes(str(i).encode())
        manifiesto.registrar(raiz_raw, r, {"catalogo": c})


def _cat(ds, version, inicio, fin):
    return {"dataset_id": ds, "version": version, "inicio": inicio, "fin": fin, "paso_h": 3.0, "variables": []}


def test_tramos_del_nodo(tmp_path):
    raiz_raw = tmp_path / "raw"
    with pytest.raises(waverys.ErrorDescarga, match="serie-waverys"):
        waverys.tramos_del_nodo(raiz_raw)
    a = _cat(DS, "202411", "1980-01-01T00:00:00Z", "2023-03-31T21:00:00Z")
    b = _cat(DS_INT, "202501", "2023-04-01T00:00:00Z", "2026-08-31T21:00:00Z")
    _registrar_nodo(raiz_raw, [b, a, a])
    t = waverys.tramos_del_nodo(raiz_raw)
    assert [(x.dataset_id, x.version) for x in t] == [(DS, "202411"), (DS_INT, "202501")]


def test_tramos_solapados_fallan(tmp_path):
    raiz_raw = tmp_path / "raw"
    a = _cat(DS, "202411", "1980-01-01T00:00:00Z", "2023-06-30T21:00:00Z")
    b = _cat(DS_INT, "202501", "2023-04-01T00:00:00Z", "2026-08-31T21:00:00Z")
    _registrar_nodo(raiz_raw, [a, b])
    with pytest.raises(waverys.ErrorDescarga, match="solapados"):
        waverys.tramos_del_nodo(raiz_raw)


def test_pedidos_parten_el_anio_entre_tramos():
    t = [
        waverys.Tramo(DS, "202411", "1980-01-01T00:00:00Z", "2023-03-31T21:00:00Z"),
        waverys.Tramo(DS_INT, "202501", "2023-04-01T00:00:00Z", "2026-08-31T21:00:00Z"),
    ]
    p = waverys.pedidos(waverys.recuadro_regional(), t)
    assert len(p) == 44 + 4  # 1980–2023 y 2023–2026
    de_2023 = [(tr.dataset_id, str(i), str(f)) for a, tr, i, f in p if a == 2023]
    assert de_2023 == [
        (DS, "2023-01-01 00:00:00+00:00", "2023-03-31 21:00:00+00:00"),
        (DS_INT, "2023-04-01 00:00:00+00:00", "2023-12-31 21:00:00+00:00"),
    ]
    assert p[-1][3] == pd.Timestamp("2026-08-31T21:00Z")
    pb = waverys.pedidos(waverys.recuadro_boya(ndbc.BOYAS[0]), t)
    assert [a for a, *_ in pb] == list(range(1986, 1996))


def _describe(version, inicio, fin):
    def _f(dataset_id):
        t0, t1 = (pd.Timestamp(x).value // 10**6 for x in (inicio, fin))
        variables = [
            {
                "short_name": v,
                "coordinates": [
                    {"coordinate_id": "time", "minimum_value": t0, "maximum_value": t1, "step": 10800000.0}
                ],
            }
            for v in serie.VARIABLES_ADR_0016
        ]
        servicio = {"service_name": "arco-time-series", "variables": variables}
        version_ = {"label": version, "parts": [{"services": [servicio]}]}
        return {"products": [{"datasets": [{"dataset_id": dataset_id, "versions": [version_]}]}]}

    return _f


def test_catalogo_de_hoy_debe_cubrir_el_tramo(monkeypatch):
    tr = waverys.Tramo(DS, "202411", "1980-01-01T00:00:00Z", "2026-08-31T21:00:00Z")
    monkeypatch.setattr(serie, "_describir", _describe("202411", "1980-01-01T00:00:00Z", "2026-10-31T21:00:00Z"))
    assert waverys.comprobar_catalogo(tr).version == "202411"
    monkeypatch.setattr(serie, "_describir", _describe("202411", "1980-01-01T00:00:00Z", "2026-06-30T21:00:00Z"))
    with pytest.raises(waverys.ErrorDescarga, match="no cubre"):
        waverys.comprobar_catalogo(tr)
    monkeypatch.setattr(serie, "_describir", _describe("202506", "1980-01-01T00:00:00Z", "2026-10-31T21:00:00Z"))
    with pytest.raises(serie.ErrorSerie, match="versión 202411"):
        waverys.comprobar_catalogo(tr)


def _subset_falso(llamadas, quitar=None, recortar=False):
    def _f(**p):
        llamadas.append(p)
        t = pd.date_range(p["start_datetime"], p["end_datetime"], freq="3h")
        if recortar:
            t = t[:-1]
        lat = np.round(np.arange(np.floor(p["minimum_latitude"] / 0.2) * 0.2, p["maximum_latitude"] + 0.2, 0.2), 4)
        lon = np.round(np.arange(np.floor(p["minimum_longitude"] / 0.2) * 0.2, p["maximum_longitude"] + 0.2, 0.2), 4)
        datos = {
            v: (("time", "latitude", "longitude"), np.ones((t.size, lat.size, lon.size), "float32"))
            for v in p["variables"]
            if v != quitar
        }
        xr.Dataset(datos, coords={"time": t, "latitude": lat, "longitude": lon}).to_netcdf(
            f"{p['output_directory']}/{p['output_filename']}"
        )

    return _f


def _cob(ds=DS):
    return serie.Cobertura(
        ds, "202411", "1980-01-01T00:00:00Z", "2026-10-31T21:00:00Z", 3.0, [], "2026-11-17T00:00:00Z"
    )


def test_descarga_waverys_anota_producto_y_tramo(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(waverys, "_subset", _subset_falso(llamadas))
    raiz_raw = tmp_path / "raw"
    tr = waverys.Tramo(DS_INT, "202411", "2023-04-01T00:00:00Z", "2026-08-31T21:00:00Z")
    rec = waverys.recuadro_boya(ndbc.BOYAS[1])
    (anio, _, ini, fin), *_ = waverys.pedidos(waverys.Recuadro(**{**rec.__dict__, "anios": None}), [tr])
    ruta, e = waverys.descargar(raiz_raw, rec, tr, _cob(DS_INT), anio, ini, fin)
    assert ruta.parent.name == "boyas" and DS_INT in ruta.name and "_202411_2023_" in ruta.name
    assert e["dataset_id"] == DS_INT and e["version_dataset"] == "202411"
    assert e["tramo_serie_nodo"]["inicio"] == "2023-04-01T00:00:00Z"
    assert e["parametros"]["variables"] == ["VHM0", "VTPK", "VTM02"]
    assert e["parametros"]["start_datetime"] == "2023-04-01T00:00:00"
    est = waverys.estructura(ruta)
    assert est["variables"] == ["VHM0", "VTM02", "VTPK"] and est["inicio"] == "2023-04-01 00:00:00"
    waverys.descargar(raiz_raw, rec, tr, _cob(DS_INT), anio, ini, fin)  # reanudación
    assert len(llamadas) == 1


@pytest.mark.parametrize(
    ("kw", "mensaje"), [({"quitar": "VTPK"}, "faltan las variables"), ({"recortar": True}, "pasos")]
)
def test_descarga_waverys_incompleta_no_se_registra(tmp_path, monkeypatch, kw, mensaje):
    monkeypatch.setattr(waverys, "_subset", _subset_falso([], **kw))
    raiz_raw = tmp_path / "raw"
    tr = waverys.Tramo(DS, "202411", "1980-01-01T00:00:00Z", "2026-08-31T21:00:00Z")
    rec = waverys.recuadro_boya(ndbc.BOYAS[0])
    anio, _, ini, fin = waverys.pedidos(rec, [tr])[0]
    with pytest.raises(waverys.ErrorDescarga, match=mensaje):
        waverys.descargar(raiz_raw, rec, tr, _cob(), anio, ini, fin)
    assert not waverys.ruta_archivo(raiz_raw, rec, tr, anio).exists()
    assert manifiesto.leer(raiz_raw) == {}


def test_cli_sin_directorio_de_datos_da_mensaje_claro(monkeypatch, capsys):
    monkeypatch.delenv("WAVE_DATA_DIR", raising=False)
    assert cli.main_descarga(["ndbc"]) == 2
    assert "--datos" in capsys.readouterr().err
