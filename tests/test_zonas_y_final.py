"""Zonas (ADR-0010, ADR-0012), mapa exploratorio con zonas y mapa definitivo (ADR-0011)."""
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from peruvian_wave_analysis import manifiesto
from peruvian_wave_analysis.sitio import batimetria, candidatos, gebco, generar_mapa, waverys, zonas
from peruvian_wave_analysis.sitio.cli import main, main_final, main_registrar_gebco
from peruvian_wave_analysis.sitio.final import ConfigFinal, ErrorGate, generar_mapa_final

from .conftest import LAT0


def _fila(df, lat, lon):
    f = df[np.isclose(df["lat"], lat) & np.isclose(df["lon"], lon)]
    assert len(f) == 1
    return f.iloc[0]


# --- Clasificación (ADR-0012) -----------------------------------------------------------------

@pytest.mark.parametrize("cate, clase", [
    ("Parque Nacional", zonas.EXCLUSION), ("Santuario Nacional", zonas.EXCLUSION),
    ("Santuario Histórico", zonas.EXCLUSION), ("Reserva Nacional", zonas.COLISION),
    ("Reserva Paisajística", zonas.COLISION),
])
def test_clasificacion_anp(cate, clase):
    assert zonas._anp_definitiva({"anp_cate": cate}) == clase


def test_zonificacion_solo_proteccion_estricta():
    assert zonas._zonificacion({"tz_cod": "PE"}) == zonas.EXCLUSION
    assert zonas._zonificacion({"tz_cod": "AD"}) == zonas.IGNORADA


# --- Descarga y registro --------------------------------------------------------------------

def test_capas_se_registran_y_la_que_falla_queda_no_evaluada(cfg_sintetica):
    raiz = Path(cfg_sintetica.directorio_datos)
    z = zonas.obtener(batimetria.recuadro_con_margen(cfg_sintetica), raiz)
    assert z.no_evaluadas == ["Propuestas de ANP"]
    assert z.capas["propuesta_anp"]["estado"] == "no evaluada"
    ok = [c for c in z.capas.values() if c["estado"] == "ok"]
    assert len(ok) == len(zonas.CAPAS) - 1
    for c in ok:  # cada capa, aun vacía, queda registrada con su hash
        manifiesto.verificar(raiz / "raw", Path(c["ruta"]))
    assert z.capas["acr"]["n_poligonos"] == 0
    assert len(z.de_clase(zonas.EXCLUSION)) == 3  # parque + protección estricta + pozo (la zona AD se ignora)
    assert len(z.de_clase(zonas.COLISION)) == 1


def test_consulta_por_lotes_de_identificadores(monkeypatch):
    """Pide los ids y luego las geometrías en lotes; no depende de la paginación del servidor."""
    pedidos = []

    class Resp:
        def __init__(self, datos):
            self.datos = datos

        def raise_for_status(self):
            pass

        def json(self):
            return self.datos

    def get(url, params, timeout):
        pedidos.append(params)
        if params.get("returnIdsOnly"):
            return Resp({"objectIds": [5, 3, 1, 4, 2]})
        ids = params["objectIds"].split(",")
        return Resp({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"id": i}, "geometry": None} for i in ids]})

    class Sesion:
        pass
    sesion = Sesion()
    sesion.get = get
    monkeypatch.setattr(zonas, "_sesion", lambda: sesion)
    gj = zonas.consultar_arcgis("u", (0, 1, 0, 1), lote=2)
    assert [f["properties"]["id"] for f in gj["features"]] == ["1", "2", "3", "4", "5"]
    assert [p.get("objectIds") for p in pedidos[1:]] == ["1,2", "3,4", "5"]


def test_lote_incompleto_es_error(monkeypatch):
    class Resp:
        def __init__(self, d):
            self.d = d

        def raise_for_status(self):
            pass

        def json(self):
            return self.d

    def get(url, params, timeout):
        if params.get("returnIdsOnly"):
            return Resp({"objectIds": [1, 2]})
        return Resp({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": None}]})

    class Sesion:
        pass
    sesion = Sesion()
    sesion.get = get
    monkeypatch.setattr(zonas, "_sesion", lambda: sesion)
    with pytest.raises(RuntimeError, match="llegaron 1"):
        zonas.consultar_arcgis("u", (0, 1, 0, 1))


# --- Evaluación por celda (ADR-0010) --------------------------------------------------------------

@pytest.fixture
def tabla_con_zonas(cfg_sintetica):
    raiz = Path(cfg_sintetica.directorio_datos)
    bat = batimetria.obtener(cfg_sintetica, raiz)
    malla = waverys.obtener(cfg_sintetica, raiz)
    z = zonas.obtener(batimetria.recuadro_con_margen(cfg_sintetica), raiz)

    def hacer(resguardo_km=1.0):
        cfg_sintetica.resguardo_km = resguardo_km
        df, _ = candidatos.tabla(cfg_sintetica, bat.elevacion, malla, z)
        return df
    return hacer


def test_exclusion_dentro_de_la_celda(tabla_con_zonas):
    f = _fila(tabla_con_zonas(), LAT0, -75.4)
    assert f["excluido_por_zona"] and "Parque Sintético" in f["zonas_exclusion"]
    assert f["motivo_descarte"].startswith("2:")
    assert f["dist_exclusion_km"] == 0


def test_resguardo_de_1_km(tabla_con_zonas):
    assert _fila(tabla_con_zonas(1.0), -16.0, -75.4)["excluido_por_zona"]  # zona a ≈ 0.56 km de la celda
    assert not _fila(tabla_con_zonas(0.5), -16.0, -75.4)["excluido_por_zona"]
    assert not _fila(tabla_con_zonas(1.0), -15.2, -75.4)["excluido_por_zona"]  # parque a ≈ 5.6 km


def test_pozo_puntual_excluye_su_celda(tabla_con_zonas):
    f = _fila(tabla_con_zonas(), -14.8, -75.4)
    assert f["excluido_por_zona"] and "Pozo Sintético" in f["zonas_exclusion"]


def test_colision_se_declara_y_no_descarta(tabla_con_zonas):
    f = _fila(tabla_con_zonas(), -14.6, -75.4)
    assert "Reserva Sintética" in f["zonas_colision"]
    assert f["cumple_criterios"]


def test_hasta_tres_candidatos_fuera_de_exclusion(tabla_con_zonas):
    df = tabla_con_zonas()
    top = df[df["candidato"].notna()]
    assert top["candidato"].tolist() == [1, 2, 3]
    assert not top["excluido_por_zona"].any()
    assert np.allclose(top["lon"], -75.4)
    assert set(top["lat"].round(1)) <= {-15.2, -15.6, -15.0, -14.8, -14.6, -14.4, -16.2, -16.4}


# --- Mapa exploratorio con zonas --------------------------------------------------------------

def test_mapa_exploratorio_declara_capas_no_evaluadas(cfg_sintetica):
    r = generar_mapa(cfg_sintetica)
    meta = json.loads(r.salidas["metadatos"].read_text(encoding="utf-8"))
    assert meta["tipo"] == "mapa exploratorio"
    assert "Propuestas de ANP" in meta["no_evaluado"]
    textos = " ".join(t.get_text() for t in r.figura.axes[0].texts)
    assert "Propuestas de ANP" in textos and "MAPA EXPLORATORIO" in textos and "GEBCO" in textos
    leyenda = [t.get_text() for t in r.figura.axes[0].get_legend().get_texts()]
    assert "EXCLUSIÓN · Áreas protegidas (SERNANP) (2)" in leyenda
    assert any(t.startswith("Resguardo de 1 km") for t in leyenda)
    assert any(t.startswith("Áreas protegidas (SERNANP)") for t in leyenda)
    assert "EXCLUSIÓN · pozo petrolero (1)" in leyenda


# --- GEBCO y mapa definitivo --------------------------------------------------------------------

@pytest.fixture
def gebco_descargado(tmp_path):
    """Recorte tipo GEBCO (variable elevation) con la misma pendiente que la batimetría sintética."""
    lon = np.round(np.arange(-77.0, -74.0, 1 / 240), 5)
    lat = np.round(np.arange(-16.8, -14.0, 1 / 240), 5)
    elev = 1000.0 * (lon[None, :] + 75.0) * np.ones((lat.size, 1))
    p = tmp_path / "descargas" / "gebco_2025_recorte.nc"
    p.parent.mkdir()
    xr.Dataset({"elevation": (("lat", "lon"), elev.astype("int16"))}, coords={"lat": lat, "lon": lon}).to_netcdf(p)
    return p


@pytest.fixture
def exploratorio(cfg_sintetica):
    return generar_mapa(cfg_sintetica)


def test_registrar_gebco(gebco_descargado, cfg_sintetica):
    raiz = Path(cfg_sintetica.directorio_datos)
    ruta, e = gebco.registrar(gebco_descargado, raiz, "GEBCO_2025")
    assert ruta.parent == raiz / "raw" / "batimetria" / "gebco"
    assert e["fuente"] == gebco.FUENTE_GEBCO and e["producto"] == "GEBCO_2025"
    assert gebco.registrar(gebco_descargado, raiz, "GEBCO_2025")[1]["sha256"] == e["sha256"]  # idempotente


def test_mapa_final_publica_y_no_descarga(exploratorio, gebco_descargado, cfg_sintetica, monkeypatch):
    raiz = Path(cfg_sintetica.directorio_datos)
    ruta_gebco, _ = gebco.registrar(gebco_descargado, raiz, "GEBCO_2025")

    def _sin_descargas(*a, **k):
        raise AssertionError("el mapa definitivo no debe descargar")
    monkeypatch.setattr(zonas, "_consultar", _sin_descargas)

    c1 = exploratorio.tabla[exploratorio.tabla["candidato"] == 1].iloc[0]
    r = generar_mapa_final(ConfigFinal(exploratorio=str(exploratorio.salidas["metadatos"]), gebco=str(ruta_gebco),
                                       nodo_lat=float(c1["lat"]), nodo_lon=float(c1["lon"])))
    assert "processed" in r.salidas["mapa"].parts
    m = json.loads(r.salidas["manifiesto"].read_text(encoding="utf-8"))
    for nombre, h in m["salidas_sha256"].items():
        assert manifiesto.sha256(r.salidas["mapa"].parent / nombre) == h
    ficha = json.loads(r.salidas["ficha"].read_text(encoding="utf-8"))
    assert ficha["nodo_elegido"]["lat"] == pytest.approx(c1["lat"])
    assert len(ficha["alternativas"]) == 2
    assert "Propuestas de ANP" in ficha["no_evaluado"]
    assert r.figura.axes[0].get_title(loc="left").startswith("Mapa definitivo")


def test_gate_falla_si_el_nodo_esta_excluido_y_borra_la_salida_anterior(exploratorio, gebco_descargado, cfg_sintetica):
    raiz = Path(cfg_sintetica.directorio_datos)
    ruta_gebco, _ = gebco.registrar(gebco_descargado, raiz, "GEBCO_2025")
    meta = str(exploratorio.salidas["metadatos"])
    c1 = exploratorio.tabla[exploratorio.tabla["candidato"] == 1].iloc[0]
    r = generar_mapa_final(ConfigFinal(meta, str(ruta_gebco), float(c1["lat"]), float(c1["lon"])))
    publicado = r.salidas["mapa"].parent
    assert publicado.exists()
    with pytest.raises(ErrorGate, match="2: zona de exclusión"):
        generar_mapa_final(ConfigFinal(meta, str(ruta_gebco), LAT0, -75.4))
    assert not publicado.exists()


def test_gebco_sin_registrar_es_error(exploratorio, gebco_descargado):
    with pytest.raises(ErrorGate, match="registrar-gebco"):
        generar_mapa_final(ConfigFinal(str(exploratorio.salidas["metadatos"]), str(gebco_descargado), -15.2, -75.4))


def test_gebco_que_no_cubre_es_error(exploratorio, cfg_sintetica, tmp_path):
    lon = np.arange(-76.0, -75.0, 0.01)
    lat = np.arange(-16.0, -15.0, 0.01)
    p = tmp_path / "chico.nc"
    xr.Dataset({"elevation": (("lat", "lon"), -np.ones((lat.size, lon.size)))}, coords={"lat": lat, "lon": lon}).to_netcdf(p)
    ruta, _ = gebco.registrar(p, Path(cfg_sintetica.directorio_datos), "GEBCO_2025")
    with pytest.raises(ErrorGate, match="no cubre"):
        generar_mapa_final(ConfigFinal(str(exploratorio.salidas["metadatos"]), str(ruta), -15.2, -75.4))


def test_cli_registrar_y_final(exploratorio, gebco_descargado, cfg_sintetica, capsys):
    raiz = cfg_sintetica.directorio_datos
    assert main_registrar_gebco(["--archivo", str(gebco_descargado), "--version", "GEBCO_2025", "--datos", raiz]) == 0
    ruta = Path(raiz) / "raw" / "batimetria" / "gebco" / gebco_descargado.name
    meta = str(exploratorio.salidas["metadatos"])
    assert main_final(["--exploratorio", meta, "--gebco", str(ruta), "--nodo-lat", str(LAT0), "--nodo-lon", "-75.4"]) == 1
    assert "GATE FALLIDO" in capsys.readouterr().err
    c1 = exploratorio.tabla[exploratorio.tabla["candidato"] == 1].iloc[0]
    assert main_final(["--exploratorio", meta, "--gebco", str(ruta), "--nodo-lat", str(c1["lat"]), "--nodo-lon", str(c1["lon"])]) == 0


def test_cli_exploratorio_avisa_gebco(cfg_sintetica, capsys, tmp_path):
    import yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg_sintetica.como_dict(), allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(p)]) == 0
    out = capsys.readouterr().out
    assert "MAPA EXPLORATORIO" in out and "download.gebco.net" in out
    assert "Propuestas de ANP" in out
