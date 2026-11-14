import json

import numpy as np
import pytest

from peruvian_wave_analysis.sitio import ConfigMapa, candidatos, generar_mapa
from peruvian_wave_analysis.sitio import batimetria, waverys
from peruvian_wave_analysis.sitio.cli import main

from .conftest import LAT0, LON0

KM_POR_GRADO_LON = 111.195 * np.cos(np.radians(LAT0))


# --- Configuración -------------------------------------------------------------------------

def test_config_por_defecto():
    c = ConfigMapa(nombre="x", latitud=-15, longitud=-75)
    assert c.isobatas == [200, 100, 50]
    assert c.isobata_principal == 200
    assert c.n_candidatos == 3 and c.resguardo_km == 1.0 and c.zonas
    assert c.recuadro == (-76, -74, -16, -14)


@pytest.mark.parametrize("kw", [
    {"nombre": "Con Mayúsculas"},
    {"isobatas": []},
    {"isobatas": [100, 100]},
    {"isobata_principal": 500},
    {"semiancho_grados": 0},
    {"batimetria": {"fuente": "archivo"}},
    {"waverys": {"fuente": "otra"}},
    {"resguardo_km": -1},
    {"n_candidatos": 0},
])
def test_config_invalida(kw):
    base = {"nombre": "x", "latitud": -15, "longitud": -75} | kw
    with pytest.raises(ValueError):
        ConfigMapa(**base)


def test_isobatas_negativas_se_toman_como_profundidad():
    c = ConfigMapa(nombre="x", latitud=-15, longitud=-75, isobatas=[-150, 75])
    assert c.isobatas == [150, 75] and c.isobata_principal == 150


# --- Distancias e isolíneas --------------------------------------------------------------------

def test_proyeccion_un_grado():
    xy = candidatos.proyectar([LON0 + 1, LON0], [LAT0, LAT0 + 1], LON0, LAT0)
    assert xy[0, 0] == pytest.approx(KM_POR_GRADO_LON, rel=1e-3)
    assert xy[1, 1] == pytest.approx(111.195, rel=1e-3)


def test_distancia_a_isobata_conocida(cfg_sintetica):
    bat = batimetria.obtener(cfg_sintetica, cfg_sintetica.directorio_datos)
    lineas = candidatos.isolineas(bat.elevacion, -200)
    assert lineas
    assert np.allclose(np.concatenate(lineas)[:, 0], -75.2, atol=1e-3)
    d = candidatos.distancia_a_lineas([-75.6], [LAT0], lineas, LON0, LAT0)
    assert d[0] == pytest.approx(0.4 * KM_POR_GRADO_LON, rel=0.01)


def test_sin_lineas_da_nan():
    assert np.isnan(candidatos.distancia_a_lineas([0.0], [0.0], [], 0, 0)).all()


# --- Tabla de candidatos -------------------------------------------------------------------

@pytest.fixture
def tabla(cfg_sintetica):
    raiz = cfg_sintetica.directorio_datos
    bat = batimetria.obtener(cfg_sintetica, raiz)
    malla = waverys.obtener(cfg_sintetica, raiz)
    df, _ = candidatos.tabla(cfg_sintetica, bat.elevacion, malla)
    return df


def _fila(df, lat, lon):
    f = df[np.isclose(df["lat"], lat) & np.isclose(df["lon"], lon)]
    assert len(f) == 1
    return f.iloc[0]


def test_tabla_cubre_solo_el_recuadro(tabla):
    assert tabla["lat"].between(-16.4 - 1e-6, -14.4 + 1e-6).all()
    assert tabla["lon"].between(-76.6 - 1e-6, -74.6 + 1e-6).all()
    assert len(tabla) == 11 * 11


def test_profundidad_y_celda(tabla):
    f = _fila(tabla, LAT0, -75.6)
    assert f["prof_nodo_m"] == pytest.approx(600, abs=5)
    assert f["prof_media_celda_m"] == pytest.approx(600, abs=5)
    assert f["prof_min_celda_m"] == pytest.approx(500, abs=5)
    assert f["frac_tierra_celda"] == 0
    costa = _fila(tabla, LAT0, -75.0)  # celda de −75.1 a −74.9: mitad tierra
    assert costa["frac_tierra_celda"] == pytest.approx(0.5, abs=0.05)
    assert not costa["valido"]


def test_vecinas(tabla):
    assert _fila(tabla, LAT0, -75.4)["vecinas_validas"] == 8
    assert _fila(tabla, LAT0, -75.2)["vecinas_validas"] == 5  # columna de −75.0 es tierra
    borde = _fila(tabla, -16.4, -76.6)  # el margen de un nodo da sus ocho vecinas
    assert borde["vecinas_presentes"] == 8


def test_orden_de_candidatos(tabla):
    primero = tabla.iloc[0]
    assert primero["candidato"] == 1
    assert primero["lon"] == pytest.approx(-75.4)  # el nodo más cercano a la isóbata que cumple
    assert primero["lat"] == pytest.approx(LAT0)  # desempate por distancia a la referencia
    assert primero["dist_iso_200m_km"] == pytest.approx(0.2 * KM_POR_GRADO_LON, rel=0.02)
    cumplen = tabla[tabla["cumple_criterios"]]
    assert len(cumplen) > 3
    assert cumplen["candidato"].head(3).tolist() == [1, 2, 3]  # ADR-0010: hasta 3
    assert cumplen["candidato"].iloc[3:].isna().all()
    assert tabla.loc[~tabla["cumple_criterios"], "candidato"].isna().all()
    assert not tabla["zonas_evaluadas"].any()
    assert not tabla.loc[np.isclose(tabla["lon"], -75.2), "cumple_criterios"].any()


def test_isobata_ausente_avisa(cfg_sintetica):
    cfg_sintetica.isobatas = [200, 9000]
    raiz = cfg_sintetica.directorio_datos
    bat = batimetria.obtener(cfg_sintetica, raiz)
    malla = waverys.obtener(cfg_sintetica, raiz)
    with pytest.warns(UserWarning, match="9000"):
        df, _ = candidatos.tabla(cfg_sintetica, bat.elevacion, malla)
    assert df["dist_iso_9000m_km"].isna().all()


# --- Extremo a extremo ---------------------------------------------------------------------

def test_generar_mapa_escribe_salidas(cfg_sintetica):
    r = generar_mapa(cfg_sintetica)
    for ruta in r.salidas.values():
        assert ruta.exists() and ruta.stat().st_size > 0
    assert "interim" in r.salidas["mapa"].parts and "F01-sitio-y-fuentes" in r.salidas["mapa"].parts
    meta = json.loads(r.salidas["metadatos"].read_text(encoding="utf-8"))
    assert meta["configuracion"]["isobatas"] == [200, 100, 50]
    assert meta["insumos"]["batimetria"]["sha256"]
    textos = [t.get_text() for t in r.figura.axes[0].get_legend().get_texts()]
    assert any("Isóbata 50 m" in t for t in textos)
    assert any("principal" in t for t in textos)
    assert r.salidas["siguiente_paso"].exists()
    assert "download.gebco.net" in r.salidas["siguiente_paso"].read_text(encoding="utf-8")
    assert meta["recuadro_gebco_oeste_este_sur_norte"] == [-76.9, -74.3, -16.7, -14.1]


def test_cli(cfg_sintetica, capsys):
    b, w = cfg_sintetica.batimetria, cfg_sintetica.waverys
    import yaml
    from pathlib import Path

    ruta_cfg = Path(cfg_sintetica.directorio_datos).parent / "cfg.yaml"
    ruta_cfg.write_text(yaml.safe_dump(cfg_sintetica.como_dict(), allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(ruta_cfg), "--isobatas", "150", "75"]) == 0
    salida = capsys.readouterr().out
    assert "dist_iso_150m_km" in salida
    assert "mapa:" in salida


def test_cli_sin_coordenada(capsys):
    assert main(["--nombre", "x"]) == 2
