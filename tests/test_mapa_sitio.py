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
    assert c.n_candidatos == 3 and c.resguardo_km == 1.0 and c.zonas and c.min_vecinas == 0
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
    {"min_vecinas": 9},
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
    """ADR-0013: las vecinas no descartan; gana el nodo sobre la isóbata de 200 m (lon −75.2, 5 vecinas)."""
    primero = tabla.iloc[0]
    assert primero["candidato"] == 1
    assert primero["lon"] == pytest.approx(-75.2)
    assert primero["lat"] == pytest.approx(LAT0)  # desempate por distancia a la referencia
    assert primero["dist_iso_200m_km"] == pytest.approx(0, abs=0.3)
    assert primero["vecinas_validas"] == 5  # se informa, no descarta
    cumplen = tabla[tabla["cumple_criterios"]]
    assert len(cumplen) > 3
    assert cumplen["candidato"].head(3).tolist() == [1, 2, 3]  # hasta 3
    assert cumplen["candidato"].iloc[3:].isna().all()
    assert tabla.loc[~tabla["cumple_criterios"], "candidato"].isna().all()
    assert not tabla["zonas_evaluadas"].any()


def test_min_vecinas_opcional(cfg_sintetica):
    """Con min_vecinas = 8 vuelve el criterio estricto: el nodo de −75.2 queda descartado."""
    cfg_sintetica.min_vecinas = 8
    raiz = cfg_sintetica.directorio_datos
    df, _ = candidatos.tabla(cfg_sintetica, batimetria.obtener(cfg_sintetica, raiz).elevacion,
                             waverys.obtener(cfg_sintetica, raiz))
    f = _fila(df, LAT0, -75.2)
    assert not f["cumple_criterios"] and f["motivo_descarte"] == "3: menos de 8 vecinas con dato"
    assert df.iloc[0]["lon"] == pytest.approx(-75.4)


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


def test_tierra_en_la_celda_es_informativa_salvo_max_tierra():
    """ADR-0014: un nodo con dato y una esquina de tierra en la celda es elegible; max_tierra lo puede filtrar."""
    from types import SimpleNamespace

    f = SimpleNamespace(valido=True, frac_tierra_celda=0.01, excluido_por_zona=False, vecinas_validas=7)
    assert candidatos._motivo(f) == ""
    assert candidatos._motivo(f, max_tierra=0.05) == ""
    assert candidatos._motivo(f, max_tierra=0.0).startswith("1: más de 0 %")
    assert candidatos._motivo(SimpleNamespace(**{**vars(f), "valido": False})).startswith("1: nodo sin dato")
    assert candidatos._motivo(SimpleNamespace(**{**vars(f), "excluido_por_zona": True})).startswith("2:")


def test_max_tierra_fuera_de_rango():
    with pytest.raises(ValueError):
        ConfigMapa(nombre="x", latitud=-15, longitud=-75, max_tierra=1.5)


# --- Puntos de referencia y malla dibujada ---------------------------------------------------

def test_grados_sexagesimales():
    from peruvian_wave_analysis.sitio.config import a_grados
    assert a_grados("15°23′20″S") == pytest.approx(-(15 + 23 / 60 + 20 / 3600))
    assert a_grados("75°04'15\"O") == pytest.approx(-(75 + 4 / 60 + 15 / 3600))
    assert a_grados("75°04′15″W") == a_grados("75°04′15″O")
    assert a_grados("-14° 39' 6.1\" S") == pytest.approx(-(14 + 39 / 60 + 6.1 / 3600))
    assert a_grados("12°N") == 12 and a_grados(-15.5) == -15.5 and a_grados("-15.5") == -15.5
    with pytest.raises(ValueError):
        a_grados("15 grados sur")


def test_puntos_en_config_y_mapa(cfg_sintetica):
    from peruvian_wave_analysis.sitio.mapa import _dms
    cfg_sintetica.puntos = [{"nombre": "Parque eólico (aprox.)", "latitud": "15°23′20″S", "longitud": "75°04′15″O"}]
    cfg_sintetica.__post_init__()
    p = cfg_sintetica.puntos[0]
    assert (_dms(p.latitud, "NS"), _dms(p.longitud, "EO")) == ("15°23′20″S", "75°04′15″O")
    r = generar_mapa(cfg_sintetica)
    meta = json.loads(r.salidas["metadatos"].read_text(encoding="utf-8"))
    assert meta["configuracion"]["puntos"][0]["latitud"] == pytest.approx(p.latitud)
    textos = [t.get_text() for t in r.figura.axes[0].get_legend().get_texts()]
    assert any(t.startswith("Parque eólico (aprox.) (15°23′20″S") for t in textos)
    assert not any("sin dato" in t for t in textos)


def test_no_se_dibujan_bordes_de_celdas_solo_de_tierra(cfg_sintetica):
    from peruvian_wave_analysis.sitio.mapa import _bordes_de_celda
    r = generar_mapa(cfg_sintetica, guardar=False)
    elev, malla = r.batimetria.elevacion, r.malla
    lados = _bordes_de_celda(malla, elev)
    h = malla.paso / 2
    fracs = {(round(float(la), 4), round(float(lo), 4)): candidatos._estadisticas_celda(elev, float(la), float(lo), malla.paso)[2]
             for la in malla.lat for lo in malla.lon}
    tierra = [c for c, f in fracs.items() if f >= 1]
    assert tierra and any(f < 1 for f in fracs.values())  # el caso sintético tiene de ambas
    centros = {(round((a[1] + b[1]) / 2, 4), round((a[0] + b[0]) / 2, 4)) for a, b in lados}
    for la, lo in tierra:  # ningún lado de una celda solo de tierra que no comparta con una celda con agua
        vecinas_con_agua = [fracs.get((round(la + dy, 4), round(lo + dx, 4)), 1) < 1
                            for dy, dx in ((0, 2 * h), (0, -2 * h), (2 * h, 0), (-2 * h, 0))]
        lados_celda = [(la, round(lo + h, 4)), (la, round(lo - h, 4)), (round(la + h, 4), lo), (round(la - h, 4), lo)]
        for lado, compartido in zip(lados_celda, vecinas_con_agua):
            assert (tuple(round(v, 4) for v in lado) in centros) == compartido


def test_punto_etiqueta_y_desplazamiento():
    from peruvian_wave_analysis.sitio.config import PuntoReferencia
    p = PuntoReferencia(nombre="S.E. Larga 220 kV (R.M. x)", latitud=-15.4, longitud=-75.1)
    assert p.etiqueta == p.nombre and p.marcador == "D" and p.desplazamiento == [7.0, -3.0]
    q = PuntoReferencia(nombre="S.E. Larga", latitud=-15.4, longitud=-75.1, etiqueta="S.E.", marcador="p",
                        desplazamiento=[-5, 4])
    assert q.etiqueta == "S.E." and q.desplazamiento == [-5.0, 4.0]
    with pytest.raises(ValueError):
        PuntoReferencia(nombre="x", latitud=-15.4, longitud=-75.1, desplazamiento=[1])
