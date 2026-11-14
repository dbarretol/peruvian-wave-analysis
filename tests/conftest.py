"""Datos sintéticos de respuesta conocida para el mapa de sitio.

Batimetría plana inclinada: elevación = 1000·(lon + 75) m. La costa está en lon = −75.0,
la isóbata de 50 m en −75.05, la de 100 m en −75.1 y la de 200 m en −75.2.
Malla tipo WAVERYS de 0.2°: nodos con lon ≥ −75.0 sin dato (tierra).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

LAT0, LON0 = -15.4, -75.6


@pytest.fixture
def datos_sinteticos(tmp_path):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    entrada = tmp_path / "entrada"
    entrada.mkdir()

    lon = np.round(np.arange(-77.0, -73.99, 0.005), 4)
    lat = np.round(np.arange(-16.8, -13.99, 0.005), 4)
    elev = 1000.0 * (lon[None, :] + 75.0) * np.ones((lat.size, 1))
    ruta_bat = entrada / "batimetria.nc"
    xr.Dataset({"elevation": (("lat", "lon"), elev.astype("float32"))}, coords={"lat": lat, "lon": lon}).to_netcdf(ruta_bat)

    glat = np.round(np.arange(-16.6, -14.19, 0.2), 4)
    glon = np.round(np.arange(-76.8, -74.39, 0.2), 4)
    vhm0 = np.where(glon[None, None, :] >= -75.0 - 1e-6, np.nan, 1.5) * np.ones((1, glat.size, 1))
    ruta_wav = entrada / "waverys.nc"
    xr.Dataset(
        {"VHM0": (("time", "latitude", "longitude"), vhm0)},
        coords={"time": [np.datetime64("2020-07-15T12:00")], "latitude": glat, "longitude": glon},
    ).to_netcdf(ruta_wav)
    return raiz, ruta_bat, ruta_wav


@pytest.fixture
def cfg_sintetica(datos_sinteticos):
    from peruvian_wave_analysis.sitio import ConfigMapa

    raiz, ruta_bat, ruta_wav = datos_sinteticos
    return ConfigMapa(
        nombre="sintetico",
        latitud=LAT0,
        longitud=LON0,
        isobatas=[200, 100, 50],
        semiancho_grados=1.0,
        n_candidatos=3,
        directorio_datos=str(raiz),
        batimetria={"fuente": "archivo", "archivo": str(ruta_bat)},
        waverys={"fuente": "archivo", "archivo": str(ruta_wav)},
    )


def _caja(o, e, s, n, **props):
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [[[o, s], [e, s], [e, n], [o, n], [o, s]]]}}


# Zonas sintéticas (nodos de la malla cada 0.2°; celdas de ±0.1°):
# - Parque nacional (exclusión) que contiene el nodo (−15.4, −75.4).
# - Zona de protección estricta (exclusión) a 0.005° (≈ 0.56 km) al norte de la celda del nodo (−16.0, −75.4):
#   la descarta solo el resguardo de 1 km. También cae dentro de la celda de (−15.8, −75.4).
# - Reserva nacional (colisión) entre −14.6 y −14.3 de latitud: se declara, no descarta.
# - Un pozo (punto, exclusión) dentro de la celda del nodo (−14.8, −75.4).
# - La capa de propuestas de ANP falla: debe quedar «no evaluada».
ZONAS_SINTETICAS = {
    "peru_sernanp_0102/MapServer/0": [
        _caja(-75.45, -75.35, -15.45, -15.35, anp_nomb="Parque Sintético", anp_cate="Parque Nacional"),
        _caja(-77.0, -74.0, -14.65, -14.3, anp_nomb="Reserva Sintética", anp_cate="Reserva Nacional"),
    ],
    "SERV_OTRAS_FUENTES/MapServer/36": [
        {"type": "Feature", "properties": {"nombre": "Pozo Sintético"},
         "geometry": {"type": "Point", "coordinates": [-75.4, -14.75]}},
    ],
    "peru_sernanp_0213/MapServer/0": [
        _caja(-75.45, -75.35, -15.895, -15.85, anp_nomb="Reserva Sintética", tz_nomb="Protección Estricta", tz_cod="PE"),
        _caja(-77.0, -74.0, -14.65, -14.3, anp_nomb="Reserva Sintética", tz_nomb="Aprovechamiento Directo", tz_cod="AD"),
    ],
}


def consultar_falso(url, recuadro, pagina=1000):
    if url.endswith("peru_sernanp_0103/MapServer/0"):
        raise RuntimeError("servicio caído (simulado)")
    for clave, feats in ZONAS_SINTETICAS.items():
        if url.endswith(clave):
            return {"type": "FeatureCollection", "features": feats}
    return {"type": "FeatureCollection", "features": []}


@pytest.fixture(autouse=True)
def sin_red(monkeypatch):
    """Ninguna prueba accede a la red: las zonas salen de ZONAS_SINTETICAS."""
    from peruvian_wave_analysis.sitio import zonas

    monkeypatch.setattr(zonas, "_consultar", consultar_falso)

    def _prohibido(*a, **k):
        raise AssertionError("acceso a la red en una prueba")

    import requests
    monkeypatch.setattr(requests, "get", _prohibido)
    monkeypatch.setattr(requests.Session, "request", _prohibido)
