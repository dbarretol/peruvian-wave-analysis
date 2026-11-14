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
        n_candidatos=5,
        directorio_datos=str(raiz),
        batimetria={"fuente": "archivo", "archivo": str(ruta_bat)},
        waverys={"fuente": "archivo", "archivo": str(ruta_wav)},
    )
