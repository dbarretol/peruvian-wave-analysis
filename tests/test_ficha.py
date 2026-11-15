"""Piezas de la ficha del sitio que no necesitan red ni datos reales."""
from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from peruvian_wave_analysis.sitio import ficha


def _recorte(ruta, o, e, s, n):
    lat = np.arange(s, n + 1e-9, 0.1)
    lon = np.arange(o, e + 1e-9, 0.1)
    elev = -1000.0 - 10.0 * (lon[None, :] - o) + 0.0 * lat[:, None]
    xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": lat, "lon": lon}).to_netcdf(ruta)
    return ruta


def test_json_limpio_sin_nan():
    x = {"a": float("nan"), "b": [1.0, np.float64("inf"), np.int64(3)], "c": np.bool_(True), "d": "x"}
    texto = json.dumps(ficha._json_limpio(x), allow_nan=False)
    assert json.loads(texto) == {"a": None, "b": [1.0, None, 3], "c": True, "d": "x"}


def test_recuadro_cubre_los_200_km_de_ambos_nodos():
    o, e, s, n = ficha._recuadro_observaciones([(-15.0, -75.8), (-14.8, -76.0)])
    assert s == pytest.approx(-15.0 - 200 / 111.195)
    assert n == pytest.approx(-14.8 + 200 / 111.195)
    assert o < -77.8 and e > -74.0


def test_elige_el_recorte_que_cubre(tmp_path):
    _recorte(tmp_path / "a_chico.nc", -76.6, -74.0, -16.7, -14.1)
    _recorte(tmp_path / "b_grande.nc", -78.5, -72.5, -17.5, -12.5)
    _recorte(tmp_path / "c_enorme.nc", -80.0, -70.0, -20.0, -10.0)
    rec = ficha._recuadro_observaciones([(-15.0, -75.8), (-14.8, -76.0)])
    assert ficha._elegir_gebco(tmp_path, rec).name == "b_grande.nc"


def test_si_ninguno_cubre_elige_el_mayor(tmp_path):
    _recorte(tmp_path / "a_chico.nc", -76.6, -74.0, -16.7, -14.1)
    _recorte(tmp_path / "b_medio.nc", -77.0, -74.0, -17.0, -13.5)
    rec = ficha._recuadro_observaciones([(-15.0, -75.8)])
    assert ficha._elegir_gebco(tmp_path, rec).name == "b_medio.nc"


def test_profundidad_vectorizada_y_nan_fuera(tmp_path):
    ruta = _recorte(tmp_path / "g.nc", -76.0, -75.0, -16.0, -15.0)
    with xr.open_dataset(ruta) as ds:
        elev = ds["elevation"].load()
    prof = ficha._profundidad(elev)
    out = prof(np.array([-15.5, -15.5, -14.0]), np.array([-76.0, -75.1, -75.5]))
    assert out[0] == pytest.approx(1000.0) and out[1] == pytest.approx(1009.0) and np.isnan(out[2])
