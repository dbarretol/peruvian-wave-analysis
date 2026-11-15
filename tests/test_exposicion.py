"""Exposición por sectores y obstáculos (contrato de la F1 §3.1, §3.2 y §5.2–5.3) con geometría sintética."""
from __future__ import annotations

import numpy as np
import pytest
import shapely
import xarray as xr
from shapely.geometry import box

from peruvian_wave_analysis.sitio import exposicion as ex

LAT0, LON0 = -15.0, -75.8
# Continente sintético: todo lo que está al este de −75.5 (≈ 32 km al este del nodo).
CONTINENTE = box(-75.5, -89.0, 179.0, 89.0)


def _prep(g):
    shapely.prepare(g)
    return g


def test_destino_y_distancia_son_coherentes():
    la, lo = ex.destino(LAT0, LON0, 270.0, 1000.0)
    assert float(ex.distancia_km(LAT0, LON0, la, lo)) == pytest.approx(1000.0, rel=1e-6)
    assert float(ex.rumbo(LAT0, LON0, *ex.destino(LAT0, LON0, 45.0, 10.0))) == pytest.approx(45.0, abs=0.05)


def test_sectores_abierto_cerrado_y_parcial():
    e = ex.calcular(_prep(CONTINENTE), LAT0, LON0)
    s = e.sectores.set_index("nombre_sector")
    assert s.loc["E", "clase"] == "cerrado" and s.loc["E", "dist_min_tierra_km"] == pytest.approx(32, abs=1.5)
    assert s.loc["O", "clase"] == "abierto"
    # N: la mitad este de la cuña (rumbos 1° a 11°) acaba tocando el continente; la del oeste no.
    assert s.loc["N", "clase"] == "parcial" and 0 < s.loc["N", "fraccion_rayos_1000_km"] < 1
    assert len(e.rayos) == 16 * 23


def test_isla_que_solo_toca_el_borde_de_una_cuna_deja_el_sector_parcial():
    # Isla pequeña a ~300 km al oeste, desplazada hacia el borde norte del sector O (rumbos 259°–281°).
    la, lo = ex.destino(LAT0, LON0, 280.5, 300.0)
    isla = box(lo - 0.05, la - 0.05, lo + 0.05, la + 0.05)
    e = ex.calcular(_prep(shapely.union(CONTINENTE, isla)), LAT0, LON0)
    o = e.sectores.set_index("nombre_sector").loc["O"]
    assert o["clase"] == "parcial" and o["fraccion_rayos_1000_km"] < 1 and o["dist_min_tierra_km"] == pytest.approx(300, abs=10)


def test_alcance_oceano_austral_y_pacifico_norte():
    e = ex.calcular(_prep(CONTINENTE), LAT0, LON0)
    s = e.sectores.set_index("nombre_sector")
    assert bool(s.loc["S", "abierto_oceano_austral"]) and not bool(s.loc["S", "abierto_pacifico_norte"])
    assert bool(s.loc["NO", "abierto_pacifico_norte"]) and not bool(s.loc["E", "abierto_pacifico_norte"])
    # Una barrera en el ecuador cierra el Pacífico Norte por el noroeste.
    barrera = box(-179.0, -1.0, -75.5, 1.0)
    e2 = ex.calcular(_prep(shapely.union(CONTINENTE, barrera)), LAT0, LON0)
    assert not bool(e2.sectores.set_index("nombre_sector").loc["NO", "abierto_pacifico_norte"])


def test_error_si_el_rayo_sale_del_insumo():
    with pytest.raises(ex.ErrorExposicion):
        ex.calcular(_prep(CONTINENTE), LAT0, LON0, extension=(-80.0, -70.0, -20.0, -10.0))


def test_mascara_del_modelo_se_compara():
    # El modelo ve tierra al este de −75.65 (más cerca que el continente sintético).
    tierra_m = (lambda la, lo: lo > -75.65, (-76.5, -75.0, -15.7, -14.3))
    e = ex.calcular(_prep(CONTINENTE), LAT0, LON0, mascara_modelo=tierra_m)
    s = e.sectores.set_index("nombre_sector")
    assert s.loc["E", "dist_min_tierra_modelo_km"] < s.loc["E", "dist_min_tierra_km"]
    assert np.isnan(s.loc["O", "dist_min_tierra_modelo_km"])


def _gebco(islas):
    lat = np.round(np.arange(-15.6, -14.4, 1 / 240), 5)
    lon = np.round(np.arange(-76.4, -75.0, 1 / 240), 5)
    z = np.where(lon[None, :] > -75.5, 100.0, -300.0) * np.ones((lat.size, 1))
    for (la, lo, r) in islas:
        z[(np.abs(lat[:, None] - la) <= r) & (np.abs(lon[None, :] - lo) <= r)] = 10.0
    return xr.DataArray(z, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def test_obstaculos_dentro_y_fuera_de_50_km_y_bajo_la_resolucion():
    sect = ex.calcular(_prep(CONTINENTE), LAT0, LON0).sectores
    cerca = (-15.0, -76.0, 0.02)   # ≈ 21 km al oeste
    lejos = (-15.55, -76.35, 0.02)  # ≈ 85 km: fuera
    df = ex.obstaculos(_gebco([cerca, lejos]), [], LAT0, LON0, sect, es_tierra_modelo=lambda la, lo: False)
    assert len(df) == 1 and df.loc[0, "dist_min_km"] == pytest.approx(19, abs=3)
    assert df.loc[0, "clase_sector"] == "abierto" and not df.loc[0, "tierra_en_modelo"]
    # Una isla menor que un píxel no aparece en GEBCO; solo la agrega Natural Earth si la trae.
    diminuta = box(-75.901, -15.101, -75.900, -15.100)
    df2 = ex.obstaculos(_gebco([]), [diminuta], LAT0, LON0, sect)
    assert list(df2["origen"]) == ["Natural Earth (islas menores)"]
