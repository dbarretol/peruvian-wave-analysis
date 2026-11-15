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


# --- Gate de la ficha (F01-K-01, F01-K-02) ---
import pandas as pd  # noqa: E402

GATE_SERIE = {
    "cumple": True,
    "catalogo": {
        "dataset_id": "ds",
        "version": "202411",
        "inicio": "1980-01-01T00:00:00Z",
        "fin": "2026-08-31T21:00:00Z",
        "paso_h": 3.0,
        "consulta_utc": "2026-11-15T04:35:53Z",
    },
}
SECTORES = pd.DataFrame({"sector": range(16), "clase": ["abierto"] * 16})


def _crono(gate=GATE_SERIE):
    return pd.DataFrame(
        ficha.CRONOLOGIA + ficha.filas_catalogo(gate), columns=["fecha", "tipo", "descripcion", "referencia"]
    )


def _res(filas, prof_archivo=0):
    m = pd.DataFrame(filas, columns=["mision", "tiempo", "asimilacion", "util_para_sesgo"])
    m["tiempo"] = pd.to_datetime(m["tiempo"])
    return {"principal": {"mediciones": m, "prof_archivo_200km": prof_archivo}}


BUENAS = [
    ("HY-2", "2015-01-01", "no asimilada", True),
    ("GEOSAT", "1987-01-01", "no asimilada", True),
    ("JASON-2", "2012-01-01", "asimilada", False),
    ("SWOT", "2024-01-01", "no verificable", False),
]


def test_gate_cumple_con_datos_correctos():
    assert ficha.gate_ficha(_res(BUENAS), _crono(), SECTORES, GATE_SERIE, "g.nc") == []


@pytest.mark.parametrize(
    "fila",
    [
        ("JASON-2", "2012-01-01", "asimilada", True),  # asimilada marcada como útil
        ("SWOT", "2024-01-01", "no verificable", True),  # no verificable marcada como útil
        ("SARAL", "2014-01-01", "no asimilada", True),  # útil sin ser de 1980–1992 ni no asimilada según el QUID
    ],
)
def test_gate_falla_con_util_indebido(fila):
    fallas = ficha.gate_ficha(_res(BUENAS + [fila]), _crono(), SECTORES, GATE_SERIE, "g.nc")
    assert any("útiles para el sesgo" in f for f in fallas)


def test_gate_falla_sin_gebco_y_sin_serie():
    fallas = ficha.gate_ficha(_res(BUENAS, prof_archivo=5), _crono(), SECTORES, None, "g.nc")
    assert any("sin profundidad de GEBCO" in f for f in fallas)
    assert any("gate de la serie" in f for f in fallas)


def test_cronologia_exige_las_filas_del_contrato():
    assert set(ficha.TIPOS_CRONOLOGIA) <= set(_crono()["tipo"])
    sin_version = _crono()[lambda d: d["tipo"] != "versión"]
    assert any("versión" in f for f in ficha.gate_ficha(_res(BUENAS), sin_version, SECTORES, GATE_SERIE, "g.nc"))
    vacia = _crono().assign(referencia=lambda d: d["referencia"].where(d["tipo"] != "física", ""))
    assert any("sin referencia" in f for f in ficha.gate_ficha(_res(BUENAS), vacia, SECTORES, GATE_SERIE, "g.nc"))


def test_cronologia_sin_catalogo_dice_no_disponible():
    filas = ficha.filas_catalogo(None)
    assert [f[1] for f in filas] == ["período", "fecha de corte", "versión"]
    assert all(f[3] == "no disponible" and "serie_gate.json" in f[2] for f in filas)


def test_gate_falla_con_sector_sin_clase():
    malos = SECTORES.assign(clase=["abierto"] * 15 + [None])
    assert any("16 sectores" in f for f in ficha.gate_ficha(_res(BUENAS), _crono(), malos, GATE_SERIE, "g.nc"))
