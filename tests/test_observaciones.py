"""Clasificación de observaciones (contrato de la F1 §3.4 y §5.4) con mediciones sintéticas y sin red."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from peruvian_wave_analysis.sitio import exposicion as ex
from peruvian_wave_analysis.sitio import observaciones as ob

LAT0, LON0 = -15.0, -75.8
TIERRA = box(-75.5, -40.0, -60.0, 10.0)  # «costa» recta a ≈ 32 km al este del nodo
shapely.prepare(TIERRA)


@pytest.fixture(scope="module")
def sectores():
    return ex.calcular(TIERRA, LAT0, LON0).sectores


@pytest.fixture(scope="module")
def costa():
    return ob.Costa.desde_tierra(TIERRA, LAT0, LON0)


def _punto(rumbo, dist_km):
    la, lo = ex.destino(LAT0, LON0, rumbo, dist_km)
    return float(la), float(lo)


def _df(filas):
    return pd.DataFrame(
        [
            {"mision": m, "tiempo": pd.Timestamp(t), "lat": la, "lon": lo, "prof_archivo_m": -3000.0}
            for m, t, (la, lo) in filas
        ]
    )


def _prof(la, lo):
    return 3000.0


def test_teselas_y_bloque():
    t = ob.teselas(LAT0, LON0)
    assert "015S-284E" in t and "017S-282E" in t and "014S-286E" in t and len(t) == 20
    assert ob._bloque("015S-284E") == "020S_280E"


def test_clases_por_distancia_costa_y_exposicion(sectores, costa):
    d = ob.clasificar(
        _df(
            [
                ("HY-2", "2015-01-01T00:00:00", _punto(270, 10)),  # sitio: cerca, mar adentro
                (
                    "HY-2",
                    "2015-01-02T00:00:00",
                    _punto(90, 20),
                ),  # 20 km hacia la costa: más costero que el nodo → región
                ("HY-2", "2015-01-03T00:00:00", _punto(270, 150)),  # región
                ("HY-2", "2015-01-04T00:00:00", _punto(270, 250)),  # cuenca
            ]
        ),
        LAT0,
        LON0,
        costa,
        _prof,
        sectores,
    )
    assert list(d["clase"]) == ["sitio", "cuenca", "región", "cuenca"]
    # El punto hacia la costa queda a ≈ 12 km de ella pero en el sector E, que es «cerrado»: no es región.
    assert d.loc[1, "clase_sector"] == "cerrado"


def test_el_nodo_cumple_su_propia_clase_sitio(sectores, costa):
    d = ob.clasificar(_df([("HY-2", "2015-01-01", (LAT0, LON0 - 1e-6))]), LAT0, LON0, costa, _prof, sectores)
    assert d.loc[0, "clase"] == "sitio"
    assert d.attrs["dist_costa_nodo_km"] == pytest.approx(32.2, abs=0.5)


def test_profundidad_minima_y_respaldo_de_la_profundidad(sectores, costa):
    somero = lambda la, lo: 50.0  # noqa: E731
    d = ob.clasificar(_df([("HY-2", "2015-01-01", _punto(270, 10))]), LAT0, LON0, costa, somero, sectores)
    assert d.loc[0, "clase"] == "cuenca"
    fuera = lambda la, lo: np.nan  # noqa: E731  fuera del recorte de GEBCO → profundidad del archivo
    d = ob.clasificar(_df([("HY-2", "2015-01-01", _punto(270, 10))]), LAT0, LON0, costa, fuera, sectores)
    assert d.loc[0, "clase"] == "sitio" and d.loc[0, "fuente_prof"] == "archivo (SRTM30+)"


@pytest.mark.parametrize(
    "mision,fecha,esperado",
    [
        ("GEOSAT", "1987-05-01", "no asimilada"),
        ("ERS-1", "1992-06-01", "no asimilada"),  # antes de 1993: sin asimilación
        ("ERS-1", "1994-06-01", "asimilada"),
        ("SARAL", "2014-06-01", "asimilada"),  # misión de la tabla, fuera de su período pero desde 1993
        ("HY-2", "2016-06-01", "no asimilada"),  # QUID: HY-2A no asimilada
        ("SENTINEL-6A", "2022-06-01", "no verificable"),
    ],
)
def test_asimilacion(mision, fecha, esperado):
    assert ob.asimilacion(mision, pd.Timestamp(fecha)) == esperado


@pytest.mark.parametrize(
    "mision,fecha,contrato,periodo",
    [
        ("SARAL", "2014-06-01", "asimilada", "no asimilada"),  # antes de su período (2016-03)
        ("SARAL", "2017-06-01", "asimilada", "asimilada"),
        ("GFO", "2007-06-01", "asimilada", "no asimilada"),  # después de su período (2006-08)
        ("CFOSAT", "2019-06-01", "asimilada", "no asimilada"),  # antes de 2020-02
        ("ERS-1", "1992-06-01", "no asimilada", "no asimilada"),
        ("HY-2", "2016-06-01", "no asimilada", "no asimilada"),
        ("SWOT", "2024-06-01", "no verificable", "no verificable"),
    ],
)
def test_sensibilidad_por_periodo(mision, fecha, contrato, periodo):
    t = pd.Timestamp(fecha)
    assert ob.asimilacion(mision, t) == contrato
    assert ob.asimilacion_por_periodo(mision, t) == periodo


def test_resumen_con_la_regla_de_sensibilidad(sectores, costa):
    p = _punto(270, 5)
    d = ob.clasificar(_df([("SARAL", "2014-01-01", p), ("SARAL", "2017-01-01", p)]), LAT0, LON0, costa, _prof, sectores)
    contrato = ob.cobertura(ob.resumen(d)).set_index("regimen")
    sens = ob.cobertura(ob.resumen(d, "asimilacion_periodo")).set_index("regimen")
    assert contrato.loc["desde 1993 (asimilado)", "pasadas_sitio_N"] == 0
    assert sens.loc["desde 1993 (asimilado)", "pasadas_sitio_N"] == 1


def test_profundidad_vectorizada(sectores, costa):
    p1, p2 = _punto(270, 10), _punto(270, 15)
    prof = lambda la, lo: np.where(np.asarray(lo) < -75.9, 3000.0, np.nan)  # noqa: E731
    d = ob.clasificar(_df([("HY-2", "2015-01-01", p1), ("HY-2", "2015-01-02", p2)]), LAT0, LON0, costa, prof, sectores)
    assert list(d["fuente_prof"]) == ["archivo (SRTM30+)", "GEBCO_2026"]


def test_pasadas_y_cobertura_por_regimen(sectores, costa):
    p = _punto(270, 5)
    filas = []
    for k in range(300):  # 300 pasadas de HY-2 (régimen asimilado), 2 mediciones por pasada a 1 s
        t = pd.Timestamp("2014-01-01") + pd.Timedelta(days=k)
        filas += [("HY-2", t, p), ("HY-2", t + pd.Timedelta(seconds=1), p)]
    filas += [("GEOSAT", pd.Timestamp("1986-01-01") + pd.Timedelta(days=k), p) for k in range(299)]
    filas += [("JASON-2", pd.Timestamp("2012-01-01") + pd.Timedelta(days=k), p) for k in range(500)]  # asimilada
    d = ob.clasificar(_df(filas), LAT0, LON0, costa, _prof, sectores)
    cob = ob.cobertura(ob.resumen(d)).set_index("regimen")
    asim = cob.loc["desde 1993 (asimilado)"]
    libre = cob.loc["1980-1992 (sin asimilación ni corrientes)"]
    assert asim["pasadas_sitio_N"] == 300 and bool(asim["cobertura_util_sitio"])  # JASON-2 no cuenta
    assert libre["pasadas_sitio_N"] == 299 and not bool(libre["cobertura_util_sitio"])


def test_regla_del_respaldo():
    def cob(util):
        return pd.DataFrame([{"regimen": "desde 1993 (asimilado)", "cobertura_util_sitio": util}])

    r = "desde 1993 (asimilado)"
    assert ob.regla_respaldo(cob(False), cob(True), r).startswith("pasar")
    assert ob.regla_respaldo(cob(True), cob(True), r) == "mantener el nodo"
    assert ob.regla_respaldo(cob(False), cob(False), r) == "mantener el nodo"


def test_leer_usa_la_banda_que_trae_el_archivo(tmp_path):
    import xarray as xr

    def archivo(mision, banda):
        d = tmp_path / mision
        d.mkdir()
        t = pd.date_range("2015-01-01", periods=3, freq="1s")
        ds = xr.Dataset(
            {f"SWH_{banda}_quality_control": ("TIME", [1.0, 2.0, 1.0]), "BOT_DEPTH": ("TIME", [-3000.0] * 3)},
            coords={"TIME": t, "LATITUDE": ("TIME", [-15.0] * 3), "LONGITUDE": ("TIME", [284.2] * 3)},
        )
        ruta = d / "x.nc"
        ds.to_netcdf(ruta)
        return ruta

    df = ob.leer([archivo("HY-2", "KU"), archivo("SARAL", "KA")])
    assert len(df) == 4 and set(df["banda"]) == {"KU", "KA"}  # la bandera 2 se descarta
    assert df["lon"].iloc[0] == pytest.approx(-75.8)


# --- Bordes de las clases (F01-K-05) ---
KX = 111.195 * np.cos(np.radians(LAT0))  # km por grado de longitud en la proyección local de Costa


def test_borde_de_25_km(sectores, costa):
    d = ob.clasificar(
        _df([("HY-2", "2015-01-01", _punto(270, 24.9)), ("HY-2", "2015-01-02", _punto(270, 25.1))]),
        LAT0,
        LON0,
        costa,
        _prof,
        sectores,
    )
    assert list(d["clase"]) == ["sitio", "región"]


def test_borde_de_profundidad_100_m(sectores, costa):
    p = _punto(270, 10)
    d100 = ob.clasificar(_df([("HY-2", "2015-01-01", p)]), LAT0, LON0, costa, lambda la, lo: 100.0, sectores)
    d101 = ob.clasificar(_df([("HY-2", "2015-01-01", p)]), LAT0, LON0, costa, lambda la, lo: 100.5, sectores)
    assert d100.loc[0, "clase"] == "cuenca" and d101.loc[0, "clase"] == "sitio"


def test_borde_de_la_distancia_del_nodo_a_la_costa(sectores, costa):
    """Al norte (sector abierto), a la misma longitud que el nodo: igual distancia a la costa → «sitio»;
    un poco hacia la costa → ya no es «sitio»."""
    la = float(ex.destino(LAT0, LON0, 0.0, 10.0)[0])
    d = ob.clasificar(
        _df([("HY-2", "2015-01-01", (la, LON0)), ("HY-2", "2015-01-02", (la, LON0 + 0.01))]),
        LAT0,
        LON0,
        costa,
        _prof,
        sectores,
    )
    assert d.loc[0, "dist_costa_km"] == pytest.approx(d.attrs["dist_costa_nodo_km"], abs=1e-9)
    assert list(d["clase"]) == ["sitio", "región"]


def test_borde_de_10_km_de_la_costa_en_region(sectores, costa):
    """Hacia el S (abierto), a 150 km: a 10.1 km de la costa es «región»; a 9.9 km, «cuenca»."""
    la = LAT0 - 150 / 111.195
    filas = [("HY-2", "2015-01-01", (la, -75.5 - 10.1 / KX)), ("HY-2", "2015-01-02", (la, -75.5 - 9.9 / KX))]
    d = ob.clasificar(_df(filas), LAT0, LON0, costa, _prof, sectores)
    assert d["dist_costa_km"].round(1).tolist() == [10.1, 9.9]
    assert set(d["clase_sector"]) <= {"abierto", "parcial"}
    assert list(d["clase"]) == ["región", "cuenca"]
