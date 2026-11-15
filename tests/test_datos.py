"""F3: ingesta y reglas sin el detector de picos (contrato de la F3, §3.1–§3.3, §3.6, §3.7 y §5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from peruvian_wave_analysis.datos import config as C
from peruvian_wave_analysis.datos import ingesta, reglas

LATS = np.round(np.arange(-15.4, -14.59, 0.2), 1)
LONS = np.round(np.arange(-76.2, -75.39, 0.2), 1)


def _valores(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    hs = np.round(1.5 + 0.5 * rng.random(n), 2)
    return {
        "VHM0": hs,
        "VHM0_WW": np.round(0.3 * hs, 2),
        "VHM0_SW1": np.round(0.8 * hs, 2),
        "VHM0_SW2": np.round(0.4 * hs, 2),
        "VTPK": np.full(n, 14.0),
        "VTM02": np.full(n, 8.0),
        "VTM10": np.full(n, 10.0),
        "VTM01_WW": np.full(n, 4.0),
        "VTM01_SW1": np.full(n, 12.0),
        "VTM01_SW2": np.full(n, 15.0),
        "VMDR": np.full(n, 200.0),
        "VMDR_WW": np.full(n, 160.0),
        "VMDR_SW1": np.full(n, 205.0),
        "VMDR_SW2": np.full(n, 230.0),
        "VPED": np.full(n, 210.0),
        "VSDX": np.full(n, 0.01),
        "VSDY": np.full(n, -0.02),
    }


def escribir_nc(ruta, tiempos, valores_nodo=None, tierra=(), cambiar=None, quitar=()):
    """NetCDF sintético con la codificación del producto (int16, factor 0.01, desplazamiento, _FillValue)."""
    n = len(tiempos)
    rng = np.random.default_rng(0)
    base = _valores(n, rng)
    data = {}
    for v in C.VARIABLES:
        if v in quitar:
            continue
        arr = np.broadcast_to(base[v][:, None, None], (n, LATS.size, LONS.size)).copy()
        if valores_nodo and v in valores_nodo:
            arr[:, 2, 2] = valores_nodo[v]
        for la, lo in tierra:
            arr[:, list(LATS).index(la), list(LONS).index(lo)] = np.nan
        data[v] = (("time", "latitude", "longitude"), arr, {"units": C.UNIDADES[v]})
    ds = xr.Dataset(data, coords={"time": tiempos, "latitude": LATS, "longitude": LONS})
    enc = {
        v: {"dtype": "int16", "scale_factor": C.ESCALA, "add_offset": C.DESPLAZAMIENTO[v], "_FillValue": C.RELLENO}
        for v in ds.data_vars
    }
    if cambiar:
        cambiar(ds, enc)
    ds.to_netcdf(ruta, encoding=enc)
    return ruta


@pytest.fixture
def tiempos():
    return pd.date_range("2000-01-01", periods=16, freq="3h")


# --- Ingesta (§3.1) -----------------------------------------------------------------------------


def test_ingesta_decodifica_exacto_y_elige_el_nodo(tmp_path, tiempos):
    hs = np.round(np.linspace(1.0, 2.5, len(tiempos)), 2)
    r = escribir_nc(tmp_path / "a.nc", tiempos, {"VHM0": hs, "VMDR": np.full(len(tiempos), 359.99)})
    s = ingesta.leer_serie([r])
    assert np.allclose(s.nodo["VHM0"].to_numpy(), hs, atol=1e-9)
    assert np.allclose(s.nodo["VMDR"], 359.99, atol=1e-9)  # desplazamiento de 180 aplicado
    assert len(s.celdas) == 25 and sum(c.anillo == 1 for c in s.celdas) == 8
    assert np.allclose(s.hs_celdas["-15.0_-75.8"].to_numpy(), hs, atol=1e-9)


def test_ingesta_concatena_y_compara_con_el_gate(tmp_path):
    t1 = pd.date_range("2000-01-01", periods=8, freq="3h")
    t2 = pd.date_range(t1[-1] + pd.Timedelta(hours=3), periods=8, freq="3h")
    s = ingesta.leer_serie([escribir_nc(tmp_path / "a.nc", t1), escribir_nc(tmp_path / "b.nc", t2)])
    gate = {"pasos": 16, "inicio": str(t1[0]), "fin": str(t2[-1])}
    ingesta.comprobar_contra_gate(s, gate)
    with pytest.raises(ingesta.ErrorIngesta):
        ingesta.comprobar_contra_gate(s, gate | {"pasos": 17})
    t3 = pd.date_range(t2[-1] + pd.Timedelta(hours=6), periods=4, freq="3h")  # hueco de un paso
    with pytest.raises(ingesta.ErrorIngesta, match="3 h"):
        ingesta.leer_serie([tmp_path / "a.nc", tmp_path / "b.nc", escribir_nc(tmp_path / "c.nc", t3)])


@pytest.mark.parametrize(
    "caso",
    [
        "falta",
        "unidad",
        "escala",
        "desplazamiento",
        "relleno",
    ],
)
def test_ingesta_rechaza_lo_que_no_coincide(tmp_path, tiempos, caso):
    def cambiar(ds, enc):
        if caso == "unidad":
            ds["VTPK"].attrs["units"] = "min"
        elif caso == "escala":
            enc["VHM0"]["scale_factor"] = 0.001
        elif caso == "desplazamiento":
            enc["VMDR"]["add_offset"] = 0.0
        elif caso == "relleno":
            enc["VHM0"]["_FillValue"] = -999

    r = escribir_nc(tmp_path / "a.nc", tiempos, cambiar=cambiar, quitar=("VTM10",) if caso == "falta" else ())
    with pytest.raises(ingesta.ErrorIngesta):
        ingesta.leer_serie([r])


def test_ingesta_exige_coincidencia_exacta_del_nodo(tmp_path, tiempos):
    r = escribir_nc(tmp_path / "a.nc", tiempos)
    with pytest.raises(ingesta.ErrorIngesta, match="no coincide"):
        ingesta.leer_serie([r], nodo=(-15.05, -75.8))


def test_vecinas_con_dato(tmp_path, tiempos):
    r = escribir_nc(tmp_path / "a.nc", tiempos, tierra=[(-14.8, -75.6)])
    s = ingesta.leer_serie([r])
    v = ingesta.vecinas_con_dato(s)
    assert len(v) == 7 and all(c.anillo == 1 for c in v)
    assert "-14.8_-75.6" not in {c.nombre for c in v}


# --- §3.2 -------------------------------------------------------------------------------------


def _df(n=200, **cols):
    rng = np.random.default_rng(1)
    base = pd.DataFrame(_valores(n, rng), index=pd.date_range("2000-01-01", periods=n, freq="3h"))
    for k, v in cols.items():
        base[k] = v
    return base


def test_particiones_ausentes_con_firma():
    df = _df()
    df.loc[df.index[:120], ["VHM0_SW2", "VTM01_SW2", "VMDR_SW2"]] = [0.0, 0.0, 180.0]
    r = reglas.particiones_ausentes(df)
    f = r.firmas["SW2"]
    assert f.aplica and f.periodo == 0.0 and f.direccion == 180.0 and f.fraccion == 1.0 and f.dominante
    assert r.marcas["particion_ausente_SW2"].sum() == 120 and not r.marcas["incoherencia_ausente_SW2"].any()
    assert r.serie.loc[df.index[:120], ["VTM01_SW2", "VMDR_SW2"]].isna().all().all()
    assert (r.serie.loc[df.index[:120], "VHM0_SW2"] == 0.0).all()  # la altura se conserva
    assert not r.firmas["WW"].aplica and not r.revision_requerida  # sin ceros, la regla no aplica
    # los límites físicos no se disparan por la partición ausente
    assert not reglas.fuera_de_limites(r.serie).any().any()


def test_particiones_ausentes_sin_firma_marca_o_detiene():
    df = _df()
    df.loc[df.index[:100], ["VHM0_WW", "VTM01_WW", "VMDR_WW"]] = [0.0, 0.0, 180.0]
    df.loc[df.index[100], ["VHM0_WW", "VTM01_WW", "VMDR_WW"]] = [0.0, 3.0, 90.0]  # 1 % sin firma
    r = reglas.particiones_ausentes(df)
    assert r.firmas["WW"].dominante and r.marcas["incoherencia_ausente_WW"].sum() == 1
    df.loc[df.index[101:110], ["VHM0_WW", "VTM01_WW", "VMDR_WW"]] = [0.0, 5.0, 45.0]  # sin par dominante
    assert reglas.particiones_ausentes(df).revision_requerida


def test_hs_total_cero_se_marca():
    df = _df()
    df.loc[df.index[5], "VHM0"] = 0.0
    assert reglas.particiones_ausentes(df).marcas["hs_total_cero"].sum() == 1


def test_acumulaciones():
    rng = np.random.default_rng(2)
    x = np.round(rng.uniform(5.0, 15.0, 20000), 2)
    assert reglas.acumulaciones(x) == []
    x[:600] = 12.34  # ≈ 30 veces la frecuencia de los cuantos vecinos (≈ 20 por cuanto)
    assert reglas.acumulaciones(x) == [12.34]


def test_discreto_y_acumulaciones_discretas():
    f = 0.035 * 1.1 ** np.arange(30)
    tp = np.round(1.0 / f, 2)
    x = np.random.default_rng(3).choice(tp[:20], 50000)
    assert reglas.es_discreto(x) and not reglas.es_discreto(np.round(np.random.default_rng(4).uniform(3, 20, 5000), 2))
    assert reglas.acumulaciones_discretas(x) == []  # ser discreto no es acumular
    assert np.allclose(reglas.razones_consecutivas(x), 1.1, atol=0.01)


def test_saturacion():
    f = 0.035 * 1.1 ** np.arange(30)
    tp = pd.Series(np.round(1.0 / f, 2))
    sat = reglas.saturados(tp, discreto=True)
    assert sat.sum() == 1 and float(tp[sat].iloc[0]) == pytest.approx(28.57, abs=0.01)
    sin_fmin = tp[tp < 27.0]  # el sitio no llega al bin de f_min: mayor ≈ 25.97 s
    assert not reglas.saturados(sin_fmin, discreto=True).any()
    cont = pd.Series([28.50, 28.53, 28.57, 20.0])  # umbral 1/0.035 − 0.05 ≈ 28.521 s
    assert reglas.saturados(cont, discreto=False).tolist() == [False, True, True, False]


# --- §3.3 -------------------------------------------------------------------------------------


def test_limites_fisicos_y_direcciones():
    df = _df(10)
    df.loc[df.index[0], "VHM0"] = 20.0  # en el borde: dentro
    df.loc[df.index[1], "VHM0"] = 20.01
    df.loc[df.index[2], "VTPK"] = 30.0  # en el borde: dentro
    df.loc[df.index[3], "VTPK"] = 30.01
    df.loc[df.index[4], "VTM02"] = 0.0
    df.loc[df.index[5], "VMDR"] = 360.0
    df = reglas.normalizar_direcciones(df)
    assert df.loc[df.index[5], "VMDR"] == 0.0
    m = reglas.fuera_de_limites(df)
    assert m.any(axis=1).tolist() == [False, True, False, True, True, False, False, False, False, False]


def test_limite_de_peralte_a_mano():
    assert reglas.limite_peralte([8.0, 11.5, 15.0, 5.0, 20.0]) == pytest.approx(
        [1 / 15, (1 / 15 + 1 / 25) / 2, 1 / 25, 1 / 15, 1 / 25]
    )
    df = _df(2)
    df["VTPK"] = 8.0
    lim_hs = C.PERALTE_LIM[0] * C.G * 64.0 / (2 * np.pi)  # Hs que da exactamente S_p = 1/15 con Tp = 8 s
    df["VHM0"] = [lim_hs - 0.01, lim_hs + 0.01]
    assert reglas.peralte_excedido(df).tolist() == [False, True]


# --- §3.6 -------------------------------------------------------------------------------------


def test_coherencia():
    df = _df(4)
    df["VHM0"] = 2.0
    df[["VHM0_WW", "VHM0_SW1", "VHM0_SW2"]] = [[0.8, 1.6, 1.0], [2.02, 0.5, 0.0], [0.5, 0.5, 0.5], [0.8, 1.6, 1.0]]
    df["VTM02"] = [8.0, 8.0, 8.0, 10.5]
    df["VTM10"] = [10.0, 10.0, 10.0, 10.0]
    df["VTPK"] = [14.0, 14.0, 14.0, 9.0]
    c = reglas.coherencia(df)
    assert c["particion_mayor"].tolist() == [False, True, False, False]
    assert c["energia"].tolist() == [False, False, True, False]  # 4.0 frente a 4.2 (5 %) y frente a 0.75
    assert c["tm10_menor_tm02"].tolist() == [False, False, False, True]
    assert c["tp_menor_tm02"].tolist() == [False, False, False, True]


# --- §3.7 -------------------------------------------------------------------------------------


def test_cobertura_bloque_incompleto_y_parcial():
    idx = pd.date_range("2025-01-01", "2026-02-28 21:00", freq="3h")
    v = pd.Series(True, index=idx)
    ene = v.index[(v.index.year == 2025) & (v.index.month == 1)]  # 248 registros
    v.loc[ene[:25]] = False  # 223/248 = 89.9 %: incompleto
    feb = v.index[(v.index.year == 2025) & (v.index.month == 2)]  # 224 registros
    v.loc[feb[:22]] = False  # 202/224 = 90.2 %: completo
    anual, mensual = reglas.cobertura(v)
    assert bool(mensual.loc[(2025, 1), "incompleto"]) and not bool(mensual.loc[(2025, 2), "incompleto"])
    assert bool(anual.loc[2026, "parcial"]) and not bool(anual.loc[2025, "parcial"])
