"""Ingesta de la serie de WAVERYS del nodo y de la Hs de sus celdas vecinas (contrato de la F3, §3.1).

- Factor de escala, desplazamiento y ``_FillValue`` se decodifican **desde los atributos del archivo** y se comparan
  con los del producto (PUM v1.6). Un atributo distinto es un error: ningún valor se convierte por suposición.
- Las unidades se leen del archivo. Una variable requerida ausente es un error.
- El nodo se elige por coincidencia exacta de coordenadas, no por la celda más cercana.
- Cada archivo se recorta al leerlo (todas las variables en el nodo, solo Hs en las demás celdas), para no cargar el
  bloque completo en memoria.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C

_NOMBRES_LAT = ("latitude", "lat")
_NOMBRES_LON = ("longitude", "lon")
PASO = pd.Timedelta(hours=3)


class ErrorIngesta(Exception):
    """La serie no se puede leer tal como exige el contrato (variable, atributo, coordenada o tiempo)."""


@dataclass(frozen=True)
class Celda:
    lat: float
    lon: float
    anillo: int  # 0 = nodo, 1 = primer anillo, 2 = segundo

    @property
    def nombre(self) -> str:
        return f"{self.lat:+.1f}_{self.lon:+.1f}"


@dataclass
class Serie:
    nodo: pd.DataFrame  # índice de tiempo UTC; una columna por variable, en unidades físicas
    hs_celdas: pd.DataFrame  # índice de tiempo; una columna de Hs por celda del bloque (Celda.nombre)
    celdas: list[Celda]


def _renombrar(ds: xr.Dataset) -> xr.Dataset:
    def nombre(opciones: Sequence[str]) -> str:
        for o in opciones:
            if o in ds.coords or o in ds.dims:
                return o
        raise ErrorIngesta(f"El archivo no tiene coordenada {opciones[0]}")

    return ds.rename({nombre(_NOMBRES_LAT): "lat", nombre(_NOMBRES_LON): "lon"})


def comprobar_variable(da: xr.DataArray, nombre: str) -> None:
    """Unidad, factor de escala, desplazamiento y _FillValue iguales a los del producto."""
    unidad = da.attrs.get("units")
    if unidad != C.UNIDADES[nombre]:
        raise ErrorIngesta(f"{nombre}: unidad {unidad!r}, se esperaba {C.UNIDADES[nombre]!r}")
    enc = da.encoding
    escala = enc.get("scale_factor")
    if escala is None or not np.isclose(float(escala), C.ESCALA, rtol=0, atol=1e-12):
        raise ErrorIngesta(f"{nombre}: scale_factor {escala!r}, se esperaba {C.ESCALA}")
    desp = float(enc.get("add_offset", 0.0))
    if not np.isclose(desp, C.DESPLAZAMIENTO[nombre], rtol=0, atol=1e-9):
        raise ErrorIngesta(f"{nombre}: add_offset {desp}, se esperaba {C.DESPLAZAMIENTO[nombre]}")
    relleno = enc.get("_FillValue", enc.get("missing_value"))
    if relleno is None or int(relleno) != C.RELLENO:
        raise ErrorIngesta(f"{nombre}: _FillValue {relleno!r}, se esperaba {C.RELLENO}")


def indice_exacto(valores: np.ndarray, x: float, tol: float = C.TOL_COORD) -> int:
    i = np.flatnonzero(np.abs(np.asarray(valores, dtype=float) - x) <= tol)
    if i.size != 1:
        raise ErrorIngesta(
            f"La coordenada {x} no coincide exactamente con ninguna de la malla ({i.size} coincidencias)"
        )
    return int(i[0])


def _anillo(lat: float, lon: float, nodo: tuple[float, float]) -> int:
    return int(round(max(abs(lat - nodo[0]), abs(lon - nodo[1])) / C.PASO_MALLA))


def leer_archivo(ruta: Path, nodo: tuple[float, float] = C.NODO) -> tuple[pd.DataFrame, pd.DataFrame, list[Celda]]:
    """(variables en el nodo, Hs por celda, celdas) de un archivo anual."""
    with xr.open_dataset(ruta, mask_and_scale=True) as ds0:
        ds = _renombrar(ds0)
        faltan = [v for v in C.VARIABLES if v not in ds.data_vars]
        if faltan:
            raise ErrorIngesta(f"{Path(ruta).name}: faltan variables requeridas {faltan}")
        for v in C.VARIABLES:
            comprobar_variable(ds[v], v)
        i = indice_exacto(ds["lat"].values, nodo[0])
        j = indice_exacto(ds["lon"].values, nodo[1])
        tiempo = pd.DatetimeIndex(ds["time"].values, name="tiempo")
        nodo_df = pd.DataFrame({v: ds[v].isel(lat=i, lon=j).values.astype(float) for v in C.VARIABLES}, index=tiempo)
        hs = ds["VHM0"].values.astype(float)  # (tiempo, lat, lon)
        celdas, columnas = [], {}
        for a, la in enumerate(ds["lat"].values):
            for b, lo in enumerate(ds["lon"].values):
                c = Celda(round(float(la), 6), round(float(lo), 6), _anillo(float(la), float(lo), nodo))
                celdas.append(c)
                columnas[c.nombre] = hs[:, a, b]
        return nodo_df, pd.DataFrame(columnas, index=tiempo), celdas


def leer_serie(rutas: Sequence[Path], nodo: tuple[float, float] = C.NODO) -> Serie:
    """Concatena los archivos anuales y comprueba la rejilla de tiempo (3 h, ordenada, sin duplicados ni huecos)."""
    if not rutas:
        raise ErrorIngesta("No hay archivos de la serie")
    partes = [leer_archivo(r, nodo) for r in rutas]
    celdas = partes[0][2]
    if any(p[2] != celdas for p in partes[1:]):
        raise ErrorIngesta("Los archivos no tienen las mismas celdas")
    nodo_df = pd.concat([p[0] for p in partes])
    hs = pd.concat([p[1] for p in partes])
    comprobar_tiempo(nodo_df.index)
    return Serie(nodo_df, hs, celdas)


def comprobar_tiempo(indice: pd.DatetimeIndex) -> None:
    if indice.has_duplicates:
        raise ErrorIngesta("Hay tiempos duplicados")
    if not indice.is_monotonic_increasing:
        raise ErrorIngesta("Los tiempos no están ordenados")
    if len(indice) > 1 and not (np.diff(indice.values) == np.timedelta64(PASO)).all():
        raise ErrorIngesta("La rejilla de tiempo no es de 3 h completa")


def comprobar_contra_gate(serie: Serie, gate: dict) -> None:
    """Mismo número de pasos y mismos extremos que el gate de la serie de la F1 (G2)."""
    idx = serie.nodo.index
    if len(idx) != int(gate["pasos"]):
        raise ErrorIngesta(f"{len(idx)} pasos; el gate de la F1 registra {gate['pasos']}")
    if idx[0] != pd.Timestamp(gate["inicio"]) or idx[-1] != pd.Timestamp(gate["fin"]):
        raise ErrorIngesta(f"Extremos {idx[0]}–{idx[-1]}; el gate de la F1 registra {gate['inicio']}–{gate['fin']}")
    if serie.nodo.isna().any().any():
        raise ErrorIngesta("Hay ausentes en el nodo: el gate de la F1 no los registra (error de ingesta)")


def vecinas_con_dato(serie: Serie, anillo: int = 1) -> list[Celda]:
    """Celdas del anillo dado con al menos un dato de Hs."""
    return [c for c in serie.celdas if c.anillo == anillo and serie.hs_celdas[c.nombre].notna().any()]
