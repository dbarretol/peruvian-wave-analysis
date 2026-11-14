"""Tabla de nodos de WAVERYS candidatos (analysis/04 §5.1), sin las capas de zonas.

Distancias en una proyección local equirrectangular centrada en la coordenada de
referencia: con recuadros de pocos grados el error es menor al 1 %.

Criterios aplicados (analysis/04 §5.2, fijados antes de mirar el mapa):
1. nodo válido de WAVERYS con celda sin tierra;
3. ocho vecinas válidas;
5. entre los que cumplen, el más cercano a la isóbata principal.
Los criterios 2 y 4 (zonas de exclusión y colisión de usos) no se evalúan aquí.
"""
from __future__ import annotations

import warnings

import contourpy
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .config import ConfigMapa
from .waverys import Malla

RADIO_TIERRA_KM = 6371.0088


def proyectar(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    """(lon, lat) en grados → (x, y) en km respecto a (lon0, lat0)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = RADIO_TIERRA_KM * np.cos(np.radians(lat0)) * np.radians(lon - lon0)
    y = RADIO_TIERRA_KM * np.radians(lat - lat0)
    return np.column_stack([x.ravel(), y.ravel()])


def isolineas(elev: xr.DataArray, nivel: float) -> list[np.ndarray]:
    """Isolíneas de elevación ``nivel`` como lista de arreglos (n, 2) de (lon, lat)."""
    gen = contourpy.contour_generator(
        x=elev["lon"].values, y=elev["lat"].values, z=np.ma.masked_invalid(elev.values),
        line_type=contourpy.LineType.Separate,
    )
    return [ln for ln in gen.lines(nivel) if len(ln) >= 2]


def _densificar(lineas: list[np.ndarray], paso_grados: float) -> np.ndarray:
    """Vértices de las líneas con puntos intermedios cada ``paso_grados`` como máximo."""
    pts = []
    for ln in lineas:
        for a, b in zip(ln[:-1], ln[1:]):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / paso_grados)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            pts.append(a + t * (b - a))
        pts.append(ln[-1:])
    return np.vstack(pts) if pts else np.empty((0, 2))


def distancia_a_lineas(lon, lat, lineas: list[np.ndarray], lon0: float, lat0: float) -> np.ndarray:
    """Distancia (km) de cada punto a la línea más cercana; NaN si no hay líneas."""
    lon = np.asarray(lon, dtype=float)
    if not lineas:
        return np.full(lon.shape, np.nan)
    vertices = _densificar(lineas, paso_grados=0.002)  # unos 200 m
    arbol = cKDTree(proyectar(vertices[:, 0], vertices[:, 1], lon0, lat0))
    d, _ = arbol.query(proyectar(lon, lat, lon0, lat0))
    return d.reshape(lon.shape)


def _estadisticas_celda(elev: xr.DataArray, lat: float, lon: float, paso: float) -> tuple[float, float, float]:
    """(profundidad mínima, profundidad media en agua, fracción de tierra) dentro de la celda del nodo."""
    h = paso / 2
    celda = elev.sel(lat=slice(lat - h, lat + h), lon=slice(lon - h, lon + h)).values
    celda = celda[np.isfinite(celda)]
    if celda.size == 0:
        return np.nan, np.nan, np.nan
    prof = -celda
    agua = prof[prof > 0]
    return float(prof.min()), float(agua.mean()) if agua.size else np.nan, float(np.mean(celda >= 0))


def _vecinas_validas(valido: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """(vecinas válidas, vecinas presentes en la malla descargada) de las ocho del nodo (i, j)."""
    n_validas = n_presentes = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            a, b = i + di, j + dj
            if 0 <= a < valido.shape[0] and 0 <= b < valido.shape[1]:
                n_presentes += 1
                n_validas += int(valido[a, b])
    return n_validas, n_presentes


def tabla(cfg: ConfigMapa, elev: xr.DataArray, malla: Malla) -> tuple[pd.DataFrame, dict[float, list[np.ndarray]]]:
    """Tabla de nodos dentro del recuadro y diccionario de isolíneas por profundidad (incluye la costa, 0 m)."""
    lon0, lat0 = cfg.longitud, cfg.latitud
    lineas = {0.0: isolineas(elev, 0.0)}
    for d in cfg.isobatas:
        lineas[d] = isolineas(elev, -d)
        if not lineas[d]:
            warnings.warn(f"La isóbata de {d:g} m no aparece en el recuadro con esta batimetría.", stacklevel=2)

    o, e, s, n = cfg.recuadro
    tol = 1e-6
    filas = []
    for i, la in enumerate(malla.lat):
        if not s - tol <= la <= n + tol:
            continue
        for j, lo in enumerate(malla.lon):
            if not o - tol <= lo <= e + tol:
                continue
            pmin, pmed, ftierra = _estadisticas_celda(elev, la, lo, malla.paso)
            nv, npres = _vecinas_validas(malla.valido, i, j)
            filas.append({
                "lat": round(float(la), 4),
                "lon": round(float(lo), 4),
                "valido": bool(malla.valido[i, j]),
                "prof_nodo_m": float(-elev.sel(lat=la, lon=lo, method="nearest")),
                "prof_min_celda_m": pmin,
                "prof_media_celda_m": pmed,
                "frac_tierra_celda": ftierra,
                "vecinas_validas": nv,
                "vecinas_presentes": npres,
            })
    df = pd.DataFrame(filas)
    if df.empty:
        raise ValueError("No hay nodos de WAVERYS dentro del recuadro.")

    df["dist_costa_km"] = distancia_a_lineas(df["lon"], df["lat"], lineas[0.0], lon0, lat0)
    for d in cfg.isobatas:
        df[f"dist_iso_{d:g}m_km"] = distancia_a_lineas(df["lon"], df["lat"], lineas[d], lon0, lat0)
    xy = proyectar(df["lon"], df["lat"], lon0, lat0)
    df["dist_referencia_km"] = np.hypot(xy[:, 0], xy[:, 1])

    principal = f"dist_iso_{cfg.isobata_principal:g}m_km"
    df["cumple_criterios_basicos"] = (
        df["valido"] & (df["frac_tierra_celda"] == 0) & (df["vecinas_validas"] == 8) & (df["vecinas_presentes"] == 8)
    )
    df = df.sort_values(
        ["cumple_criterios_basicos", principal, "dist_referencia_km"], ascending=[False, True, True], na_position="last"
    ).reset_index(drop=True)
    orden = np.where(df["cumple_criterios_basicos"], np.cumsum(df["cumple_criterios_basicos"]), 0)
    df.insert(0, "candidato", pd.array(np.where(orden > 0, orden, pd.NA), dtype="Int64"))
    return df, lineas
