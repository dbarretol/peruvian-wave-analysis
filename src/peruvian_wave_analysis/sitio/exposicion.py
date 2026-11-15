"""Exposición por sectores y obstáculos alrededor del nodo (contrato de la F1 §3.1 y §3.2; F01-C-01, C-10, N4).

Geometría: esfera de 6371 km; rayos por gran círculo desde el nodo. La dirección de cada sector es
la de **donde viene el oleaje**, que coincide con el rumbo del rayo trazado desde el nodo hacia mar adentro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely

R_TIERRA_KM = 6371.0
N_SECTORES = 16
ANCHO_SECTOR = 360.0 / N_SECTORES  # 22.5°
DESVIOS_CUNA = np.arange(-11, 12, 1.0)  # 23 rayos por sector, cada 1°
ALCANCE_KM, PASO_KM = 1000.0, 1.0
ALCANCE_CUENCA_KM, PASO_CUENCA_KM = 12000.0, 5.0
LAT_OCEANO_AUSTRAL = -40.0
LAT_PACIFICO_NORTE = 10.0
RADIO_OBSTACULOS_KM = 50.0
NOMBRES_SECTOR = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


class ErrorExposicion(RuntimeError):
    """El cálculo pidió algo fuera del insumo: no se trata como «sin tierra»."""


def destino(lat0: float, lon0: float, rumbo_deg, dist_km) -> tuple[np.ndarray, np.ndarray]:
    """Punto a ``dist_km`` por gran círculo con rumbo inicial ``rumbo_deg`` (vectorizado). Longitud en [−180, 180)."""
    f1, l1 = np.radians(lat0), np.radians(lon0)
    th = np.radians(rumbo_deg)
    d = np.asarray(dist_km, float) / R_TIERRA_KM
    f2 = np.arcsin(np.sin(f1) * np.cos(d) + np.cos(f1) * np.sin(d) * np.cos(th))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(f1), np.cos(d) - np.sin(f1) * np.sin(f2))
    return np.degrees(f2), (np.degrees(l2) + 180.0) % 360.0 - 180.0


def distancia_km(lat0, lon0, lat, lon) -> np.ndarray:
    f1, f2 = np.radians(lat0), np.radians(np.asarray(lat, float))
    dl = np.radians(np.asarray(lon, float) - lon0)
    a = np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(dl / 2) ** 2
    return 2 * R_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rumbo(lat0, lon0, lat, lon) -> np.ndarray:
    f1, f2 = np.radians(lat0), np.radians(np.asarray(lat, float))
    dl = np.radians(np.asarray(lon, float) - lon0)
    x = np.sin(dl) * np.cos(f2)
    y = np.cos(f1) * np.sin(f2) - np.sin(f1) * np.cos(f2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360.0


def sector_de(rumbo_deg) -> np.ndarray:
    """Sector de 22.5° de un rumbo, con intervalos semiabiertos [centro − 11.25°, centro + 11.25°) (F01-K-06).

    Así un rumbo en el límite exacto va siempre al sector siguiente en sentido horario. La cuña de rayos
    del contrato (§3.1) cubre ±11° alrededor del centro: los 0.25° de cada borde no tienen rayo propio.
    """
    return (np.floor((np.asarray(rumbo_deg, float) + ANCHO_SECTOR / 2) / ANCHO_SECTOR) % N_SECTORES).astype(int)


def _dentro(extension, lat, lon) -> bool:
    o, e, s, n = extension
    return bool(np.all((lon >= o) & (lon <= e) & (lat >= s) & (lat <= n)))


def _primer_toque(tierra, lat, lon) -> int | None:
    dentro = shapely.contains_xy(tierra, lon, lat)
    i = int(np.argmax(dentro))
    return i if dentro[i] else None


@dataclass
class Exposicion:
    rayos: pd.DataFrame  # una fila por rayo
    sectores: pd.DataFrame  # una fila por sector


def calcular(
    tierra, lat0: float, lon0: float, extension=(-180.0, 180.0, -90.0, 90.0), mascara_modelo=None
) -> Exposicion:
    """Exposición por sectores (§3.1).

    ``tierra``: geometría (lon, lat) preparada. ``extension``: (oeste, este, sur, norte) del insumo;
    un rayo que sale de ella es un error. ``mascara_modelo``: función (lat, lon) → bool «tierra para el
    modelo» y su extensión, como tupla (función, extensión), para comparar el abrigo del modelo.
    """
    filas = []
    d_loc = np.arange(PASO_KM, ALCANCE_KM + PASO_KM / 2, PASO_KM)
    d_cue = np.arange(PASO_CUENCA_KM, ALCANCE_CUENCA_KM + PASO_CUENCA_KM / 2, PASO_CUENCA_KM)
    for k in range(N_SECTORES):
        centro = k * ANCHO_SECTOR
        for desvio in DESVIOS_CUNA:
            rb = (centro + desvio) % 360.0
            la, lo = destino(lat0, lon0, rb, d_loc)
            if not _dentro(extension, la, lo):
                raise ErrorExposicion(
                    f"El rayo de rumbo {rb:.0f}° sale del insumo {extension} antes de {ALCANCE_KM:g} km."
                )
            i = _primer_toque(tierra, la, lo)
            dist = float(d_loc[i]) if i is not None else np.nan
            # Alcance de cuenca: hasta la primera tierra (o 12 000 km).
            lac, loc = destino(lat0, lon0, rb, d_cue)
            fuera = ~((loc >= extension[0]) & (loc <= extension[1]) & (lac >= extension[2]) & (lac <= extension[3]))
            j = _primer_toque(tierra, lac, loc)
            lim = j if j is not None else lac.size
            if fuera[:lim].any():
                raise ErrorExposicion(f"El rayo de rumbo {rb:.0f}° sale del insumo en el tramo de cuenca.")
            libre = lac[:lim]
            austral = bool((libre <= LAT_OCEANO_AUSTRAL).any())
            norte = bool((libre >= LAT_PACIFICO_NORTE).any())
            fila = {
                "sector": k,
                "nombre_sector": NOMBRES_SECTOR[k],
                "rumbo": rb,
                "dist_tierra_km": dist,
                "llega_1000_km": i is None,
                "abierto_oceano_austral": austral,
                "abierto_pacifico_norte": norte,
                "dist_tierra_cuenca_km": float(d_cue[j]) if j is not None else np.nan,
            }
            if mascara_modelo is not None:
                es_tierra_modelo, ext_m = mascara_modelo
                o, e, s, n = ext_m
                en = (lo >= o) & (lo <= e) & (la >= s) & (la <= n)
                hasta = int(np.argmin(en)) if not en.all() else en.size
                tm = np.array([es_tierra_modelo(a, b) for a, b in zip(la[:hasta], lo[:hasta])], bool)
                fila["dist_tierra_modelo_km"] = float(d_loc[int(np.argmax(tm))]) if tm.any() else np.nan
                fila["extension_malla_km"] = float(d_loc[hasta - 1]) if hasta else 0.0
            filas.append(fila)
    rayos = pd.DataFrame(filas)
    sect = rayos.groupby(["sector", "nombre_sector"], as_index=False).agg(
        fraccion_rayos_1000_km=("llega_1000_km", "mean"),
        dist_min_tierra_km=("dist_tierra_km", "min"),
        abierto_oceano_austral=("abierto_oceano_austral", "any"),
        abierto_pacifico_norte=("abierto_pacifico_norte", "any"),
    )
    sect["centro"] = sect["sector"] * ANCHO_SECTOR
    sect["clase"] = np.select(
        [sect["fraccion_rayos_1000_km"] == 1.0, sect["fraccion_rayos_1000_km"] == 0.0],
        ["abierto", "cerrado"],
        "parcial",
    )
    if mascara_modelo is not None:
        sect = sect.merge(
            rayos.groupby("sector", as_index=False).agg(dist_min_tierra_modelo_km=("dist_tierra_modelo_km", "min")),
            on="sector",
        )
    return Exposicion(rayos=rayos, sectores=sect)


def obstaculos(
    elev,
    tierra_islas,
    lat0: float,
    lon0: float,
    sectores: pd.DataFrame,
    es_tierra_modelo=None,
    radio_km: float = RADIO_OBSTACULOS_KM,
) -> pd.DataFrame:
    """Islas e islotes a menos de ``radio_km`` del nodo (§3.2).

    ``elev``: DataArray (lat, lon) de GEBCO; tierra = elevación > 0. El continente es la componente de
    mayor área del recorte; las demás son islas. ``tierra_islas``: polígonos de islas menores (lista de
    geometrías shapely) que se agregan si no coinciden con una isla de GEBCO.
    """
    from scipy import ndimage

    lat, lon = elev["lat"].values, elev["lon"].values
    t = elev.values > 0
    etiquetas, n = ndimage.label(t, structure=np.ones((3, 3)))
    filas = []
    if n:
        tamanos = ndimage.sum(t, etiquetas, index=np.arange(1, n + 1))
        continente = int(np.argmax(tamanos)) + 1
        dlat = float(np.median(np.diff(lat)))
        dlon = float(np.median(np.diff(lon)))
        for k in range(1, n + 1):
            if k == continente:
                continue
            ii, jj = np.nonzero(etiquetas == k)
            d = distancia_km(lat0, lon0, lat[ii], lon[jj])
            if d.min() > radio_km:
                continue
            cla, clo = float(lat[ii].mean()), float(lon[jj].mean())
            area = float(ii.size * (dlat * 111.195) * (dlon * 111.195 * np.cos(np.radians(cla))))
            filas.append(
                {
                    "origen": "GEBCO_2026",
                    "lat": cla,
                    "lon": clo,
                    "area_km2": area,
                    "dist_min_km": float(d.min()),
                    "rumbo": float(rumbo(lat0, lon0, cla, clo)),
                }
            )
    for g in tierra_islas or []:
        c = g.representative_point()
        d = float(distancia_km(lat0, lon0, c.y, c.x))
        if d > radio_km + 5:
            continue
        ya = any(distancia_km(f["lat"], f["lon"], c.y, c.x) < 1.0 for f in filas)
        if not ya:
            filas.append(
                {
                    "origen": "Natural Earth (islas menores)",
                    "lat": c.y,
                    "lon": c.x,
                    "area_km2": np.nan,
                    "dist_min_km": d,
                    "rumbo": float(rumbo(lat0, lon0, c.y, c.x)),
                }
            )
    df = pd.DataFrame(filas, columns=["origen", "lat", "lon", "area_km2", "dist_min_km", "rumbo"])
    if df.empty:
        return df.assign(
            sector=pd.Series(dtype=int), clase_sector=pd.Series(dtype=str), tierra_en_modelo=pd.Series(dtype=bool)
        )
    df["sector"] = sector_de(df["rumbo"])
    df["clase_sector"] = df["sector"].map(sectores.set_index("sector")["clase"])
    df["tierra_en_modelo"] = [
        bool(es_tierra_modelo(a, b)) if es_tierra_modelo else np.nan for a, b in zip(df["lat"], df["lon"])
    ]
    return df.sort_values("dist_min_km").reset_index(drop=True)
