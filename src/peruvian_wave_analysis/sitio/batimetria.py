"""Batimetría del recuadro: descarga desde GMRT o lectura de un NetCDF local (p. ej. GEBCO).

La batimetría se devuelve como ``xarray.DataArray`` de elevación (m, positiva hacia arriba,
negativa en el mar) con dimensiones ``(lat, lon)`` en orden creciente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import xarray as xr

from .. import manifiesto, rutas
from .config import ConfigMapa

URL_GMRT = "https://www.gmrt.org/services/GridServer"
MARGEN_GRADOS = 0.25  # la batimetría se pide más grande que el recuadro: cubre las celdas del borde
_NOMBRES_ELEVACION = ("altitude", "elevation", "z", "topo", "depth")
_NOMBRES_LAT = ("lat", "latitude", "y")
_NOMBRES_LON = ("lon", "longitude", "x")


@dataclass
class Batimetria:
    elevacion: xr.DataArray  # (lat, lon), m, positiva hacia arriba
    origen: dict  # entrada del manifiesto o descripción del archivo local
    etiqueta: str  # texto corto para la leyenda del mapa


def _fmt(v: float) -> str:
    return f"{'m' if v < 0 else 'p'}{abs(v):07.3f}".replace(".", "_")


def recuadro_con_margen(cfg: ConfigMapa) -> tuple[float, float, float, float]:
    o, e, s, n = cfg.recuadro
    m = MARGEN_GRADOS
    return (round(o - m, 3), round(e + m, 3), round(s - m, 3), round(n + m, 3))


def _descargar_gmrt(cfg: ConfigMapa, raiz_raw: Path) -> tuple[Path, dict]:
    o, e, s, n = recuadro_con_margen(cfg)
    res = cfg.batimetria.resolucion_gmrt
    destino = raiz_raw / "batimetria" / "gmrt" / f"gmrt_{res}_lat{_fmt(s)}_{_fmt(n)}_lon{_fmt(o)}_{_fmt(e)}.nc"
    if destino.exists():
        return destino, manifiesto.verificar(raiz_raw, destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    params = {"north": n, "south": s, "west": o, "east": e, "layer": "topo", "format": "coards", "resolution": res}
    tmp = destino.with_suffix(".parcial")
    with requests.get(URL_GMRT, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        nombre_servidor = r.headers.get("Content-Disposition", "")
    try:
        with xr.open_dataset(tmp) as ds:  # comprueba que es un NetCDF legible antes de registrarlo
            version = ds.attrs.get("history", "").splitlines()[:2]
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"GMRT no devolvió un NetCDF válido para {params}: {exc}") from exc
    tmp.rename(destino)
    origen = {
        "fuente": "GMRT (Global Multi-Resolution Topography), Lamont-Doherty Earth Observatory",
        "servicio": URL_GMRT,
        "parametros": params,
        "archivo_servidor": nombre_servidor,
        "historia": version,
        "modulo": "peruvian_wave_analysis.sitio.batimetria",
    }
    return destino, manifiesto.registrar(raiz_raw, destino, origen)


def _primer_nombre(candidatos, disponibles) -> str:
    for c in candidatos:
        if c in disponibles:
            return c
    raise KeyError(f"No se encontró ninguno de {candidatos} en {list(disponibles)}")


def normalizar(ds: xr.Dataset, variable: str | None, recuadro: tuple[float, float, float, float]) -> xr.DataArray:
    """Devuelve la elevación como (lat, lon) creciente, recortada al recuadro."""
    var = variable or _primer_nombre(_NOMBRES_ELEVACION, ds.data_vars)
    da = ds[var]
    lat = _primer_nombre(_NOMBRES_LAT, da.coords)
    lon = _primer_nombre(_NOMBRES_LON, da.coords)
    da = da.rename({lat: "lat", lon: "lon"}).sortby("lat").sortby("lon")
    o, e, s, n = recuadro
    da = da.sel(lat=slice(s, n), lon=slice(o, e))
    if da.sizes["lat"] < 2 or da.sizes["lon"] < 2:
        raise ValueError(f"La batimetría no cubre el recuadro {recuadro}.")
    if var == "depth":  # profundidad positiva hacia abajo → elevación
        da = -da
    return da.astype(np.float32).rename("elevacion").transpose("lat", "lon").load()


def obtener(cfg: ConfigMapa, raiz_datos: Path) -> Batimetria:
    recuadro = recuadro_con_margen(cfg)
    if cfg.batimetria.fuente == "gmrt":
        raiz_raw = rutas.raw(raiz_datos)
        ruta, origen = _descargar_gmrt(cfg, raiz_raw)
        etiqueta = f"Batimetría: GMRT ({cfg.batimetria.resolucion_gmrt}),\n  acceso {origen['fecha_acceso_utc'][:10]}"
    else:
        ruta = Path(cfg.batimetria.archivo)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de batimetría: {ruta}")
        raiz_raw = rutas.raw(raiz_datos)
        try:
            ruta.resolve().relative_to(raiz_raw.resolve())
            origen = manifiesto.verificar(raiz_raw, ruta)
        except ValueError:  # archivo fuera de WAVE-DATA/raw: se documenta su hash, pero no se registra
            origen = {"archivo": str(ruta), "sha256": manifiesto.sha256(ruta), "fecha_acceso_utc": "sin registrar"}
        etiqueta = f"Batimetría: {ruta.name},\n  acceso {origen['fecha_acceso_utc'][:10]}"
    with xr.open_dataset(ruta) as ds:
        elev = normalizar(ds, cfg.batimetria.variable, recuadro)
    return Batimetria(elevacion=elev, origen={"ruta": str(ruta), **origen}, etiqueta=etiqueta)
