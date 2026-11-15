"""Malla de WAVERYS (Copernicus Marine, GLOBAL_MULTIYEAR_WAV_001_032) alrededor del sitio.

Las coordenadas de los nodos se leen de un archivo del producto; no se suponen
(analysis/04 §3.2). Un nodo es válido si tiene dato en el paso de tiempo descargado;
los enmascarados como tierra quedan como no válidos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .. import manifiesto, rutas
from .batimetria import _fmt, _primer_nombre, _NOMBRES_LAT, _NOMBRES_LON
from .config import PASO_WAVERYS, ConfigMapa

PRODUCTO_WAVERYS = "GLOBAL_MULTIYEAR_WAV_001_032"


@dataclass
class Malla:
    lat: np.ndarray  # (n_lat,) creciente
    lon: np.ndarray  # (n_lon,) creciente
    valido: np.ndarray  # (n_lat, n_lon) bool
    paso: float  # grados
    origen: dict
    etiqueta: str


def recuadro_malla(cfg: ConfigMapa) -> tuple[float, float, float, float]:
    """Recuadro más un nodo de margen, para conocer las ocho vecinas de los nodos del borde."""
    o, e, s, n = cfg.recuadro
    m = PASO_WAVERYS
    return (round(o - m, 3), round(e + m, 3), round(s - m, 3), round(n + m, 3))


def _descargar(cfg: ConfigMapa, raiz_raw: Path) -> tuple[Path, dict]:
    o, e, s, n = recuadro_malla(cfg)
    w = cfg.waverys
    fecha = w.fecha.replace(":", "").replace("-", "")
    destino = (
        raiz_raw
        / "waverys"
        / "malla"
        / f"waverys_{w.variable}_{fecha}_lat{_fmt(s)}_{_fmt(n)}_lon{_fmt(o)}_{_fmt(e)}.nc"
    )
    if destino.exists():
        return destino, manifiesto.verificar(raiz_raw, destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    import copernicusmarine  # importación diferida: es pesada y solo hace falta al descargar

    params = dict(
        dataset_id=w.dataset_id,
        variables=[w.variable],
        minimum_longitude=o,
        maximum_longitude=e,
        minimum_latitude=s,
        maximum_latitude=n,
        start_datetime=w.fecha,
        end_datetime=w.fecha,
        coordinates_selection_method="outside",
    )
    try:
        copernicusmarine.subset(**params, output_directory=str(destino.parent), output_filename=destino.name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Falló la descarga de WAVERYS ({exc}). Si faltan credenciales, ejecuta `copernicusmarine login`."
        ) from exc
    if not destino.exists():
        raise RuntimeError(f"copernicusmarine no creó {destino}.")
    with xr.open_dataset(destino) as ds:
        atributos = {
            k: str(ds.attrs.get(k, ""))
            for k in ("title", "product_version", "version", "source", "copernicusmarine_version")
        }
    origen = {
        "fuente": "Copernicus Marine Service",
        "producto": PRODUCTO_WAVERYS,
        "servicio": "copernicusmarine.subset (Copernicus Marine Toolbox)",
        "parametros": params,
        "atributos": atributos,
        "version_toolbox": getattr(copernicusmarine, "__version__", "desconocida"),
        "modulo": "peruvian_wave_analysis.sitio.waverys",
    }
    return destino, manifiesto.registrar(raiz_raw, destino, origen)


def leer_malla(
    ruta: Path, variable: str, recuadro: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lee (lat, lon, valido) de un archivo de WAVERYS, recortado al recuadro dado."""
    with xr.open_dataset(ruta) as ds:
        da = ds[variable]
        lat_n = _primer_nombre(_NOMBRES_LAT, da.coords)
        lon_n = _primer_nombre(_NOMBRES_LON, da.coords)
        da = da.rename({lat_n: "lat", lon_n: "lon"}).sortby("lat").sortby("lon")
        if "time" in da.dims:
            da = da.isel(time=0)
        o, e, s, n = recuadro
        tol = 1e-4
        da = da.sel(lat=slice(s - tol, n + tol), lon=slice(o - tol, e + tol)).transpose("lat", "lon").load()
    if da.sizes["lat"] == 0 or da.sizes["lon"] == 0:
        raise ValueError(f"El archivo de WAVERYS {ruta} no cubre el recuadro {recuadro}.")
    return da["lat"].values.astype(float), da["lon"].values.astype(float), np.isfinite(da.values)


def obtener(cfg: ConfigMapa, raiz_datos: Path) -> Malla:
    raiz_raw = rutas.raw(raiz_datos)
    if cfg.waverys.fuente == "copernicus":
        ruta, origen = _descargar(cfg, raiz_raw)
    else:
        ruta = Path(cfg.waverys.archivo)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de WAVERYS: {ruta}")
        try:
            ruta.resolve().relative_to(raiz_raw.resolve())
            origen = manifiesto.verificar(raiz_raw, ruta)
        except ValueError:
            origen = {"archivo": str(ruta), "sha256": manifiesto.sha256(ruta), "fecha_acceso_utc": "sin registrar"}
    lat, lon, valido = leer_malla(ruta, cfg.waverys.variable, recuadro_malla(cfg))
    pasos = np.concatenate([np.diff(lat), np.diff(lon)])
    paso = float(np.median(pasos)) if pasos.size else PASO_WAVERYS
    if pasos.size and not np.allclose(pasos, paso, atol=1e-3):
        raise ValueError(f"La malla de {ruta} no es regular (pasos {np.unique(pasos.round(4))}).")
    etiqueta = (
        f"Malla WAVERYS ({paso:.1f}°), {cfg.waverys.variable} {cfg.waverys.fecha[:10]},\n"
        f"  acceso {origen['fecha_acceso_utc'][:10]}"
    )
    return Malla(lat=lat, lon=lon, valido=valido, paso=paso, origen={"ruta": str(ruta), **origen}, etiqueta=etiqueta)
