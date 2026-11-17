"""WAVERYS en la región (círculo de 200 km del nodo) y en las celdas de las boyas: descarga por año.

Contrato de la F5 (v6), §1:
- **Regional:** ``VHM0`` en las celdas del círculo de 200 km, en todo el período de la serie del nodo.
- **Boyas:** ``VHM0``, ``VTPK`` y ``VTM02`` alrededor de cada boya, en sus años. La etapa 1 elige la celda con dato más
  cercana; el recuadro de ±1° cubre los recambios del fondeo de 32012.

**Producto y versión por tramo.** Se validan los datos de la serie del nodo, así que se descarga exactamente su
producto: los tramos (dataset, versión, inicio y fin) se leen del catálogo anotado en el manifiesto de
``raw/waverys/serie/`` el día de la descarga del nodo. Si la serie del nodo combina más de un dataset (por ejemplo, un
tramo interino servido aparte), se descarga cada tramo con el suyo, y cada archivo anota su dataset, su versión y el
tramo en el manifiesto. El catálogo de hoy debe tener ese dataset en esa versión y cubrir el tramo.

**G4.** Nada aquí lee valores de observación; los archivos son del modelo. Aun así, ``estructura`` y los mensajes solo
dan dimensiones, variables y fechas extremas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import manifiesto
from ..sitio import serie
from ..sitio.batimetria import _fmt
from ..sitio.waverys import PRODUCTO_WAVERYS
from .ndbc import BOYAS, Boya

NODO = (-15.0, -75.8)
RADIO_KM = 200.0  # el de la F1 y la F4
PASO_GRADOS = 0.2
KM_POR_GRADO = 111.195
MARGEN_BOYA_GRADOS = 1.0
VARIABLES_REGION = ("VHM0",)
VARIABLES_BOYA = ("VHM0", "VTPK", "VTM02")


class ErrorDescarga(RuntimeError):
    """El producto pedido no coincide con el de la serie del nodo, o falta información para descargar."""


@dataclass(frozen=True)
class Tramo:
    dataset_id: str
    version: str
    inicio: str  # ISO, UTC
    fin: str


@dataclass(frozen=True)
class Recuadro:
    nombre: str  # «regional» o «boya_<estación>»
    oeste: float
    este: float
    sur: float
    norte: float
    variables: tuple[str, ...]
    anios: tuple[int, ...] | None = None  # None: todo el período de los tramos


def recuadro_regional(lat0: float = NODO[0], lon0: float = NODO[1], radio_km: float = RADIO_KM) -> Recuadro:
    """Recuadro que contiene el círculo, más una celda de margen."""
    dlat = radio_km / KM_POR_GRADO + PASO_GRADOS
    dlon = radio_km / (KM_POR_GRADO * np.cos(np.radians(lat0))) + PASO_GRADOS
    r = lambda v: float(np.round(v, 3))  # noqa: E731
    return Recuadro("regional", r(lon0 - dlon), r(lon0 + dlon), r(lat0 - dlat), r(lat0 + dlat), VARIABLES_REGION)


def recuadro_boya(boya: Boya, margen: float = MARGEN_BOYA_GRADOS) -> Recuadro:
    return Recuadro(
        f"boya_{boya.estacion}",
        boya.lon - margen,
        boya.lon + margen,
        boya.lat - margen,
        boya.lat + margen,
        VARIABLES_BOYA,
        boya.anios,
    )


def recuadros() -> list[Recuadro]:
    return [recuadro_regional(), *(recuadro_boya(b) for b in BOYAS)]


# --- Tramos de la serie del nodo -----------------------------------------------------------------------


def tramos_del_nodo(raiz_raw: Path) -> list[Tramo]:
    """Tramos (dataset, versión, cobertura) anotados en el manifiesto de la serie del nodo, en orden de inicio."""
    tramos = set()
    for rel, e in manifiesto.leer(raiz_raw).items():
        if not rel.replace("\\", "/").startswith("waverys/serie/"):
            continue
        c = e.get("catalogo")
        if not c:
            raise ErrorDescarga(f"{Path(rel).name}: el manifiesto de la serie del nodo no anota el catálogo.")
        tramos.add(Tramo(c["dataset_id"], c["version"], c["inicio"], c["fin"]))
    if not tramos:
        raise ErrorDescarga("No hay serie del nodo en el manifiesto: hace falta `serie-waverys` (F1).")
    orden = sorted(tramos, key=lambda t: pd.Timestamp(t.inicio))
    for a, b in zip(orden, orden[1:], strict=False):
        if pd.Timestamp(b.inicio) <= pd.Timestamp(a.fin):
            raise ErrorDescarga(f"Tramos de la serie del nodo solapados: {a} y {b}.")
    return orden


def comprobar_catalogo(tramo: Tramo) -> serie.Cobertura:
    """El catálogo de hoy debe tener el dataset del tramo en su versión y cubrir el tramo."""
    cob = serie.consultar_catalogo(serie.ConfigSerie(dataset_id=tramo.dataset_id, version=tramo.version))
    if pd.Timestamp(cob.inicio) > pd.Timestamp(tramo.inicio) or pd.Timestamp(cob.fin) < pd.Timestamp(tramo.fin):
        raise ErrorDescarga(
            f"El catálogo de hoy de {tramo.dataset_id} v{tramo.version} ({cob.inicio} → {cob.fin}) no cubre el tramo "
            f"de la serie del nodo ({tramo.inicio} → {tramo.fin})."
        )
    return cob


def pedidos(rec: Recuadro, tramos: list[Tramo]) -> list[tuple[int, Tramo, pd.Timestamp, pd.Timestamp]]:
    """(año, tramo, inicio, fin) de cada archivo: uno por año y tramo, recortado a la cobertura del tramo."""
    out = []
    for tr in tramos:
        t0, t1 = pd.Timestamp(tr.inicio), pd.Timestamp(tr.fin)
        for anio in range(t0.year, t1.year + 1):
            if rec.anios is not None and anio not in rec.anios:
                continue
            ini = max(pd.Timestamp(f"{anio}-01-01T00:00Z"), t0)
            fin = min(pd.Timestamp(f"{anio}-12-31T21:00Z"), t1)
            if ini <= fin:
                out.append((anio, tr, ini, fin))
    return out


def ruta_archivo(raiz_raw: Path, rec: Recuadro, tramo: Tramo, anio: int) -> Path:
    carpeta = "regional" if rec.nombre == "regional" else "boyas"
    return (
        Path(raiz_raw) / "waverys" / carpeta / f"waverys_{rec.nombre}_{tramo.dataset_id}_{tramo.version}_{anio}"
        f"_lat{_fmt(rec.sur)}_{_fmt(rec.norte)}_lon{_fmt(rec.oeste)}_{_fmt(rec.este)}.nc"
    )


# --- Descarga ---------------------------------------------------------------------------------------


def _subset(**params) -> None:
    """Punto de sustitución en las pruebas."""
    import copernicusmarine  # noqa: PLC0415 (importación diferida: solo hace falta al descargar)

    copernicusmarine.subset(**params)


def descargar(
    raiz_raw: Path, rec: Recuadro, tramo: Tramo, cob: serie.Cobertura, anio: int, ini: pd.Timestamp, fin: pd.Timestamp
) -> tuple[Path, dict]:
    """Descarga (o verifica por hash) un año de un tramo. Solo un archivo completo y comprobado llega a su nombre."""
    destino = ruta_archivo(raiz_raw, rec, tramo, anio)
    if destino.exists():
        return destino, manifiesto.verificar(raiz_raw, destino)
    params = dict(
        dataset_id=tramo.dataset_id,
        dataset_version=tramo.version,
        variables=list(rec.variables),
        minimum_longitude=rec.oeste,
        maximum_longitude=rec.este,
        minimum_latitude=rec.sur,
        maximum_latitude=rec.norte,
        start_datetime=ini.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=fin.strftime("%Y-%m-%dT%H:%M:%S"),
        coordinates_selection_method="outside",
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.stem + ".parcial.nc")
    if temporal.exists():
        temporal.unlink()
    try:
        _subset(
            **params,
            output_directory=str(temporal.parent),
            output_filename=temporal.name,
            disable_progress_bar=True,
            raise_if_updating=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Falló la descarga de {rec.nombre} {anio} ({type(exc).__name__}). "
            "Si faltan credenciales: `copernicusmarine login`."
        ) from exc
    if not temporal.exists():
        raise RuntimeError(f"copernicusmarine no creó {temporal.name}.")
    est = estructura(temporal)
    fallas = comprobar_estructura(est, rec, ini, fin)
    if fallas:
        temporal.unlink()
        raise ErrorDescarga(f"{destino.name}: " + "; ".join(fallas) + ".")
    temporal.replace(destino)
    origen = {
        "fuente": "Copernicus Marine Service",
        "producto": PRODUCTO_WAVERYS,
        "servicio": "copernicusmarine.subset (Copernicus Marine Toolbox)",
        "parametros": params,
        "dataset_id": tramo.dataset_id,
        "version_dataset": tramo.version,
        "tramo_serie_nodo": asdict(tramo),
        "catalogo": asdict(cob),
        "recuadro": rec.nombre,
        "modulo": "peruvian_wave_analysis.validacion.waverys",
    }
    return destino, manifiesto.registrar(raiz_raw, destino, origen)


# --- Comprobación de estructura (sin valores) ---------------------------------------------------------


def estructura(ruta: Path) -> dict:
    """Dimensiones, variables y extremos de las coordenadas. No lee los datos de ninguna variable."""
    with xr.open_dataset(ruta) as ds:
        t = pd.DatetimeIndex(ds["time"].values)
        lat = next(c for c in ("latitude", "lat") if c in ds.coords)
        lon = next(c for c in ("longitude", "lon") if c in ds.coords)
        return {
            "archivo": Path(ruta).name,
            "dimensiones": {k: int(v) for k, v in ds.sizes.items()},
            "variables": sorted(map(str, ds.data_vars)),
            "inicio": str(t.min()) if len(t) else None,
            "fin": str(t.max()) if len(t) else None,
            "lat": [float(ds[lat].min()), float(ds[lat].max())],
            "lon": [float(ds[lon].min()), float(ds[lon].max())],
        }


def comprobar_estructura(est: dict, rec: Recuadro, ini: pd.Timestamp, fin: pd.Timestamp) -> list[str]:
    fallas = []
    faltan = sorted(set(rec.variables) - set(est["variables"]))
    if faltan:
        fallas.append(f"faltan las variables {faltan}")
    esperado = pd.date_range(ini.tz_convert(None), fin.tz_convert(None), freq="3h")
    if est["inicio"] is None or pd.Timestamp(est["inicio"]) != esperado[0] or pd.Timestamp(est["fin"]) != esperado[-1]:
        fallas.append(f"fechas {est['inicio']} → {est['fin']}, se esperaba {esperado[0]} → {esperado[-1]}")
    if est["dimensiones"].get("time") != len(esperado):
        fallas.append(f"{est['dimensiones'].get('time')} pasos de tiempo, se esperaban {len(esperado)}")
    tol = PASO_GRADOS / 2 + 1e-6
    if est["lat"][0] > rec.sur + tol or est["lat"][1] < rec.norte - tol:
        fallas.append(f"latitudes {est['lat']} no cubren [{rec.sur}, {rec.norte}]")
    if est["lon"][0] > rec.oeste + tol or est["lon"][1] < rec.este - tol:
        fallas.append(f"longitudes {est['lon']} no cubren [{rec.oeste}, {rec.este}]")
    return fallas
