"""Serie de WAVERYS del bloque alrededor del nodo elegido (ADR-0016; contrato de la F1 §3.5 y gate §4.2).

Se descarga un archivo por año, con la versión del dataset explícita y reanudación: un año ya
descargado se verifica por hash y no se vuelve a pedir. La cobertura, la versión y la lista de
variables se leen del catálogo de Copernicus Marine el día de la descarga, no de los atributos del
archivo (riesgo de la F1 en analysis/02), y quedan en el manifiesto.

El gate (``verificar``) exige:
- eje temporal completo, cada 3 h, entre el inicio y el fin de la cobertura anotada, sin duplicados;
- dato en el nodo en el 100 % de los pasos para las variables obligatorias;
- máscara de celdas con dato igual a la de la malla ya registrada;
- la misma versión de dataset en todos los archivos.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import manifiesto
from .batimetria import _NOMBRES_LAT, _NOMBRES_LON, _fmt, _primer_nombre
from .waverys import PRODUCTO_WAVERYS

DATASET = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
VARIABLES_ADR_0016 = ["VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VPED",
                      "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"]
OBLIGATORIAS = ["VHM0", "VTPK", "VMDR", "VTM02", "VTM10"]  # contrato §4.2 (F01-C-04)
PASO = pd.Timedelta(hours=3)


class ErrorSerie(RuntimeError):
    """La serie no cumple el gate o el catálogo no coincide con lo decidido."""


@dataclass
class Cobertura:
    """Lo que dice el catálogo el día de la consulta."""
    dataset_id: str
    version: str
    inicio: str  # ISO, UTC
    fin: str
    paso_h: float
    variables: list[str]
    consulta_utc: str


@dataclass
class ConfigSerie:
    nodo_lat: float = -15.0
    nodo_lon: float = -75.8
    # Bloque de 5 × 5 (ADR-0016): -15.4 a -14.6 y -76.2 a -75.4.
    oeste: float = -76.2
    este: float = -75.4
    sur: float = -15.4
    norte: float = -14.6
    dataset_id: str = DATASET
    version: str = "202411"
    variables: list[str] = field(default_factory=lambda: list(VARIABLES_ADR_0016))
    obligatorias: list[str] = field(default_factory=lambda: list(OBLIGATORIAS))


# --- Catálogo ----------------------------------------------------------------------------------

def _describir(dataset_id: str) -> dict:
    """Punto de sustitución en las pruebas: devuelve el JSON de ``copernicusmarine describe``."""
    import copernicusmarine

    return copernicusmarine.describe(dataset_id=dataset_id).model_dump()


def consultar_catalogo(cfg: ConfigSerie) -> Cobertura:
    """Cobertura, versión y variables del dataset según el catálogo. Falla si no coinciden con lo decidido."""
    d = _describir(cfg.dataset_id)
    for p in d.get("products", []):
        for ds in p.get("datasets", []):
            if ds.get("dataset_id") != cfg.dataset_id:
                continue
            for v in ds.get("versions", []):
                if v.get("label") != cfg.version:
                    continue
                for part in v.get("parts", []):
                    for s in part.get("services", []):
                        if s.get("service_name") not in ("arco-time-series", "arco-geo-series"):
                            continue
                        variables = sorted({var["short_name"] for var in s["variables"]})
                        t = next(c for c in s["variables"][0]["coordinates"] if c["coordinate_id"] == "time")
                        a = lambda ms: pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
                        cob = Cobertura(cfg.dataset_id, cfg.version, a(t["minimum_value"]), a(t["maximum_value"]),
                                        t["step"] / 3.6e6, variables,
                                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                        _comprobar_catalogo(cfg, cob)
                        return cob
    raise ErrorSerie(f"El catálogo no tiene {cfg.dataset_id} en la versión {cfg.version}: hace falta una ADR.")


def _comprobar_catalogo(cfg: ConfigSerie, cob: Cobertura) -> None:
    faltan = sorted(set(cfg.variables) - set(cob.variables))
    sobran = sorted(set(cob.variables) - set(cfg.variables))
    if faltan or sobran:
        raise ErrorSerie(f"Las variables del catálogo difieren de ADR-0016 (faltan {faltan}, sobran {sobran}): "
                         "se detiene y hace falta una ADR.")
    if abs(cob.paso_h - 3.0) > 1e-9:
        raise ErrorSerie(f"El paso del catálogo es {cob.paso_h} h, no 3 h.")


# --- Descarga ----------------------------------------------------------------------------------

def _subset(**params) -> None:
    """Punto de sustitución en las pruebas."""
    import copernicusmarine

    copernicusmarine.subset(**params)


def ruta_anio(raiz_raw: Path, cfg: ConfigSerie, anio: int) -> Path:
    return (raiz_raw / "waverys" / "serie"
            / f"waverys_{cfg.version}_{anio}_lat{_fmt(cfg.sur)}_{_fmt(cfg.norte)}_lon{_fmt(cfg.oeste)}_{_fmt(cfg.este)}.nc")


def anios(cob: Cobertura) -> list[int]:
    return list(range(pd.Timestamp(cob.inicio).year, pd.Timestamp(cob.fin).year + 1))


def descargar_anio(anio: int, cfg: ConfigSerie, cob: Cobertura, raiz_raw: Path) -> tuple[Path, dict]:
    """Descarga (o reanuda) un año. Un archivo ya registrado se verifica por hash y no se vuelve a pedir."""
    destino = ruta_anio(raiz_raw, cfg, anio)
    if destino.exists():
        return destino, manifiesto.verificar(raiz_raw, destino)
    ini = max(pd.Timestamp(f"{anio}-01-01T00:00Z"), pd.Timestamp(cob.inicio))
    fin = min(pd.Timestamp(f"{anio}-12-31T21:00Z"), pd.Timestamp(cob.fin))
    params = dict(
        dataset_id=cfg.dataset_id, dataset_version=cfg.version, variables=list(cfg.variables),
        minimum_longitude=cfg.oeste, maximum_longitude=cfg.este,
        minimum_latitude=cfg.sur, maximum_latitude=cfg.norte,
        start_datetime=ini.strftime("%Y-%m-%dT%H:%M:%S"), end_datetime=fin.strftime("%Y-%m-%dT%H:%M:%S"),
        coordinates_selection_method="outside",
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.stem + ".parcial.nc")
    if temporal.exists():
        temporal.unlink()  # restos de un intento interrumpido: no están registrados
    try:
        _subset(**params, output_directory=str(temporal.parent), output_filename=temporal.name,
                disable_progress_bar=True, raise_if_updating=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falló la descarga de {anio} ({exc}). Si faltan credenciales: `copernicusmarine login`.") from exc
    if not temporal.exists():
        raise RuntimeError(f"copernicusmarine no creó {temporal}.")
    temporal.replace(destino)  # solo un archivo completo llega a su nombre final
    origen = {
        "fuente": "Copernicus Marine Service", "producto": PRODUCTO_WAVERYS,
        "servicio": "copernicusmarine.subset (Copernicus Marine Toolbox)",
        "parametros": params, "version_dataset": cfg.version, "catalogo": asdict(cob),
        "modulo": "peruvian_wave_analysis.sitio.serie",
    }
    return destino, manifiesto.registrar(raiz_raw, destino, origen)


# --- Gate --------------------------------------------------------------------------------------

def _abrir(rutas: list[Path]) -> xr.Dataset:
    partes = []
    for r in rutas:
        ds = xr.open_dataset(r)
        ds = ds.rename({_primer_nombre(_NOMBRES_LAT, ds.coords): "lat", _primer_nombre(_NOMBRES_LON, ds.coords): "lon"})
        partes.append(ds.sortby("lat").sortby("lon"))
    return xr.concat(partes, dim="time", data_vars="all", coords="minimal", compat="override", join="override")


def verificar(rutas: list[Path], cob: Cobertura, cfg: ConfigSerie, raiz_raw: Path,
              valido_malla: xr.DataArray) -> dict:
    """Gate de la serie (contrato §4.2). Devuelve el informe; lanza ErrorSerie si algo falla.

    ``valido_malla``: máscara de la malla registrada (True = con dato), con coordenadas lat y lon.
    """
    fallas: list[str] = []
    versiones = {manifiesto.verificar(raiz_raw, r).get("version_dataset") for r in rutas}  # verifica también el hash
    if versiones != {cob.version}:
        fallas.append(f"versiones de dataset distintas o inesperadas: {sorted(map(str, versiones))}")

    ds = _abrir(rutas)
    try:
        t = pd.DatetimeIndex(ds["time"].values)
        esperado = pd.date_range(pd.Timestamp(cob.inicio).tz_convert(None), pd.Timestamp(cob.fin).tz_convert(None), freq=PASO)
        duplicados = int(t.duplicated().sum())
        faltantes = esperado.difference(t)
        sobrantes = t.difference(esperado)
        if duplicados:
            fallas.append(f"{duplicados} pasos de tiempo duplicados")
        if len(faltantes):
            fallas.append(f"{len(faltantes)} pasos faltantes (primero {faltantes[0]})")
        if len(sobrantes):
            fallas.append(f"{len(sobrantes)} pasos fuera de la cobertura (primero {sobrantes[0]})")

        nodo = ds.sel(lat=cfg.nodo_lat, lon=cfg.nodo_lon, method="nearest", tolerance=1e-3)
        faltan_nodo = {}
        for v in cfg.variables:
            if v not in ds:
                fallas.append(f"falta la variable {v}")
                continue
            frac = float(np.isnan(nodo[v].values).mean())
            faltan_nodo[v] = round(frac, 6)
            if v in cfg.obligatorias and frac > 0:
                fallas.append(f"{v} sin dato en el nodo en {100 * frac:.4f} % de los pasos")

        con_dato = ds["VHM0"].notnull().any("time")
        ref = valido_malla.sel(lat=con_dato["lat"], lon=con_dato["lon"], method="nearest", tolerance=1e-3)
        distintas = [(float(a), float(b)) for a, b, x, y in zip(
            np.repeat(con_dato["lat"].values, con_dato.sizes["lon"]), np.tile(con_dato["lon"].values, con_dato.sizes["lat"]),
            con_dato.transpose("lat", "lon").values.ravel(), ref.transpose("lat", "lon").values.ravel()) if bool(x) != bool(y)]
        if distintas:
            fallas.append(f"máscara distinta de la malla registrada en {distintas}")
        informe = {
            "pasos": int(t.size), "pasos_esperados": int(esperado.size), "duplicados": duplicados,
            "faltantes": int(len(faltantes)), "fuera_de_cobertura": int(len(sobrantes)),
            "inicio": str(t.min()) if t.size else None, "fin": str(t.max()) if t.size else None,
            "fraccion_sin_dato_en_el_nodo": faltan_nodo, "celdas_con_dato": int(con_dato.values.sum()),
            "celdas": int(con_dato.size), "versiones": sorted(map(str, versiones)), "archivos": len(rutas),
            "fallas": fallas, "cumple": not fallas,
        }
    finally:
        ds.close()
    if fallas:
        raise ErrorSerie("Gate de la serie: " + "; ".join(fallas) + ".\n" + json.dumps(informe, ensure_ascii=False, default=str))
    return informe
