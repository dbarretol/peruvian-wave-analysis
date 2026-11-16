"""Registro de un recorte de GEBCO descargado a mano (ADR-0011).

GEBCO no ofrece un servicio para pedir un recuadro desde el código. El recorte se descarga en
https://download.gebco.net/ (formato NetCDF) y esta función lo copia a
``WAVE-DATA/raw/batimetria/gebco/`` y lo registra en el manifiesto con su hash.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import xarray as xr

from .. import manifiesto, rutas
from .batimetria import _NOMBRES_ELEVACION, _NOMBRES_LAT, _NOMBRES_LON, _primer_nombre

FUENTE_GEBCO = "GEBCO Compilation Group"
URL_DESCARGA = "https://download.gebco.net/"


def registrar(archivo: str | Path, raiz_datos: Path, version: str) -> tuple[Path, dict]:
    """Copia el NetCDF de GEBCO a raw/ y lo registra. Devuelve (ruta en raw, entrada del manifiesto)."""
    origen = Path(archivo)
    if not origen.exists():
        raise FileNotFoundError(f"No existe {origen}")
    with xr.open_dataset(origen) as ds:
        var = _primer_nombre(_NOMBRES_ELEVACION, ds.data_vars)
        lat = ds[_primer_nombre(_NOMBRES_LAT, ds.coords)].values
        lon = ds[_primer_nombre(_NOMBRES_LON, ds.coords)].values
        atributos = {k: str(ds.attrs.get(k, "")) for k in ("title", "institution", "source", "history", "Conventions")}
    raiz_raw = rutas.raw(raiz_datos)
    destino = raiz_raw / "batimetria" / "gebco" / origen.name
    if destino.exists():
        return destino, manifiesto.verificar(raiz_raw, destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)
    entrada = manifiesto.registrar(
        raiz_raw,
        destino,
        {
            "fuente": FUENTE_GEBCO,
            "producto": version,
            "servicio": f"{URL_DESCARGA} (descarga manual)",
            "archivo_original": origen.name,
            "variable": var,
            "recuadro_oeste_este_sur_norte": [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
            "atributos": atributos,
            "modulo": "peruvian_wave_analysis.sitio.gebco",
        },
    )
    return destino, entrada
