"""Línea de costa de referencia: Natural Earth 1:10 m (contrato de la F1 §1, F01-C-01 y C-10).

Se usan las capas ``ne_10m_land`` y ``ne_10m_minor_islands`` en GeoJSON, de una versión fija del
repositorio oficial de Natural Earth. Se registran en ``raw/costa/natural_earth/`` con su hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import shapely
from shapely.geometry import shape

from .. import manifiesto

VERSION_NE = "5.1.2"
CAPAS_NE = ("ne_10m_land", "ne_10m_minor_islands")
URL_NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v{version}/geojson/{capa}.geojson"


def _bajar(url: str) -> bytes:
    """Punto de sustitución en las pruebas."""
    import requests

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def obtener_natural_earth(raiz_raw: Path, version: str = VERSION_NE) -> list[Path]:
    """Descarga (o verifica) las dos capas y devuelve sus rutas en raw."""
    rutas = []
    for capa in CAPAS_NE:
        destino = raiz_raw / "costa" / "natural_earth" / f"{capa}_v{version}.geojson"
        if destino.exists():
            manifiesto.verificar(raiz_raw, destino)
        else:
            url = URL_NE.format(version=version, capa=capa)
            contenido = _bajar(url)
            json.loads(contenido)  # que sea GeoJSON válido antes de registrarlo
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(contenido)
            manifiesto.registrar(
                raiz_raw,
                destino,
                {
                    "fuente": "Natural Earth",
                    "producto": f"{capa} (1:10 m)",
                    "version": version,
                    "servicio": url,
                    "licencia": "dominio público (naturalearthdata.com)",
                    "modulo": "peruvian_wave_analysis.sitio.costa",
                },
            )
        rutas.append(destino)
    return rutas


def leer_tierra(rutas: list[Path]):
    """Une todos los polígonos de las capas en una geometría preparada (lon, lat en grados)."""
    geoms = []
    for r in rutas:
        datos = json.loads(Path(r).read_text(encoding="utf-8"))
        geoms += [shape(f["geometry"]) for f in datos["features"] if f.get("geometry")]
    tierra = shapely.union_all(shapely.make_valid(geoms))
    shapely.prepare(tierra)
    return tierra
