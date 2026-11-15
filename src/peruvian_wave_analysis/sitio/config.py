"""Configuración del mapa de selección de sitio. Nada del sitio va fijo en el código."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATASET_WAVERYS = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
PASO_WAVERYS = 0.2  # grados; resolución nominal del producto


@dataclass
class ConfigBatimetria:
    fuente: str = "gmrt"  # "gmrt" (descarga) o "archivo" (NetCDF local, p. ej. GEBCO)
    resolucion_gmrt: str = "med"  # low | med | high | max
    archivo: str | None = None
    variable: str | None = None  # None: se detecta (altitude, elevation, z)


@dataclass
class ConfigWaverys:
    fuente: str = "copernicus"  # "copernicus" (descarga) o "archivo" (NetCDF local)
    dataset_id: str = DATASET_WAVERYS
    fecha: str = "2020-07-15T12:00:00"  # un solo paso de tiempo: basta para leer la malla y la máscara
    variable: str = "VHM0"
    archivo: str | None = None


@dataclass
class ConfigMapa:
    nombre: str
    latitud: float
    longitud: float
    isobatas: list[float] = field(default_factory=lambda: [200.0, 100.0, 50.0])
    isobata_principal: float | None = None  # None: la primera de la lista
    semiancho_grados: float = 1.0
    n_candidatos: int = 3  # ADR-0013: hasta 3 (vigente en ADR-0014)
    resguardo_km: float = 1.0  # ADR-0014: resguardo alrededor de las zonas de exclusión
    min_vecinas: int = 0  # ADR-0014: las vecinas no descartan salvo que se fije un mínimo (0 a 8)
    max_tierra: float = 1.0  # ADR-0014: fracción de tierra tolerada en la celda (1 = no descarta)
    zonas: bool = True  # False solo para pruebas o exploración rápida: el mapa lo declara
    directorio_datos: str | None = None  # None: variable de entorno WAVE_DATA_DIR
    batimetria: ConfigBatimetria = field(default_factory=ConfigBatimetria)
    waverys: ConfigWaverys = field(default_factory=ConfigWaverys)

    def __post_init__(self) -> None:
        if isinstance(self.batimetria, dict):
            self.batimetria = ConfigBatimetria(**self.batimetria)
        if isinstance(self.waverys, dict):
            self.waverys = ConfigWaverys(**self.waverys)
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.nombre):
            raise ValueError("nombre: solo minúsculas, dígitos, '-' y '_' (se usa en rutas de archivo).")
        if not -90 < self.latitud < 90 or not -180 <= self.longitud <= 180:
            raise ValueError("Coordenada de referencia fuera de rango.")
        self.isobatas = [abs(float(d)) for d in self.isobatas]
        if not self.isobatas or any(d <= 0 for d in self.isobatas):
            raise ValueError("isobatas: lista no vacía de profundidades positivas (m).")
        if len(set(self.isobatas)) != len(self.isobatas):
            raise ValueError("isobatas: hay profundidades repetidas.")
        if self.isobata_principal is None:
            self.isobata_principal = self.isobatas[0]
        self.isobata_principal = abs(float(self.isobata_principal))
        if self.isobata_principal not in self.isobatas:
            raise ValueError("isobata_principal debe estar en la lista de isobatas.")
        if not 0 < self.semiancho_grados <= 5:
            raise ValueError("semiancho_grados debe estar en (0, 5].")
        if self.resguardo_km < 0:
            raise ValueError("resguardo_km no puede ser negativo.")
        if not 0 <= self.max_tierra <= 1:
            raise ValueError("max_tierra debe estar entre 0 y 1.")
        if not 0 <= self.min_vecinas <= 8:
            raise ValueError("min_vecinas debe estar entre 0 y 8.")
        if self.n_candidatos < 1:
            raise ValueError("n_candidatos debe ser ≥ 1.")
        if self.batimetria.fuente not in {"gmrt", "archivo"}:
            raise ValueError("batimetria.fuente: 'gmrt' o 'archivo'.")
        if self.batimetria.fuente == "archivo" and not self.batimetria.archivo:
            raise ValueError("batimetria.fuente = 'archivo' exige batimetria.archivo.")
        if self.waverys.fuente not in {"copernicus", "archivo"}:
            raise ValueError("waverys.fuente: 'copernicus' o 'archivo'.")
        if self.waverys.fuente == "archivo" and not self.waverys.archivo:
            raise ValueError("waverys.fuente = 'archivo' exige waverys.archivo.")

    @property
    def recuadro(self) -> tuple[float, float, float, float]:
        """(oeste, este, sur, norte) en grados."""
        s = self.semiancho_grados
        return (self.longitud - s, self.longitud + s, self.latitud - s, self.latitud + s)

    def como_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_yaml(cls, ruta: str | Path) -> "ConfigMapa":
        datos = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
        return cls(**datos)
