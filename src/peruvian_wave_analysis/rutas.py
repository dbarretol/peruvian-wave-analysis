"""Ubicación del directorio de datos (WAVE-DATA), fuera del repositorio de código."""
from __future__ import annotations

import os
from pathlib import Path

VARIABLE_ENTORNO = "WAVE_DATA_DIR"


def directorio_datos(explicito: str | Path | None = None) -> Path:
    """Devuelve la raíz de WAVE-DATA: el valor explícito o la variable de entorno ``WAVE_DATA_DIR``."""
    valor = explicito or os.environ.get(VARIABLE_ENTORNO)
    if not valor:
        raise RuntimeError(
            f"No se indicó el directorio de datos. Pásalo en la configuración (directorio_datos) "
            f"o define la variable de entorno {VARIABLE_ENTORNO}."
        )
    p = Path(valor).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"El directorio de datos no existe: {p}")
    return p


def raw(raiz: Path) -> Path:
    p = Path(raiz) / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def interim(raiz: Path, fase: str, *partes: str) -> Path:
    p = Path(raiz).joinpath("interim", fase, *partes)
    p.mkdir(parents=True, exist_ok=True)
    return p
