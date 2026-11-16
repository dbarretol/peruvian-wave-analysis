"""Carga de la serie de la F3 para las fases siguientes (contrato de la F3, §2).

``cargar_serie`` aplica **siempre** ``valido``: los registros descartados llegan con todas sus variables ausentes.
La serie sin descartar (por ejemplo, para la sensibilidad de la F9 o la prueba de altímetros de la F4) solo se
obtiene pidiéndola de forma explícita con ``sin_descartar=True``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import rutas
from ..comun import etapa
from . import config as C
from .run import NOMBRE, ruta_candidatos, ruta_serie


def cargar_serie(
    raiz_datos: str | Path | None = None, nombre: str = NOMBRE, sin_descartar: bool = False
) -> pd.DataFrame:
    raiz = rutas.directorio_datos(raiz_datos)
    if sin_descartar:
        d = ruta_candidatos(raiz, nombre)
        etapa.exigir_publicada(d)
        return pd.read_parquet(d / "serie_decodificada.parquet")
    d = ruta_serie(raiz, nombre)
    etapa.exigir_publicada(d)
    df = pd.read_parquet(d / "serie_nodo.parquet")
    df.loc[~df["valido"], list(C.VARIABLES)] = float("nan")  # garantía, aunque ya vengan ausentes
    return df
