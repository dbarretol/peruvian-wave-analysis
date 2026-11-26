"""Carpetas de salida de la F7 (contrato §2 y §5.0).

``etapa.publicar`` reemplaza la carpeta de salida entera. Por eso cada etapa publica en su propia carpeta y ninguna
carpeta de una etapa contiene la de otra: el conteo previo y la simulación son subcarpetas **hermanas** de
``simulacion-previa/``, y la simulación nunca borra el conteo.
"""

from __future__ import annotations

from pathlib import Path

FASE = "F07-eventos"


def raiz_fase(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE


def ruta_conteo_previo(raiz: Path) -> Path:
    return raiz_fase(raiz) / "simulacion-previa" / "conteo-previo"


def ruta_simulacion(raiz: Path) -> Path:
    return raiz_fase(raiz) / "simulacion-previa" / "simulacion"


def ruta_avance_simulacion(raiz: Path) -> Path:
    return Path(raiz) / "interim" / FASE / "simulacion-avance"
