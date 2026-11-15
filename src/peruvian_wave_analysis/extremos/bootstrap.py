"""Bootstrap por bloques no solapados de un año completo (Carlstein, 1986).

- **Bloque = un año de bloque completo con todos sus estratos** (``Estratificacion.anio_bloque``). Se remuestrean
  años enteros con reemplazo: los estratos de un mismo año viajan juntos y se conserva su dependencia.
- **El estadístico recibe la lista de bloques remuestreados** y calcula dentro de cada bloque lo que dependa del
  tiempo: ningún intervalo, hueco ni desfase cruza de un bloque a otro.
- Remuestrear cada estrato por separado solo se permite si quien llama declara independientes los estratos.
- **Supuesto:** años independientes y del mismo régimen. Si hay dependencia entre años (por ejemplo por el ENSO), el
  IC sale estrecho; la fase que lo usa mide y reporta la autocorrelación entre años.
- Con ξ̂ ≤ −0.5 o ajuste en la frontera, el IC se marca no regular: el bootstrap de estadísticos que dependen del
  máximo muestral no es consistente.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .estratos import Estratificacion


def bloques_anuales(serie: pd.Series | pd.DataFrame, est: Estratificacion) -> dict[int, Any]:
    """Divide una serie indexada por tiempo en años de bloque completos, en orden."""
    anio = est.anio_bloque(serie.index)
    return {int(a): serie.iloc[np.flatnonzero(anio == a)] for a in np.unique(anio)}


def bloques_por_estrato(serie: pd.Series | pd.DataFrame, est: Estratificacion) -> dict[tuple[int, str], Any]:
    """Bloques (año de bloque, estrato), solo para el remuestreo por estrato con estratos declarados independientes."""
    anio, etq = est.anio_bloque(serie.index), est.etiqueta(serie.index)
    claves = sorted(set(zip(anio.tolist(), etq.tolist(), strict=True)), key=lambda c: (c[0], est.nombres.index(c[1])))
    return {(int(a), str(s)): serie.iloc[np.flatnonzero((anio == a) & (etq == s))] for a, s in claves}


def remuestrear_claves(claves: list, B: int, rng: np.random.Generator, por_estrato: bool = False) -> list[list]:
    if not por_estrato:
        return [[claves[i] for i in rng.integers(0, len(claves), len(claves))] for _ in range(B)]
    grupos: dict[str, list] = {}
    for c in claves:
        grupos.setdefault(c[1], []).append(c)
    reps = []
    for _ in range(B):
        r: list = []
        for s in grupos:
            g = grupos[s]
            r.extend(g[i] for i in rng.integers(0, len(g), len(g)))
        reps.append(r)
    return reps


def bootstrap_bloques(
    bloques: Mapping[Any, Any],
    estadistico: Callable[[list[Any]], Any],
    B: int,
    rng: np.random.Generator,
    estratos_independientes: bool = False,
) -> np.ndarray:
    """Réplicas del estadístico (forma B, o B × k si devuelve un vector).

    Con ``bloques_anuales`` (por defecto) se remuestrean años completos. Para remuestrear cada estrato por separado
    hay que pasar ``bloques_por_estrato`` **y** ``estratos_independientes=True``.
    """
    claves = list(bloques)
    por_estrato = bool(claves) and isinstance(claves[0], tuple)
    if por_estrato and not estratos_independientes:
        raise ValueError("Remuestrear por estrato exige declarar estratos_independientes=True")
    reps = remuestrear_claves(claves, B, rng, por_estrato)
    return np.asarray([estadistico([bloques[k] for k in r]) for r in reps], dtype=float)


@dataclass(frozen=True)
class IC:
    punto: float
    lo: float
    hi: float
    nivel: float
    n_validas: int
    n_no_finitas: int
    no_regular: bool = False
    estratos_independientes: bool = False

    @property
    def contiene_punto(self) -> bool:
        return bool(self.lo <= self.punto <= self.hi)


def ic_percentil(
    replicas, punto: float, *, no_regular: bool, nivel: float = 0.95, estratos_independientes: bool = False
) -> IC:
    """IC percentil. ``contiene_punto`` falso es un síntoma que la fase debe reportar.

    ``no_regular`` es obligatorio (contrato §3.7): para un estadístico de un ajuste GPD se pasa
    ``AjusteGPD.no_regular`` (ξ̂ ≤ −0.5 o frontera); para otro estadístico (una media, una tasa), ``False``."""
    r = np.asarray(replicas, dtype=float).ravel()
    ok = r[np.isfinite(r)]
    a = (1.0 - nivel) / 2.0
    lo, hi = np.quantile(ok, [a, 1.0 - a]) if ok.size else (np.nan, np.nan)
    return IC(
        float(punto),
        float(lo),
        float(hi),
        nivel,
        int(ok.size),
        int(r.size - ok.size),
        bool(no_regular),
        bool(estratos_independientes),
    )
