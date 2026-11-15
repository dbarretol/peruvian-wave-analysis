"""Estratos estacionales y año de bloque, definidos por quien llama.

La biblioteca **no trae estaciones**: cada fase define sus estratos (meses de cada uno) y el mes en que empieza el
año de bloque. Reglas:

- cada mes pertenece a un solo estrato y todos los meses están cubiertos;
- cada estrato es un tramo de meses consecutivos (en el círculo del año);
- ningún estrato queda partido entre dos años de bloque: un estrato dic–feb exige que el año de bloque empiece en
  diciembre. Si no, es un error;
- el año de bloque se nombra por el año calendario de su mes de enero: con inicio en diciembre, dic-1993 pertenece
  al año de bloque 1994.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estratificacion:
    meses: tuple[tuple[str, tuple[int, ...]], ...]  # (estrato, meses), en orden
    mes_inicio: int = 1

    @classmethod
    def desde(cls, meses: Mapping[str, Sequence[int]], mes_inicio: int = 1) -> Estratificacion:
        e = cls(tuple((str(k), tuple(int(m) for m in v)) for k, v in meses.items()), int(mes_inicio))
        e.validar()
        return e

    @classmethod
    def mensual(cls, mes_inicio: int = 1) -> Estratificacion:
        return cls.desde({f"{m:02d}": [m] for m in range(1, 13)}, mes_inicio)

    @classmethod
    def anual(cls, mes_inicio: int = 1) -> Estratificacion:
        return cls.desde({"anual": list(range(1, 13))}, mes_inicio)

    @property
    def nombres(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.meses)

    def validar(self) -> None:
        if not 1 <= self.mes_inicio <= 12:
            raise ValueError(f"Mes de inicio fuera de 1..12: {self.mes_inicio}")
        todos = [m for _, ms in self.meses for m in ms]
        if sorted(todos) != list(range(1, 13)):
            raise ValueError("Cada mes debe pertenecer a un solo estrato y deben estar los 12")
        if len(set(self.nombres)) != len(self.nombres):
            raise ValueError("Nombres de estrato repetidos")
        orden = [(self.mes_inicio - 1 + i) % 12 + 1 for i in range(12)]  # meses en el orden del año de bloque
        pos = {m: i for i, m in enumerate(orden)}
        for nombre, ms in self.meses:
            p = sorted(pos[m] for m in ms)
            if p != list(range(p[0], p[0] + len(p))):
                circular = sorted(((m - 1) % 12) for m in ms)
                es_tramo = any(sorted((c - s) % 12 for c in circular) == list(range(len(ms))) for s in circular)
                if es_tramo:
                    raise ValueError(
                        f"El estrato «{nombre}» queda partido entre dos años de bloque con inicio en el "
                        f"mes {self.mes_inicio}; elija otro mes de inicio"
                    )
                raise ValueError(f"El estrato «{nombre}» no es un tramo de meses consecutivos")

    def _tabla(self) -> np.ndarray:
        t = np.empty(13, dtype=object)
        for nombre, ms in self.meses:
            for m in ms:
                t[m] = nombre
        return t

    def etiqueta(self, tiempos) -> np.ndarray:
        meses = pd.DatetimeIndex(tiempos).month.to_numpy()
        return self._tabla()[meses].astype(str)

    def codigo(self, tiempos) -> np.ndarray:
        """Índice del estrato en ``nombres`` (0..k−1)."""
        idx = {n: i for i, n in enumerate(self.nombres)}
        return np.array([idx[e] for e in self.etiqueta(tiempos)], dtype=int)

    def anio_bloque(self, tiempos) -> np.ndarray:
        t = pd.DatetimeIndex(tiempos)
        anio = t.year.to_numpy(dtype=int)
        if self.mes_inicio == 1:
            return anio
        return anio + (t.month.to_numpy() >= self.mes_inicio)

    def horas_ventana(self, anio_bloque: int, estrato: str) -> float:
        """Horas de la ventana del estrato en un año de bloque completo."""
        ms = dict(self.meses)[estrato]
        horas = 0
        for m in ms:
            anio = anio_bloque - 1 if (self.mes_inicio != 1 and m >= self.mes_inicio) else anio_bloque
            horas += calendar.monthrange(anio, m)[1] * 24
        return float(horas)
