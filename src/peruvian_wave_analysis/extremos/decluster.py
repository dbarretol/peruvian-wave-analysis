"""Declustering por rachas sobre el eje de tiempo completo.

La serie debe estar en su rejilla temporal completa: un registro ausente es una fila con NaN, nunca una fila
eliminada. Si se eliminaran filas (por ejemplo al filtrar una población), los huecos partirían tormentas; por eso
una rejilla incompleta es un error. Las etiquetas (estrato, tipo de mar) se asignan a los picos después.

Reglas:

- una tormenta nueva empieza si pasan **más** de IET horas desde la excedencia anterior; empate en el máximo: el
  primero;
- un registro NaN cuenta como tiempo sin excedencia al medir el IET;
- una tormenta con algún NaN entre IET horas antes de su inicio e IET horas después de su fin (extremos incluidos)
  se marca ``toca_hueco``: su pico pudo estar en el hueco, o el hueco pudo partirla o unirla con otra.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNAS_PICO = ["exceso", "t_inicio", "t_fin", "n_excedencias", "toca_hueco"]


def _horas(indice: pd.DatetimeIndex) -> np.ndarray:
    return ((indice - indice[0]) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def comprobar_rejilla(indice: pd.DatetimeIndex, paso_horas: float) -> None:
    if not isinstance(indice, pd.DatetimeIndex):
        raise ValueError("El índice debe ser un DatetimeIndex")
    if not indice.is_monotonic_increasing or not indice.is_unique:
        raise ValueError("El índice temporal debe ser creciente y sin duplicados")
    if len(indice) > 1 and not np.allclose(np.diff(_horas(indice)), paso_horas):
        raise ValueError(f"La serie no está en su rejilla completa de {paso_horas} h: marque los registros "
                         "ausentes con NaN en vez de eliminarlos")


def decluster_rachas(df: pd.DataFrame, u: float, iet_horas: float, paso_horas: float = 3.0,
                     columna: str = "Hs") -> pd.DataFrame:
    """Picos independientes por rachas, indexados por la fecha del pico.

    Devuelve todas las columnas de ``df`` en el instante del pico más ``exceso``, ``t_inicio``, ``t_fin``,
    ``n_excedencias`` y ``toca_hueco``.
    """
    comprobar_rejilla(df.index, paso_horas)
    v = df[columna].to_numpy(dtype=float)
    hueco = np.isnan(v)
    exc = np.flatnonzero(np.nan_to_num(v, nan=-np.inf) > u)
    columnas = list(df.columns) + COLUMNAS_PICO
    if exc.size == 0:
        return pd.DataFrame(columns=columnas, index=pd.DatetimeIndex([], tz=df.index.tz, name="t_pico"))
    t_h = _horas(df.index)[exc]
    nueva = np.concatenate([[True], np.diff(t_h) > iet_horas])
    grupo = np.cumsum(nueva) - 1
    inicios = np.flatnonzero(nueva)
    fines = np.concatenate([inicios[1:], [exc.size]])
    pos_pico = np.array([exc[s + int(np.argmax(v[exc[s:e]]))] for s, e in zip(inicios, fines)])
    pos_ini, pos_fin = exc[inicios], exc[fines - 1]
    k = int(np.floor(iet_horas / paso_horas + 1e-9))
    acum = np.concatenate([[0], np.cumsum(hueco)])
    a = np.clip(pos_ini - k, 0, len(v))
    b = np.clip(pos_fin + k + 1, 0, len(v))
    toca = (acum[b] - acum[a]) > 0
    # el borde de la serie también es un hueco: no se sabe qué pasó antes del primer registro ni después del último
    toca |= (pos_ini - k < 0) | (pos_fin + k >= len(v))
    out = df.iloc[pos_pico].copy()
    out.index.name = "t_pico"
    out["exceso"] = out[columna].to_numpy() - u
    out["t_inicio"] = df.index[pos_ini]
    out["t_fin"] = df.index[pos_fin]
    out["n_excedencias"] = np.bincount(grupo)
    out["toca_hueco"] = toca
    return out
