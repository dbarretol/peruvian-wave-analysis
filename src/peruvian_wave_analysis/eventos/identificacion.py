"""Identificación de las tormentas, una sola vez y común a las ramas (contrato de la F7, §3.2).

1. **Rachas.** Excedencias estrictas de u_id (``Hs > u_id``) sobre el eje completo de 3 h. Una racha nueva empieza
   si pasan **más** de IET horas desde la última excedencia (la regla de ``extremos.decluster``). Un NaN cuenta como
   tiempo sin excedencia: un hueco no parte una tormenta. Los bordes de una racha son su primera y su última
   excedencia.
2. **Bajada intermedia** (rotulada: el 0.5 m se fijó en la revisión visual de la F6). Una parte [a, b] se corta en
   la posición m, con la parte izquierda [a, m] y la derecha [m + 1, b], si se cumplen **las dos** condiciones:

   - las crestas (el máximo de Hs de cada parte, primera aparición) están separadas por ≥ IET horas;
   - Hs(m) es el mínimo de Hs entre las dos crestas y **cada** cresta lo supera en ≥ 0.5 m.

   **Comprobación frente a las vecinas** (precisión anterior a los datos, confirmada por el usuario el 2026-11-20):
   la subparte que queda junto a una parte vecina ya cortada debe cumplir también las dos condiciones frente a ella
   (su cresta a ≥ IET de la cresta vecina y ≥ 0.5 m sobre el corte compartido). Sin esto, la recursión podía dejar
   como eventos distintos dos crestas a menos de IET.

   **Orden de los cortes** (fijado antes del conteo): en cada paso se aplica, entre todas las partes de la racha, el
   corte válido de bajada más honda, min(cresta izquierda, cresta derecha) − Hs(m); con empate, el de más a la
   izquierda. Se repite hasta que ningún corte cumpla. Es el sentido jerárquico de la regla y no depende del sentido
   del tiempo. (``orden="izquierda"`` existe solo para la prueba que muestra que el orden importa.)

   Un registro NaN no puede ser el mínimo de un corte. La racha original se conserva en ``id_racha`` (la prueba de
   Poisson corre sobre ella).
3. **Evento:** cada parte. Su pico en E0 sin corregir es el máximo de Hs en [t_inicio, t_fin] (primera aparición);
   ese instante asigna el evento a las ramas y al período.

``toca_hueco`` es la marca de ``extremos.decluster`` evaluada sobre la **racha** (algún NaN, o el borde de la serie,
entre IET horas antes de su inicio y IET horas después de su fin): un hueco pudo partirla o unirla con otra, así que
la marca es de la racha y la heredan todas sus partes. ``duracion_h`` = número de estados del evento × 3 h.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..extremos.decluster import comprobar_rejilla

PASO_H = 3.0
BAJADA_M = 0.5
TOL = 1e-9

COLUMNAS = [
    "id_evento",
    "id_racha",
    "t_inicio",
    "t_fin",
    "t_pico",
    "i_inicio",
    "i_fin",
    "i_pico",
    "n_excedencias",
    "partes_racha",
    "dividido_por_bajada",
    "toca_hueco",
    "duracion_h",
]


@dataclass(frozen=True)
class Rachas:
    ini: np.ndarray  # posición de la primera excedencia
    fin: np.ndarray  # posición de la última excedencia
    toca_hueco: np.ndarray


def rachas(v: np.ndarray, u: float, iet_horas: float, paso_horas: float = PASO_H) -> Rachas:
    """Rachas de excedencias estrictas de ``u`` con la regla del IET de ``extremos.decluster``."""
    v = np.asarray(v, dtype=float)
    exc = np.flatnonzero(np.nan_to_num(v, nan=-np.inf) > u)
    if exc.size == 0:
        vacio = np.array([], dtype=int)
        return Rachas(vacio, vacio, np.array([], dtype=bool))
    nueva = np.concatenate([[True], np.diff(exc) * paso_horas > iet_horas])
    ini_idx = np.flatnonzero(nueva)
    fin_idx = np.concatenate([ini_idx[1:], [exc.size]]) - 1
    ini, fin = exc[ini_idx], exc[fin_idx]
    k = int(np.floor(iet_horas / paso_horas + TOL))
    acum = np.concatenate([[0], np.cumsum(np.isnan(v))])
    a = np.clip(ini - k, 0, v.size)
    b = np.clip(fin + k + 1, 0, v.size)
    toca = ((acum[b] - acum[a]) > 0) | (ini - k < 0) | (fin + k >= v.size)
    return Rachas(ini, fin, toca)


def _arg_max_prefijo(x: np.ndarray) -> np.ndarray:
    """Posición del máximo de x[0..j] (primera aparición), con NaN como −∞."""
    out = np.empty(x.size, dtype=int)
    mejor, pos = -np.inf, 0
    for j, xj in enumerate(x):
        if xj > mejor:
            mejor, pos = xj, j
        out[j] = pos
    return out


def _arg_max_sufijo(x: np.ndarray) -> np.ndarray:
    """Posición del máximo de x[j..] (primera aparición), con NaN como −∞."""
    out = np.empty(x.size, dtype=int)
    mejor, pos = -np.inf, x.size - 1
    for j in range(x.size - 1, -1, -1):
        if x[j] >= mejor:
            mejor, pos = x[j], j
        out[j] = pos
    return out


@dataclass(frozen=True)
class Vecina:
    """Parte vecina ya cortada: posición de su pico (relativa a la parte que se corta) y Hs del corte compartido."""

    pico: int
    hs_corte: float


def _cumple(separacion_pasos: float, bajada_m: float, sep_pasos: float, bajada: float) -> bool:
    return separacion_pasos >= sep_pasos - TOL and bajada_m >= bajada - TOL


def corte(
    x: np.ndarray,
    sep_pasos: float,
    bajada: float = BAJADA_M,
    izquierda: Vecina | None = None,
    derecha: Vecina | None = None,
) -> tuple[int, float] | None:
    """(posición, profundidad) del corte válido de bajada más honda de una parte, o None. Empate: el primero.

    ``x`` es Hs de una parte; ``sep_pasos`` = IET / paso. El corte deja [0, m] y [m + 1, fin]. Con una vecina ya
    cortada, la subparte que queda junto a ella también debe cumplir las dos condiciones frente a ella. Un corte que
    falla frente a una vecina no detiene la búsqueda: se sigue con los demás.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return None
    xm = np.where(np.isnan(x), -np.inf, x)
    izq, der = _arg_max_prefijo(xm), _arg_max_sufijo(xm)
    mejor, prof_mejor = None, -np.inf
    for m in range(1, x.size - 1):
        if np.isnan(x[m]):
            continue
        tl, tr = izq[m], der[m + 1]
        if tr - tl < sep_pasos - TOL:
            continue
        if x[m] > np.nanmin(x[tl : tr + 1]):
            continue  # m no es el mínimo entre las crestas
        prof = min(x[tl], x[tr]) - x[m]
        if prof < bajada - TOL:
            continue
        if izquierda is not None and not _cumple(tl - izquierda.pico, x[tl] - izquierda.hs_corte, sep_pasos, bajada):
            continue
        if derecha is not None and not _cumple(derecha.pico - tr, x[tr] - derecha.hs_corte, sep_pasos, bajada):
            continue
        if prof > prof_mejor + TOL:
            mejor, prof_mejor = m, prof
    return None if mejor is None else (mejor, float(prof_mejor))


def mejor_corte(x, sep_pasos: float, bajada: float = BAJADA_M, izquierda=None, derecha=None) -> int | None:
    """Posición del corte de ``corte``, o None."""
    c = corte(x, sep_pasos, bajada, izquierda, derecha)
    return None if c is None else c[0]


def _pico(v: np.ndarray, s: int, e: int) -> int:
    return s + int(np.argmax(np.where(np.isnan(v[s : e + 1]), -np.inf, v[s : e + 1])))


def corte_de_parte(
    v: np.ndarray, partes: list[tuple[int, int]], i: int, sep_pasos: float, bajada: float = BAJADA_M
) -> tuple[int, float] | None:
    """Corte válido de la parte ``i`` frente a sus vecinas tal como están (posición relativa al inicio de la parte)."""
    s, e = partes[i]
    izq = Vecina(_pico(v, *partes[i - 1]) - s, float(v[s - 1])) if i > 0 else None
    der = Vecina(_pico(v, *partes[i + 1]) - s, float(v[e])) if i + 1 < len(partes) else None
    return corte(v[s : e + 1], sep_pasos, bajada, izq, der)


ORDENES = ("mas_honda", "izquierda")


def dividir(
    v: np.ndarray, a: int, b: int, sep_pasos: float, bajada: float = BAJADA_M, orden: str = "mas_honda"
) -> list[tuple[int, int]]:
    """Partes [a_i, b_i] de la racha [a, b] tras aplicar la bajada intermedia, en orden.

    ``orden="mas_honda"`` (la regla): en cada paso, el corte válido más hondo entre todas las partes (empate: el de
    más a la izquierda). ``orden="izquierda"``: la parte de más a la izquierda que admita un corte; solo para pruebas.
    """
    if orden not in ORDENES:
        raise ValueError(f"orden debe ser uno de {ORDENES}: {orden}")
    partes = [(a, b)]
    while True:
        elegido = None
        for i in range(len(partes)):
            c = corte_de_parte(v, partes, i, sep_pasos, bajada)
            if c is None:
                continue
            if orden == "izquierda":
                elegido = (i, c[0])
                break
            if elegido is None or c[1] > elegido[2] + TOL:
                elegido = (i, c[0], c[1])
        if elegido is None:
            return partes
        i, m = elegido[0], elegido[1]
        s, e = partes[i]
        partes[i : i + 1] = [(s, s + m), (s + m + 1, e)]


def identificar(
    hs: pd.Series,
    u_id: float,
    iet_horas: float,
    bajada: float | None = BAJADA_M,
    paso_horas: float = PASO_H,
    orden: str = "mas_honda",
) -> pd.DataFrame:
    """Eventos de §3.2 sobre la serie ``hs`` (E0 sin corregir, rejilla completa). ``bajada=None``: sin división."""
    comprobar_rejilla(hs.index, paso_horas)
    v = hs.to_numpy(dtype=float)
    r = rachas(v, u_id, iet_horas, paso_horas)
    sep = iet_horas / paso_horas
    filas = []
    for j, (a, b, toca) in enumerate(zip(r.ini, r.fin, r.toca_hueco, strict=True)):
        partes = [(int(a), int(b))] if bajada is None else dividir(v, int(a), int(b), sep, bajada, orden)
        for s, e in partes:
            seg = np.where(np.isnan(v[s : e + 1]), -np.inf, v[s : e + 1])
            filas.append(
                {
                    "id_racha": j,
                    "i_inicio": s,
                    "i_fin": e,
                    "i_pico": s + int(np.argmax(seg)),
                    "n_excedencias": int(np.sum(seg > u_id)),
                    "partes_racha": len(partes),
                    "toca_hueco": bool(toca),
                    "duracion_h": (e - s + 1) * paso_horas,
                }
            )
    if not filas:
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.DataFrame(filas)
    df.insert(0, "id_evento", np.arange(len(df)))
    df["dividido_por_bajada"] = df["partes_racha"] > 1
    idx = hs.index
    df["t_inicio"], df["t_fin"], df["t_pico"] = idx[df["i_inicio"]], idx[df["i_fin"]], idx[df["i_pico"]]
    return df[COLUMNAS]


def en_periodo(eventos: pd.DataFrame, inicio, fin) -> pd.DataFrame:
    """Eventos con el pico de E0 sin corregir en [inicio, fin]."""
    t = pd.DatetimeIndex(eventos["t_pico"])
    return eventos[(t >= pd.Timestamp(inicio)) & (t <= pd.Timestamp(fin))]
