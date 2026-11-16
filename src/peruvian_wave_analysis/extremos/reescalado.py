"""Reescalado de tiempo de los inicios de tormenta y prueba de Poisson (Brown et al., 2002).

Con declustering por rachas, una tormenta nueva solo puede **empezar** más de IET horas después de la **última**
excedencia de la anterior. Por eso:

- el evento es el inicio de la tormenta;
- el proceso no está en riesgo durante (inicio, fin + IET] de cada tormenta ni en los huecos;
- ``intensidad_por_hora_en_riesgo`` del estrato = inicios / horas en riesgo del estrato. Es la magnitud de esta
  prueba y **no** se usa para niveles de retorno (allí la tasa va en eventos por año, ``tasa.py``);
- z_i = Σ intensidad·Δt sobre las celdas en riesgo entre inicios consecutivos ~ Exp(1) bajo un Poisson por
  estratos de inicios con duraciones independientes.

**Decide Anderson–Darling** contra Exp(1); KS se publica como informativo. Como la intensidad se estima con los
mismos eventos, los p-valores se calibran con un bootstrap paramétrico del procedimiento completo: inicios Poisson
por estrato en el tiempo en riesgo, duraciones remuestreadas del mismo estrato, misma rejilla y huecos, y
re-estimación en cada réplica.

- k = réplicas **válidas** con A² ≥ A² observado (los empates cuentan en contra);
- una réplica con menos de 5 intervalos es inválida; se generan réplicas hasta tener B válidas, y si hacen falta
  más de 1.05·B generaciones es un error;
- p = (1 + k)/(B + 1); rechazo ⇔ p ≤ 0.05 (con B = 99, k ≤ 4: tamaño 5 % para una nula simple).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MIN_INTERVALOS = 5
FACTOR_MAX_GENERACIONES = 1.05
B_POR_DEFECTO = 99
ALFA = 0.05


@dataclass(frozen=True)
class Reescalado:
    z: np.ndarray
    intensidad_por_hora_en_riesgo: dict[str, float]


def _posiciones(rejilla: pd.DatetimeIndex, tiempos) -> np.ndarray:
    pos = rejilla.get_indexer(pd.DatetimeIndex(tiempos))
    if np.any(pos < 0):
        raise ValueError("Hay instantes de tormenta fuera de la rejilla temporal")
    return pos


def _codigos(estratos) -> tuple[np.ndarray, tuple[str, ...]]:
    etiquetas = np.asarray(estratos).astype(str)
    nombres = tuple(dict.fromkeys(etiquetas.tolist()))
    idx = {n: i for i, n in enumerate(nombres)}
    return np.array([idx[e] for e in etiquetas], dtype=int), nombres


def _reescalar(
    validos: np.ndarray,
    codigo: np.ndarray,
    nombres,
    ini: np.ndarray,
    fin: np.ndarray,
    celdas_muertas: int,
    paso_horas: float,
) -> Reescalado:
    n = len(validos)
    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, np.minimum(ini + 1, n), 1)
    np.add.at(delta, np.minimum(fin + 1 + celdas_muertas, n), -1)
    en_riesgo = validos & (np.cumsum(delta[:-1]) == 0)
    k = len(nombres)
    horas = paso_horas * np.bincount(codigo[en_riesgo], minlength=k)
    cuentas = np.bincount(codigo[ini], minlength=k)
    inten = np.divide(cuentas, horas, out=np.full(k, np.nan), where=horas > 0)
    lam = np.nan_to_num(inten)[codigo]
    acum = np.cumsum(np.where(en_riesgo, lam * paso_horas, 0.0))
    return Reescalado(np.diff(acum[ini]), {s: float(inten[i]) for i, s in enumerate(nombres)})


def intervalos_reescalados(
    rejilla, validos, inicios, fines, estratos, iet_horas: float, paso_horas: float = 3.0
) -> Reescalado:
    """Intervalos reescalados entre inicios. ``inicios``/``fines`` = ``t_inicio``/``t_fin`` del declustering;
    ``estratos`` = etiqueta de cada celda de la rejilla."""
    rejilla = pd.DatetimeIndex(rejilla)
    validos = np.asarray(validos, dtype=bool)
    codigo, nombres = _codigos(estratos)
    if codigo.size != len(rejilla) or validos.size != len(rejilla):
        raise ValueError("validos y estratos deben tener una entrada por celda de la rejilla")
    ini, fin = _posiciones(rejilla, inicios), _posiciones(rejilla, fines)
    if ini.size != fin.size or np.any(fin < ini):
        raise ValueError("Cada tormenta necesita inicio ≤ fin")
    if np.any(np.diff(ini) <= 0):
        raise ValueError("Las tormentas deben estar ordenadas y sin repetir")
    if not validos[ini].all():
        raise ValueError("Hay inicios de tormenta en tiempo no válido")
    muertas = int(round(iet_horas / paso_horas))
    if np.any(ini[1:] <= fin[:-1] + muertas):
        raise ValueError(
            "Una tormenta empieza dentro de fin + IET de la anterior: el declustering no es coherente con el IET"
        )
    return _reescalar(validos, codigo, nombres, ini, fin, muertas, paso_horas)


def estadistico_ks(z) -> float:
    return float(stats.kstest(np.asarray(z, dtype=float), "expon").statistic)


def estadistico_ad(z) -> float:
    """A² de Anderson–Darling contra Exp(1) completamente especificada."""
    z = np.sort(np.asarray(z, dtype=float))
    n = z.size
    F = np.clip(-np.expm1(-z), 1e-300, 1.0 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(F) + np.log1p(-F[::-1]))))


def _reservas_de_duracion(codigo_ini: np.ndarray, duraciones: np.ndarray, k: int) -> list[np.ndarray]:
    return [duraciones[codigo_ini == j] if np.any(codigo_ini == j) else duraciones for j in range(k)]


def simular_nula(
    validos: np.ndarray,
    codigo: np.ndarray,
    intensidad: np.ndarray,
    duraciones: list[np.ndarray],
    celdas_muertas: int,
    paso_horas: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inicios Poisson por estrato en el tiempo en riesgo, con duración remuestreada del estrato del inicio."""
    lam = np.nan_to_num(intensidad)[codigo]
    candidatos = np.flatnonzero((rng.random(len(validos)) < lam * paso_horas) & validos)
    n = len(validos)
    ini, fin = [], []
    bloqueado = -1
    for i in candidatos:
        if i > bloqueado:
            reserva = duraciones[codigo[i]]
            e = min(i + int(reserva[rng.integers(reserva.size)]), n - 1)
            ini.append(i)
            fin.append(e)
            bloqueado = e + celdas_muertas
    return np.asarray(ini, dtype=int), np.asarray(fin, dtype=int)


@dataclass(frozen=True)
class PruebaPoisson:
    ad: float
    p_ad: float  # decide
    rechaza: bool  # p_ad ≤ 0.05
    ks: float
    p_ks: float  # informativo
    n_intervalos: int
    B: int
    generadas: int


def prueba_poisson(
    rejilla,
    validos,
    inicios,
    fines,
    estratos,
    iet_horas: float,
    paso_horas: float,
    rng: np.random.Generator,
    B: int = B_POR_DEFECTO,
) -> PruebaPoisson:
    rejilla = pd.DatetimeIndex(rejilla)
    validos = np.asarray(validos, dtype=bool)
    res = intervalos_reescalados(rejilla, validos, inicios, fines, estratos, iet_horas, paso_horas)
    if res.z.size < MIN_INTERVALOS:
        raise ValueError(f"Hacen falta al menos {MIN_INTERVALOS} intervalos; hay {res.z.size}")
    codigo, nombres = _codigos(estratos)
    ini, fin = _posiciones(rejilla, inicios), _posiciones(rejilla, fines)
    reservas = _reservas_de_duracion(codigo[ini], fin - ini, len(nombres))
    inten = np.array([res.intensidad_por_hora_en_riesgo[s] for s in nombres])
    muertas = int(round(iet_horas / paso_horas))
    ks0, ad0 = estadistico_ks(res.z), estadistico_ad(res.z)
    k_ad = k_ks = validas = generadas = 0
    tope = int(np.floor(FACTOR_MAX_GENERACIONES * B))
    while validas < B:
        if generadas >= tope:
            raise ValueError(f"Hicieron falta más de {tope} réplicas para tener {B} válidas: el p-valor no es fiable")
        generadas += 1
        sb, eb = simular_nula(validos, codigo, inten, reservas, muertas, paso_horas, rng)
        if sb.size < MIN_INTERVALOS + 1:
            continue
        zb = _reescalar(validos, codigo, nombres, sb, eb, muertas, paso_horas).z
        validas += 1
        k_ad += estadistico_ad(zb) >= ad0
        k_ks += estadistico_ks(zb) >= ks0
    p_ad = (1 + k_ad) / (B + 1)
    return PruebaPoisson(
        ad0, p_ad, bool(p_ad <= ALFA + 1e-12), ks0, (1 + k_ks) / (B + 1), int(res.z.size), B, generadas
    )
