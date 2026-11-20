"""Insumos y ramas de la F6 (contrato §1 y §0, «Ramas»).

- **G1:** cada salida de la que se lee está publicada y sus archivos coinciden con su manifiesto
  (``etapa.exigir_publicada``); los insumos se registran con su hash.
- **Ramas:** E0 y E1 (recorte desde el inicio que publica la F4 para E1), cada una sin corregir y corregida por
  cuantiles (F5, ``serie_corregida.parquet``): E0 corregida = ``VHM0_cuantiles_toda`` (la corrección se aplica a
  toda la serie, contrato de la F5 §6.2, línea de E0); E1 corregida = ``VHM0_cuantiles_asimilado``. Solo cambia Hs;
  dirección, períodos y particiones son los de E0.
- **Dudosos de la F3:** los registros de ``candidatos.csv`` (detector), que la F3 mantuvo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..comun import etapa, json_estricto
from ..datos import carga
from ..datos.run import ruta_candidatos, ruta_serie

NOMBRE = "marcona-200m"
RAMAS = ("E0_sin_corregir", "E0_cuantiles", "E1_sin_corregir", "E1_cuantiles")
RAMA_DECISION = "E0_sin_corregir"
COLUMNA_CORREGIDA = {"E0_cuantiles": "VHM0_cuantiles_toda", "E1_cuantiles": "VHM0_cuantiles_asimilado"}


def ruta_f04(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F04-homogeneidad" / "homogeneidad" / nombre


def ruta_f05(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F05-validacion-externa" / "validacion" / nombre


def rutas_insumos(raiz: Path, nombre: str = NOMBRE) -> dict[str, Path]:
    """Insumos de §1, tras exigir que cada salida esté publicada y sin cambios (G1)."""
    ds, dc = ruta_serie(raiz, nombre), ruta_candidatos(raiz, nombre)
    d4, d5 = ruta_f04(raiz, nombre), ruta_f05(raiz, nombre)
    for d in (ds, dc, d4, d5):
        etapa.exigir_publicada(d)
    return {
        "serie_nodo": ds / "serie_nodo.parquet",
        "serie_corregida": d5 / "serie_corregida.parquet",
        "resumen_f05": d5 / "resumen_f05.json",
        "correccion": d5 / "correccion.json",
        "resumen_f04": d4 / "resumen_f04.json",
        "candidatos": dc / "candidatos.csv",
    }


@dataclass
class Datos:
    serie: pd.DataFrame  # E0 sin corregir, rejilla completa, NaN en los no válidos
    ramas: dict[str, pd.Series]  # Hs de cada rama, en su rejilla completa
    inicio_e1: pd.Timestamp
    dudosos: pd.DatetimeIndex
    escenarios_f05: dict
    fechas: dict  # C1 y C2 de la F4


def inicio_e1(resumen_f04: dict) -> pd.Timestamp:
    e1 = next(e for e in resumen_f04["escenarios_hs"] if e["nombre"] == "E1")
    return pd.Timestamp(e1["periodo"][0])


def armar_ramas(serie: pd.DataFrame, corregida: pd.DataFrame, ini_e1: pd.Timestamp) -> dict[str, pd.Series]:
    hs = serie["VHM0"].astype(float)
    if not corregida.index.equals(serie.index):
        raise ValueError("La serie corregida no está en la misma rejilla que la serie del nodo")
    ramas = {
        "E0_sin_corregir": hs,
        "E0_cuantiles": corregida[COLUMNA_CORREGIDA["E0_cuantiles"]].astype(float).where(hs.notna()),
        "E1_sin_corregir": hs[hs.index >= ini_e1],
        "E1_cuantiles": corregida[COLUMNA_CORREGIDA["E1_cuantiles"]]
        .astype(float)
        .where(hs.notna())[hs.index >= ini_e1],
    }
    for k, v in ramas.items():
        v.name = k
    return ramas


def cargar(raiz: Path, rutas: dict[str, Path], nombre: str = NOMBRE) -> Datos:
    serie = carga.cargar_serie(raiz, nombre)
    if "tiempo" in serie.columns:
        serie = serie.set_index("tiempo")
    corregida = pd.read_parquet(rutas["serie_corregida"])
    if "tiempo" in corregida.columns:
        corregida = corregida.set_index("tiempo")
    r4 = json_estricto.leer(rutas["resumen_f04"])
    r5 = json_estricto.leer(rutas["resumen_f05"])
    ini = inicio_e1(r4)
    cand = pd.read_csv(rutas["candidatos"], parse_dates=["tiempo"])
    e2 = next(e for e in r4["escenarios_hs"] if e["nombre"] == "E2")
    return Datos(
        serie,
        armar_ramas(serie, corregida, ini),
        ini,
        pd.DatetimeIndex(cand["tiempo"]),
        r5["escenarios"],
        {"C1": str(ini.date()), "C2": e2["fecha_escalon"]},
    )


def marcar_dudosos(picos: pd.DataFrame, dudosos: pd.DatetimeIndex) -> np.ndarray:
    """True si algún registro dudoso de la F3 cae dentro de la tormenta [t_inicio, t_fin]."""
    if picos.empty or dudosos.empty:
        return np.zeros(len(picos), bool)
    d = np.sort(dudosos.to_numpy())
    ini, fin = picos["t_inicio"].to_numpy(), picos["t_fin"].to_numpy()
    return np.searchsorted(d, fin, side="right") > np.searchsorted(d, ini, side="left")
