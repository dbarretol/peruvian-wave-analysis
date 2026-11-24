"""Sesgo de la regla R1b y referencia de la sensibilidad, ``eventos-sesgo-regla`` (ADR-0037; contrato §3.4 y §5.4).

Lee la simulación previa ya publicada (``simulacion-previa/simulacion/replicas.csv``) y no simula nada. Da la evidencia
de ADR-0037 de forma reproducible (ADR-0019), en lugar de los scripts de la sesión:

1. **Sesgo por celda y regla** (punto 2): media del error del log RL100, su error típico, y k, u\\* y excedencias
   medias de cada regla.
2. **Selección** (punto 3): en cada celda de A, el error de R1b y el de k = 10 en las mismas réplicas, por tramos del k
   que elige R1b; la correlación entre ese k y el error de k = 10; y R1b − k10 con k < 10 y con k > 10.
3. **Referencia de la sensibilidad** (punto 6), que la etapa con datos usa:
   - d_RL = RL100(R) / RL100(R1b) − 1 y Δu = u\\*(R) − u\\*(R1b), réplica a réplica, para R = R2 y R3;
   - en las 6 celdas de A con σ = 0.35, juntas (los dos N0 y las tres formas);
   - se excluyen las réplicas con sustituto de R1b o de la regla comparada (un «sin resultado» es «no comparable»);
   - intervalo central con signo [P2.5, P97.5]; el de cada celda es descriptivo.

``ubicar`` clasifica una diferencia del sitio frente al intervalo: «dentro», «bajo», «sobre» o «no comparable».

En todo el módulo, «%» es 100 × sesgo medio del log RL100.

Salida en ``processed/F07-eventos/simulacion-previa/sesgo-regla/``: ``sesgo_por_celda.csv``, ``seleccion.csv``,
``diferencias.csv`` y ``referencia_sensibilidad.json``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..comun import etapa, json_estricto, orquestador
from ..comun.errores import GateFallido

ETAPA = orquestador.Etapa("eventos-sesgo-regla", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
REGLA = "R1b"
COMPARADAS = ("R2", "R3")
REFERENCIAS = ("k1", "k10")
CASO_REFERENCIA = "A"
SIGMA = 0.35
CUANTILES = (0.025, 0.975)
TRAMOS_K = (0, 4, 8, 12, 16, 20)
K_FIJO = 10
COLUMNAS = ["celda", "caso", "forma", "N0", "sigma", "replica", "regla", "k_usado", "u_usado", "n_usado", "rl100",
            "error_log", "sustituto"]  # fmt: skip
NO_COMPARABLE = "no comparable"


def ruta_simulacion(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "simulacion-previa" / "simulacion"


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "simulacion-previa" / "sesgo-regla"


def cargar(ruta_replicas: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta_replicas, usecols=COLUMNAS)
    df["con_sustituto"] = df["sustituto"].notna()
    return df


def _sigma(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    return df[np.isclose(df["sigma"].astype(float), sigma)]


def sesgo_por_celda(df: pd.DataFrame, sigma: float = SIGMA) -> pd.DataFrame:
    """Sesgo medio del log RL100, su error típico, y k, u\\* y excedencias medias, por celda y regla."""
    d = _sigma(df, sigma)
    g = d.groupby(["caso", "celda", "regla"], sort=True)
    out = g.agg(
        replicas=("error_log", "size"),
        sesgo=("error_log", "mean"),
        desviacion=("error_log", "std"),
        k_medio=("k_usado", "mean"),
        u_medio_m=("u_usado", "mean"),
        excedencias_medias=("n_usado", "mean"),
        fraccion_sustituto=("con_sustituto", "mean"),
    ).reset_index()
    out["error_tipico"] = out["desviacion"] / np.sqrt(out["replicas"])
    return out


def _ancho(df: pd.DataFrame, celda: str, valores: str) -> pd.DataFrame:
    d = df[df["celda"] == celda]
    return d.pivot_table(index="replica", columns="regla", values=valores, aggfunc="first")


def _correlacion(x: pd.Series, y: pd.Series) -> float:
    """Correlación de Pearson; NaN si alguna de las dos es constante."""
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def seleccion(df: pd.DataFrame, sigma: float = SIGMA, caso: str = CASO_REFERENCIA) -> pd.DataFrame:
    """Por celda y tramo del k de R1b: réplicas, error medio de R1b y de k = 10 en esas mismas réplicas.

    Añade, por celda, la correlación entre el k de R1b y el error de k = 10, y R1b − k10 con k < 10 y con k > 10.
    """
    d = _sigma(df, sigma)
    d = d[d["caso"] == caso]
    filas = []
    for celda in sorted(d["celda"].unique()):
        err = _ancho(d, celda, "error_log")
        k = _ancho(d, celda, "k_usado")[REGLA]
        r1b, k10 = err[REGLA], err[f"k{K_FIJO}"]
        diff = r1b - k10
        corr = _correlacion(k, k10)
        tramo = pd.cut(k, list(TRAMOS_K))
        for t, idx in k.groupby(tramo, observed=True).groups.items():
            filas.append({"celda": celda, "tramo_k": f"({t.left}, {t.right}]", "replicas": len(idx),
                          "error_r1b": float(r1b[idx].mean()), "error_k10": float(k10[idx].mean()),
                          "correlacion_k_r1b_error_k10": corr,
                          "r1b_menos_k10_k_bajo_10": float(diff[k < K_FIJO].mean()),
                          "replicas_k_bajo_10": int((k < K_FIJO).sum()),
                          "r1b_menos_k10_k_sobre_10": float(diff[k > K_FIJO].mean()),
                          "replicas_k_sobre_10": int((k > K_FIJO).sum())})  # fmt: skip
    return pd.DataFrame(filas)


def diferencias(df: pd.DataFrame, comparada: str, sigma: float = SIGMA, caso: str = CASO_REFERENCIA) -> pd.DataFrame:
    """d_RL y Δu de ``comparada`` frente a R1b, réplica a réplica, con la marca de exclusión por sustituto."""
    d = _sigma(df, sigma)
    d = d[(d["caso"] == caso) & d["regla"].isin([REGLA, comparada])]
    w = d.pivot_table(index=["celda", "N0", "forma", "replica"], columns="regla",
                      values=["rl100", "u_usado", "con_sustituto"], aggfunc="first")  # fmt: skip
    sust_r1b = w["con_sustituto"][REGLA].astype(bool)
    sust_otra = w["con_sustituto"][comparada].astype(bool)
    out = pd.DataFrame({
        "d_rl": w["rl100"][comparada] / w["rl100"][REGLA] - 1.0,
        "delta_u_m": w["u_usado"][comparada] - w["u_usado"][REGLA],
        "sustituto_r1b": sust_r1b,
        "sustituto_comparada": sust_otra,
    }).reset_index()  # fmt: skip
    out["comparable"] = ~(out["sustituto_r1b"] | out["sustituto_comparada"])
    out.insert(0, "comparada", comparada)
    return out


def _intervalo(x: pd.Series) -> list[float]:
    return [float(np.quantile(x, q)) for q in CUANTILES]


def referencia(dif: pd.DataFrame) -> dict:
    """Intervalo central [P2.5, P97.5] de d_RL y Δu con las celdas juntas, más el de cada celda (descriptivo)."""
    ok = dif[dif["comparable"]]
    por_celda = {
        c: {"d_rl": _intervalo(g["d_rl"]), "delta_u_m": _intervalo(g["delta_u_m"]), "replicas": int(len(g))}
        for c, g in ok.groupby("celda", sort=True)
    }
    return {
        "cuantiles": list(CUANTILES),
        "d_rl": _intervalo(ok["d_rl"]),
        "delta_u_m": _intervalo(ok["delta_u_m"]),
        "media_d_rl": float(ok["d_rl"].mean()),
        "media_delta_u_m": float(ok["delta_u_m"].mean()),
        "replicas_comparables": int(len(ok)),
        "fraccion_excluida_solo_por_la_comparada": float((dif["sustituto_comparada"] & ~dif["sustituto_r1b"]).mean()),
        "fraccion_excluida_por_r1b": float(dif["sustituto_r1b"].mean()),
        "por_celda_descriptivo": por_celda,
    }


def ubicar(valor: float | None, intervalo: Sequence[float]) -> str:
    """«dentro», «bajo» o «sobre» el intervalo; «no comparable» si falta el valor (sin resultado de una regla)."""
    if valor is None or not math.isfinite(valor):
        return NO_COMPARABLE
    lo, hi = intervalo
    if valor < lo:
        return "bajo"
    if valor > hi:
        return "sobre"
    return "dentro"


def calcular(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ses = sesgo_por_celda(df)
    sel = seleccion(df)
    difs = [diferencias(df, r) for r in COMPARADAS]
    ref = {
        "regla": REGLA,
        "caso": CASO_REFERENCIA,
        "sigma": SIGMA,
        "celdas": sorted(_sigma(df, SIGMA).loc[lambda d: d["caso"] == CASO_REFERENCIA, "celda"].unique().tolist()),
        "nota": "Celdas de A con σ = 0.35 juntas (dos N0, tres formas). Δu en m depende de σ. "
                "Se excluyen las réplicas con sustituto de R1b o de la comparada. ADR-0037, punto 6.",
        "comparadas": {r: referencia(d) for r, d in zip(COMPARADAS, difs, strict=True)},
    }  # fmt: skip
    return ses, sel, pd.concat(difs, ignore_index=True), ref


def ejecutar(raiz_datos=None) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    sim = ruta_simulacion(raiz)
    etapa.exigir_publicada(sim)
    ins = {"replicas_simulacion": sim / "replicas.csv", "manifiesto_simulacion": sim / "MANIFEST.json"}
    etapa.exigir_insumos(ins)
    df = cargar(ins["replicas_simulacion"])
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        ses, sel, dif, ref = calcular(df)
        opciones = {"index": False, "float_format": "%.8g", "lineterminator": "\n"}
        ses.to_csv(st / "sesgo_por_celda.csv", **opciones)
        sel.to_csv(st / "seleccion.csv", **opciones)
        dif.to_csv(st / "diferencias.csv", **opciones)
        json_estricto.escribir(st / "referencia_sensibilidad.json", ref)
        motivos = []
        if len(ref["celdas"]) != 6:
            motivos.append(f"Se esperaban 6 celdas de A y hay {len(ref['celdas'])}")
        for r, v in ref["comparadas"].items():
            if v["replicas_comparables"] == 0:
                motivos.append(f"Sin réplicas comparables para {r}")
        gate = etapa.Gate(not motivos, {"celdas": len(ref["celdas"])}, motivos)
        return etapa.publicar(st, salida, gate, raices, ins, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-sesgo-regla", description="Sesgo de R1b y referencia de sensibilidad")
    ap.add_argument("--datos")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = ejecutar(a.datos)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
