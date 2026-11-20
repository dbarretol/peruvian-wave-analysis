"""Simulación previa de la regla de umbral, ``eventos-simulacion-previa`` (contrato de la F7, §5.1, §5.2 y §5.4).

**Fase 1:** las 18 celdas de la métrica y las 4 del control (§5.1), 500 réplicas cada una, con σ = 0.35 m. En cada
réplica, sobre **los mismos picos**:
- rejilla y barrido de §3.4 (``umbral.rejilla``, ``umbral.barrido``, con ``ajustar_gpd``);
- R1a, R1b, R2 y R3 (R3 con B = 99 y los reajustes de ADR-0034);
- las referencias: k = 1, k = 10 (si su ajuste falla, k = 11 y luego k = 9; C2-07) y, en B+, B− y el control, el
  oráculo: el primer u_k de la rejilla por encima de u0 (ADR-0033);
- una regla «sin resultado» entra con el RL100 de k = 10 (C-04), marcado en ``sustituto``;
- la pendiente de ξ̂ frente a log n_k en los u_k ≥ u* (con al menos 3 umbrales con ξ̂), para el nulo de §5.2.

**Elección (§5.4):** ``eleccion.elegir`` sobre la fase 1.

**Fase 2, sensibilidad de σ (C2-06):** la regla elegida (y las referencias) con σ = 0.26 y 0.5 m, en las 18 celdas y
en las 4 del control, para ver si sigue cumpliendo las restricciones y el filtro (§5.1). No cambia la elección.

Reanudable (trozos de 10 réplicas, cada fase con su avance). ``--previo`` cronometra una réplica por celda y estima
la corrida sin publicar. Semillas de ``comun/semillas`` por nombre: réplica (celda, σ, i) y bootstrap de R3
(celda, σ, i, posición del u_k); el resultado no depende del número de procesos.

Salida: ``processed/F07-eventos/simulacion-previa/simulacion/`` (hermana del conteo previo), con ``replicas.csv``,
``resumen_celdas.csv`` y ``eleccion.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..validacion import reanudable
from . import casos as CS
from . import eleccion as EL
from . import r2
from . import salidas as SA
from . import umbral as U

ETAPA = orquestador.Etapa("eventos-simulacion-previa", Path(__file__).resolve(), Path("."))
REPLICAS = 500
TROZO = 10
SIGMAS_SENSIBILIDAD = (0.26, 0.5)
REGLAS = EL.REGLAS
K_REFERENCIA = 10
K_RESPALDO = (10, 11, 9)
MIN_PUNTOS_PENDIENTE = 3


def nombre_celda(c: CS.Celda) -> str:
    return c.nombre if c.sigma == CS.SIGMA else f"{c.nombre}_s{c.sigma:g}"


def celdas_de(conteo: dict, sigma: float = CS.SIGMA) -> list[CS.Celda]:
    n0 = conteo["n0_simulacion"]
    args = (tuple(int(x) for x in n0["n0"]), tuple(float(x) for x in n0["anios_simulados"]), float(conteo["u_id_m"]))
    cs = CS.celdas(*args) + CS.celdas(*args, control=True)
    return [CS.Celda(c.caso, c.forma, c.N0, c.anios, c.u_id, sigma) for c in cs]


# --- Posiciones de referencia --------------------------------------------------------------------


def posicion_de_k(rej: U.Rejilla, k: int) -> int:
    """Posición en la rejilla sin repetidos del u de k: si k se quitó por repetido, la del k conservado con su u."""
    return int(np.flatnonzero(rej.k <= k)[-1])


def referencia_k10(rej: U.Rejilla, rl100: np.ndarray) -> tuple[int | None, str]:
    """k = 10; si su RL100 no existe, k = 11 y luego k = 9 (C2-07). Devuelve (posición, etiqueta)."""
    for k in K_RESPALDO:
        i = posicion_de_k(rej, k)
        if np.isfinite(rl100[i]):
            return i, f"k{k}"
    return None, "ninguno"


def indice_oraculo(rej: U.Rejilla, u0: float | None) -> int | None:
    """El primer u_k de la rejilla por encima de u0 (ADR-0033)."""
    if u0 is None:
        return None
    i = np.flatnonzero(rej.u > u0)
    return int(i[0]) if i.size else None


def pendiente_xi(xi: np.ndarray, n: np.ndarray, desde: int) -> float:
    """Pendiente MCO de ξ̂ frente a log n_k en las posiciones ≥ desde con ξ̂ finito (≥ 3 puntos; si no, NaN)."""
    x = np.log(np.asarray(n, dtype=float)[desde:])
    y = np.asarray(xi, dtype=float)[desde:]
    ok = np.isfinite(y)
    if ok.sum() < MIN_PUNTOS_PENDIENTE:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


# --- Una réplica ---------------------------------------------------------------------------------


def aplicar_reglas(
    picos: np.ndarray, rej: U.Rejilla, aj: list[U.AjusteUmbral], reglas: Sequence[str], semilla_r3: tuple[str, ...]
) -> dict[str, U.ResultadoRegla]:
    rl = np.array([a.rl_valor(100.0) for a in aj])
    out: dict[str, U.ResultadoRegla] = {}
    for v in ("R1a", "R1b"):
        if v in reglas:
            out[v] = U.regla_r1(rl, v)
    if "R2" in reglas:
        pr = r2.pruebas_r2(picos, r2.umbrales_r2(rej.u), [a.ajuste for a in aj][0::2])
        out["R2"] = r2.regla_r2(pr)
    if "R3" in reglas:
        p = []
        for k, (u, a) in enumerate(zip(rej.u, aj, strict=True)):
            if not a.valido:
                p.append(np.nan)
                continue
            exc = picos[picos > u] - u
            p.append(U.p_valor_r3(exc, a.ajuste, float(u), semillas.generador(*semilla_r3, str(k))).p)
        out["R3"] = U.regla_r3(p)
    return out


def evaluar_replica(celda: CS.Celda, i: int, reglas: Sequence[str] = REGLAS) -> list[dict]:
    nombre = nombre_celda(celda)
    rep = CS.replica(celda, semillas.generador("f07", "simulacion-previa", nombre, str(i)))
    picos = rep.picos
    rej = U.rejilla(picos)
    aj = U.barrido(picos, rej, celda.anios)
    rl = np.array([a.rl_valor(100.0) for a in aj])
    xi = np.array([a.ajuste.xi if a.valido else np.nan for a in aj])
    log_v = np.log(rep.rl100_verdadero)
    res = aplicar_reglas(picos, rej, aj, reglas, ("f07", "simulacion-previa", "r3", nombre, str(i)))
    i10, et10 = referencia_k10(rej, rl)
    i_or = indice_oraculo(rej, rep.u0)
    comun = {
        "celda": nombre,
        "caso": celda.caso,
        "forma": celda.forma,
        "N0": celda.N0,
        "anios": celda.anios,
        "sigma": celda.sigma,
        "replica": i,
        "n_picos": int(picos.size),
        "k0": rep.k0 if rep.k0 is not None else np.nan,
        "u0": rep.u0 if rep.u0 is not None else np.nan,
        "rl100_verdadero": rep.rl100_verdadero,
        "con_repetidos": rej.con_repetidos,
        "indice_oraculo": i_or if i_or is not None else -1,
    }

    def fila(regla, estado, indice, usado, sustituto):
        ok = usado is not None and np.isfinite(rl[usado])
        return comun | {
            "regla": regla,
            "estado": estado,
            "indice": indice if indice is not None else -1,
            "indice_usado": usado if usado is not None else -1,
            "k_usado": int(rej.k[usado]) if usado is not None else -1,
            "u_usado": float(rej.u[usado]) if usado is not None else np.nan,
            "n_usado": int(rej.n_real[usado]) if usado is not None else -1,
            "sustituto": sustituto,
            "rl100": float(rl[usado]) if ok else np.nan,
            "error_log": float(np.log(rl[usado]) - log_v) if ok else np.nan,
            "pendiente_xi": pendiente_xi(xi, rej.n_real, usado) if usado is not None else np.nan,
        }

    filas = []
    for regla, r in res.items():
        if r.indice is not None and np.isfinite(rl[r.indice]):
            filas.append(fila(regla, r.estado, r.indice, r.indice, ""))
        else:  # «sin resultado» (o RL100 inexistente en u*): entra con k = 10 (C-04)
            filas.append(fila(regla, r.estado, r.indice, i10, et10))
    filas.append(fila("k1", U.CON_RESULTADO, 0, 0, ""))
    filas.append(fila("k10", U.CON_RESULTADO, i10, i10, "" if et10 == "k10" else et10))
    if i_or is not None:
        filas.append(fila("oraculo", U.CON_RESULTADO, i_or, i_or, ""))
    return filas


def _trabajo(args) -> list[dict]:
    celda, indices, reglas = args
    return [f for i in indices for f in evaluar_replica(celda, i, reglas)]


def tareas(cs: Sequence[CS.Celda], replicas: int, reglas: Sequence[str]) -> list[tuple[str, tuple]]:
    return [
        (f"{nombre_celda(c)}__{a:04d}", (c, list(range(a, min(a + TROZO, replicas))), tuple(reglas)))
        for c in cs
        for a in range(0, replicas, TROZO)
    ]


# --- Etapa ---------------------------------------------------------------------------------------


def ruta_avance(raiz: Path) -> Path:
    return SA.ruta_avance_simulacion(raiz)


def _correr(cs, replicas, reglas, avance: Path, procesos: int, rotulo: str) -> pd.DataFrame:
    aviso = lambda t: print(f"  {rotulo}: {t}", flush=True)  # noqa: E731
    partes = reanudable.ejecutar(_trabajo, tareas(cs, replicas, reglas), avance, procesos, aviso)
    return pd.DataFrame([f for p in partes for f in p])


def sensibilidad(df: pd.DataFrame, regla: str) -> dict:
    res = EL.resumen_celdas(df)
    return {"sigma": float(df["sigma"].iloc[0]), "metrica": EL.metrica(res, regla)} | EL.restricciones(res, regla)


def _previo(cs: list[CS.Celda], replicas: int, procesos: int) -> None:
    t = {}
    for c in cs:
        t0 = time.time()
        evaluar_replica(c, 0)
        t[nombre_celda(c)] = time.time() - t0
        print(f"  {nombre_celda(c)}: {t[nombre_celda(c)]:.1f} s", flush=True)
    seg = float(np.mean(list(t.values())))
    p = max(procesos, 1)
    f1 = seg * replicas * len(cs) / p / 3600
    f2 = seg * replicas * len(cs) * len(SIGMAS_SENSIBILIDAD) / p / 3600  # cota: si gana R3, la regla cara
    print(
        f"previo: {seg:.1f} s por réplica; fase 1 ≈ {f1:.1f} h y fase 2 ≤ {f2:.1f} h con {p} procesos", flush=True
    )


def ejecutar(raiz_datos=None, procesos_: int = 4, replicas: int = REPLICAS, previo: bool = False) -> Path | None:
    raiz = rutas.directorio_datos(raiz_datos)
    d_conteo = SA.ruta_conteo_previo(raiz)
    etapa.exigir_publicada(d_conteo)
    ins = {"conteo_previo": d_conteo / "conteo_previo.json"}
    conteo = json_estricto.leer(ins["conteo_previo"])
    cs = celdas_de(conteo)
    if previo:
        _previo(cs, replicas, procesos_)
        return None
    salida = SA.ruta_simulacion(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA), etapa.exigir_insumos(ins), replicas, TROZO)
    avance = ruta_avance(raiz) / clave
    print(f"Avance reanudable en {avance}", flush=True)
    with etapa.staging_protegido(salida, raices) as st:
        df = _correr(cs, replicas, REGLAS, avance / "fase1", procesos_, "fase 1")
        res = EL.resumen_celdas(df)
        el = EL.elegir(df, res)
        regla = el["elegida"]
        print(f"Regla elegida: {regla} (empatadas: {el['empatadas']})", flush=True)
        sens, dfs = [], [df]
        if regla is not None:
            for s in SIGMAS_SENSIBILIDAD:
                d = _correr(celdas_de(conteo, s), replicas, (regla,), avance / f"fase2_{regla}_s{s:g}", procesos_,
                            f"sensibilidad σ = {s:g}")  # fmt: skip
                sens.append(sensibilidad(d, regla))
                dfs.append(d)
        todo = pd.concat(dfs, ignore_index=True)
        todo.to_csv(st / "replicas.csv", index=False, float_format="%.8g", lineterminator="\n")
        res_todo = EL.resumen_celdas(todo)
        res_todo.to_csv(st / "resumen_celdas.csv", index=False, float_format="%.8g", lineterminator="\n")
        salida_json = {
            "eleccion": el,
            "nulo_pendiente_xi": EL.nulo_pendiente(df, regla) if regla else None,
            "para_la_f9": EL.para_la_f9(res, regla) if regla else None,
            "sensibilidad_sigma": sens,
            "metricas_r1": {v: el["por_regla"][v]["metrica"] for v in ("R1a", "R1b") if v in el["por_regla"]},
            "replicas_por_celda": replicas,
            "celdas": [nombre_celda(c) for c in cs],
        }
        json_estricto.escribir(st / "eleccion.json", salida_json)
        esperadas = replicas * len(cs)
        motivos = []
        if df["replica"].groupby(df["celda"]).nunique().sum() != esperadas:
            motivos.append("Faltan réplicas de la fase 1")
        criterios = {"replicas_por_celda": replicas, "celdas": len(cs), "regla_elegida": regla}
        gate = etapa.Gate(not motivos, criterios, motivos)
        publicada = etapa.publicar(st, salida, gate, raices, ins, raiz, semillas.semilla("f07", "simulacion-previa"))
        reanudable.limpiar(avance)
        if regla is None:
            print(f"AVISO: {el['motivo']}", flush=True)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-simulacion-previa", description="Simulación previa de la regla de umbral")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=REPLICAS)
    ap.add_argument("--previo", action="store_true", help="Cronometra una réplica por celda; no publica.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = ejecutar(a.datos, a.procesos, a.replicas, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    if salida:
        print(f"Publicado en {salida}")
    return 0
