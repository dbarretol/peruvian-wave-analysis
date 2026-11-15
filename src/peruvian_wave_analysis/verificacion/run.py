"""Etapa ``verificacion``: estudio por simulación de la biblioteca de extremos, con gate y manifiesto.

Bandas = nominal ± 3 errores estándar binomiales con el número de réplicas. Escenarios con banda: A1–A4 (cobertura
del RL100 de dos colas y por cola, y cobertura de ξ), B (RL100 con dos estratos, tres bandas), D1 y D2 (tamaño de la
prueba de Poisson con Anderson–Darling). En total, 21 bandas.

Regla de repetición: si falla una banda, el escenario se repite **una sola vez** con R = 1600 y semillas derivadas
distintas, y decide la repetición. Criterio de ``tope`` (A2–A4): límites abiertos por ``tope`` ≤ 1 %; si se supera,
el gate falla sin repetición. Un ajuste fallido cuenta como no cobertura de los dos lados.

Uso::

    uv run verificacion-f2 --procesos 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .. import estilo, rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from . import escenarios as esc

R = 400
R_REPETICION = 1600
FASE = "F02-biblioteca-comun"
TOPE_MAX = 0.01
N_BANDAS = 21  # contrato, §5.2: A1–A4 (3 + 1 cada uno), B (3), D1 y D2 (1 cada uno)
POTENCIA_MIN = 0.80
VARIABLES_HILOS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
ETAPA = orquestador.Etapa("verificacion", Path(__file__).resolve().parent, Path("."))


def banda(nominal: float, r: int) -> tuple[float, float]:
    ee = math.sqrt(nominal * (1 - nominal) / r)
    return max(0.0, nominal - 3 * ee), min(1.0, nominal + 3 * ee)


def _fraccion(serie: pd.Series) -> float:
    return float(serie.astype(bool).mean())


def _evaluar_banda(nombre: str, valor: float, nominal: float, r: int) -> dict:
    lo, hi = banda(nominal, r)
    return {
        "banda": nombre,
        "valor": valor,
        "nominal": nominal,
        "lo": lo,
        "hi": hi,
        "cumple": bool(lo - 1e-12 <= valor <= hi + 1e-12),
    }


def _no_falla(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool) & ~df["ajuste_fallido"].astype(bool)


def evaluar(e: esc.Escenario, df: pd.DataFrame) -> dict:
    """Bandas y métricas de un escenario a partir de sus réplicas."""
    r = len(df)
    out: dict = {
        "id": e.id,
        "descripcion": e.descripcion,
        "parametros": e.params,
        "replicas": r,
        "bandas": [],
        "metricas": {},
    }
    if e.tipo in ("rl", "rl_multi"):
        ab, ar = _no_falla(df, "rl_cubre_abajo"), _no_falla(df, "rl_cubre_arriba")
        out["metricas"] |= {
            "ajustes_fallidos": int(df["ajuste_fallido"].sum()),
            "cobertura_rl": _fraccion(ab & ar),
            "falla_rl_arriba": _fraccion(~ar),
            "falla_rl_abajo": _fraccion(~ab),
        }
        for lado in ("lo", "hi"):
            causas = df.get(f"rl_causa_{lado}", pd.Series(dtype=object)).dropna().value_counts().to_dict()
            out["metricas"][f"rl_abiertos_{lado}"] = {k: int(v) for k, v in sorted(causas.items())}
        tope = sum(int((df.get(f"rl_causa_{lado}") == esc.TOPE).sum()) for lado in ("lo", "hi"))
        out["metricas"]["fraccion_tope"] = tope / r
        out["bandas"] += [
            _evaluar_banda("cobertura_rl", out["metricas"]["cobertura_rl"], 0.95, r),
            _evaluar_banda("falla_rl_arriba", out["metricas"]["falla_rl_arriba"], 0.025, r),
            _evaluar_banda("falla_rl_abajo", out["metricas"]["falla_rl_abajo"], 0.025, r),
        ]
        if e.tipo == "rl":
            xab, xar = _no_falla(df, "xi_cubre_abajo"), _no_falla(df, "xi_cubre_arriba")
            out["metricas"] |= {
                "cobertura_xi": _fraccion(xab & xar),
                "falla_xi_arriba": _fraccion(~xar),
                "falla_xi_abajo": _fraccion(~xab),
                "xi_no_regular": _fraccion(df.get("xi_no_regular", pd.Series(False)).fillna(False)),
            }
            out["bandas"].append(_evaluar_banda("cobertura_xi", out["metricas"]["cobertura_xi"], 0.95, r))
        if e.tipo == "rl_multi":
            out["metricas"]["n_sin_resolver"] = int(df["n_sin_resolver"].fillna(0).sum())
    elif e.tipo.startswith("poisson"):
        out["metricas"] |= {
            "rechazo_ad": _fraccion(df["rechaza_ad"]),
            "rechazo_ks": _fraccion(df["rechaza_ks"]),
            "tormentas_media": float(df["n"].mean()),
        }
        if e.id != "E":
            out["bandas"].append(_evaluar_banda("tamano_ad", out["metricas"]["rechazo_ad"], 0.05, r))
    else:
        cols = [c for c in df.columns if c.endswith(("cubre_abajo", "cubre_arriba"))]
        for pref in sorted({c.rsplit("cubre_", 1)[0] for c in cols}):
            ab, ar = df[f"{pref}cubre_abajo"].astype(bool), df[f"{pref}cubre_arriba"].astype(bool)
            nombre = pref.rstrip("_") or "ic"
            out["metricas"] |= {
                f"cobertura_{nombre}": _fraccion(ab & ar),
                f"falla_{nombre}_arriba": _fraccion(~ar),
                f"falla_{nombre}_abajo": _fraccion(~ab),
            }
        if "autocorr_lag1" in df:
            out["metricas"]["autocorr_lag1_media"] = float(df["autocorr_lag1"].mean())
    if not e.con_banda:
        out["bandas"] = []  # informativos: se publican sus métricas, no entran en el gate
    return out


def correr_escenario(e: esc.Escenario, r: int, procesos: int, repeticion: bool = False) -> pd.DataFrame:
    args = [(e.id, i, repeticion) for i in range(r)]
    if procesos <= 1:
        filas = [esc.correr_replica(*a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=procesos) as ex:
            filas = list(ex.map(esc.correr_replica, *zip(*args, strict=True), chunksize=max(1, r // (4 * procesos))))
    return pd.DataFrame(filas).sort_values("replica").reset_index(drop=True)


def _escribir_csv(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, float_format="%.12g", lineterminator="\n")


def _figura(resultados: list[dict], ruta: Path) -> None:
    filas = [(r["id"], b) for r in resultados for b in r["bandas"]]
    fig, ax = estilo.nueva_figura(figsize=(8.0, 0.32 * len(filas) + 1.4))
    for k, (_, b) in enumerate(filas):
        y = len(filas) - 1 - k
        ax.plot([100 * b["lo"], 100 * b["hi"]], [y, y], color=estilo.TINTA_TENUE, lw=3, solid_capstyle="butt")
        ax.plot([100 * b["valor"]], [y], "o", color=estilo.TINTA if b["cumple"] else estilo.COLOR_BORDE_EXCLUSION, ms=4)
    ax.set_yticks(range(len(filas)), [f"{eid} · {b['banda']}" for eid, b in reversed(filas)], fontsize=7)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Valor medido y banda de aceptación (%)")
    ax.set_ylabel("Escenario y banda")
    ax.set_title("Estudio de verificación: bandas del gate (punto rojo = fuera)")
    estilo.asegurar_visible(ax, [100 * b["valor"] for _, b in filas], eje="x")
    fig.tight_layout()
    problemas = estilo.revisar_figura(fig, ejes_categoricos={"y"})
    if problemas:
        raise ValueError("Figura con problemas: " + "; ".join(problemas))
    estilo.guardar_figura(fig, ruta)


def ruta_canonica(raiz_datos: Path | None) -> Path:
    return rutas.directorio_datos(raiz_datos) / "processed" / FASE / "verificacion"


def comprobar_salida(raiz_datos: Path | None, canonica: bool, salida: Path | None) -> Path:
    """Carpeta de salida. Una corrida no canónica (``--solo`` o otro número de réplicas) exige ``--salida`` y no puede
    escribir en la carpeta canónica: si pasara la reemplazaría con una salida parcial, y si fallara la borraría. Se
    comprueba antes de abrir el staging, que borra la salida al fallar."""
    canon = ruta_canonica(raiz_datos)
    if salida is None:
        if not canonica:
            raise ValueError("Una corrida no canónica (--solo o --replicas distinto de 400) exige --salida")
        return canon
    salida = Path(salida)
    if not canonica and salida.resolve() == canon.resolve():
        raise ValueError(f"Una corrida no canónica no puede escribir en la carpeta canónica {canon}")
    return salida


def motivos_bandas(n_bandas: int, canonica: bool) -> list[str]:
    """La corrida canónica debe evaluar exactamente las 21 bandas del contrato."""
    if canonica and n_bandas != N_BANDAS:
        return [f"la corrida canónica evaluó {n_bandas} bandas; el contrato fija {N_BANDAS}"]
    return []


def ejecutar(
    raiz_datos: Path, procesos: int, r: int = R, solo: list[str] | None = None, salida: Path | None = None
) -> Path:
    # Canónica = todos los escenarios con R = 400 (con regla de repetición). La carpeta de salida no cuenta: la
    # comprobación de determinismo (G4) corre la misma etapa en otra carpeta y debe dar el mismo manifiesto.
    canonica = r == R and solo is None
    salida = comprobar_salida(raiz_datos, canonica, salida)
    lista = [e for e in esc.ESCENARIOS if solo is None or e.id in solo]
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        resultados, motivos = [], []
        for e in lista:
            df = correr_escenario(e, r, procesos)
            _escribir_csv(df, st / "replicas" / f"{e.id}.csv")
            ev = evaluar(e, df)
            if e.id in ("A2", "A3", "A4") and ev["metricas"]["fraccion_tope"] > TOPE_MAX:
                motivos.append(f"{e.id}: límites abiertos por tope {ev['metricas']['fraccion_tope']:.2%} > 1 %")
            if e.con_banda and not all(b["cumple"] for b in ev["bandas"]) and canonica:
                dfr = correr_escenario(e, R_REPETICION, procesos, repeticion=True)
                _escribir_csv(dfr, st / "replicas" / f"{e.id}-repeticion.csv")
                ev["repeticion"] = evaluar(e, dfr)
                if e.id in ("A2", "A3", "A4") and ev["repeticion"]["metricas"]["fraccion_tope"] > TOPE_MAX:
                    motivos.append(f"{e.id} (repetición): límites abiertos por tope > 1 %")
            decisivas = ev.get("repeticion", ev)["bandas"]
            motivos += [
                f"{e.id}{' (repetición)' if 'repeticion' in ev else ''}: {b['banda']} = {b['valor']:.4f} "
                f"fuera de [{b['lo']:.4f}, {b['hi']:.4f}]"
                for b in decisivas
                if not b["cumple"]
            ]
            resultados.append(ev)
            print(
                f"{e.id}: "
                + ", ".join(f"{b['banda']}={b['valor']:.3f}{'' if b['cumple'] else ' ✗'}" for b in ev["bandas"])
                + f" {ev['metricas']}",
                flush=True,
            )
        n_bandas = sum(len(r_["bandas"]) for r_ in resultados)
        motivos += motivos_bandas(n_bandas, canonica)
        potencia = next((r_["metricas"]["rechazo_ad"] for r_ in resultados if r_["id"] == "E"), None)
        resumen = {
            "replicas": r,
            "replicas_repeticion": R_REPETICION,
            "n_bandas": n_bandas,
            "potencia_E": potencia,
            "potencia_E_alcanza_80": None if potencia is None else bool(potencia >= POTENCIA_MIN),
            "semilla_raiz": semillas.SEMILLA_RAIZ,
            "escenarios": resultados,
        }
        json_estricto.escribir(st / "resumen.json", resumen)
        _figura(resultados, st / "bandas.png")
        gate = etapa.Gate(not motivos, {"n_bandas": n_bandas, "canonica": canonica, "tope_max": TOPE_MAX}, motivos)
        return etapa.publicar(st, salida, gate, raices, {}, rutas.directorio_datos(raiz_datos), semillas.SEMILLA_RAIZ)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Estudio de verificación de la biblioteca de extremos (F2).")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument(
        "--replicas", type=int, default=R, help="solo para pruebas: con otro valor no es la corrida canónica"
    )
    ap.add_argument("--solo", nargs="*", help="solo estos escenarios (no canónica)")
    ap.add_argument(
        "--salida", type=Path, help="otra carpeta de salida (por ejemplo, para la comprobación de determinismo)"
    )
    ap.add_argument("--directorio-datos", type=Path)
    a = ap.parse_args(argv)
    for v in VARIABLES_HILOS:  # un hilo de BLAS por proceso; los procesos hijos heredan el entorno
        os.environ[v] = "1"
    try:
        ruta = ejecutar(a.directorio_datos, a.procesos, a.replicas, a.solo, a.salida)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
