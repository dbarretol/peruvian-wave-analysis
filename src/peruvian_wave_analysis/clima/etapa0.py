"""Etapa 0 sintética de la F6, ``clima-etapa0`` (contrato §3.5, regla de concordancia, y §8; v5, S-17). Sin datos.

Calibra el **factor de la regla de concordancia** antes de leer datos: el P95 de la razón máx/mín de las tres escalas
de la cola sobre toda la rejilla de procesos sintéticos, con estimación dentro del invierno (mayo–octubre):

- procesos: AR(1) de la anomalía con τ_e ∈ {24, 48, 72} h y el proceso de la ec. 6 de Ferro y Segers con
  θ ∈ {0.25, 0.5, 0.75}; cada uno sin ciclo anual y con el ciclo calibrado al 86 % de excedencias en invierno;
- largo de la serie del nodo, u = P99 de cada réplica, 200 réplicas por combinación, semillas derivadas del nombre.

Escalas de la cola de cada réplica, en horas: T_(C) a P99 (con piso de un paso: T_(C) = 0 significa que cada
excedencia es una tormenta), el corte natural de los huecos (si existe) y la escala del extremograma con factor 1.5.
La razón se calcula con las escalas disponibles (al menos dos). Si el factor supera 3, la regla no discrimina y se
dice así (no es un fallo del gate).

Reanudable: cada trozo de réplicas se guarda al terminar. ``--previo`` cronometra unas réplicas sin publicar.
Salida en ``processed/F06-clima-de-oleaje/etapa0-sintetica/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..extremos import dependencia as D
from ..validacion import reanudable
from . import sinteticos as S

ETAPA0 = orquestador.Etapa("clima-etapa0", Path(__file__).resolve(), Path("."))
PROCESOS = (("ar1", 24.0), ("ar1", 48.0), ("ar1", 72.0), ("fs", 0.25), ("fs", 0.5), ("fs", 0.75))
CICLOS = (False, True)
REPLICAS = 200
TROZO = 25
FACTOR_EXTREMOGRAMA = 1.5
SENSIBILIDADES_FACTOR = (1.25, 2.0)
UMBRAL_NO_DISCRIMINA = 3.0
CUANTIL_FACTOR = 0.95


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / "F06-clima-de-oleaje" / "etapa0-sintetica"


def ruta_avance(raiz: Path) -> Path:
    return Path(raiz) / "interim" / "F06-clima-de-oleaje" / "etapa0-avance"


def nombre_config(proceso: str, parametro: float, ciclo: bool) -> str:
    return f"{proceso}_{parametro:g}_{'ciclo' if ciclo else 'sin-ciclo'}"


def razon_escalas(T_C_h: float, corte_h: float | None, escala_h: float) -> float:
    """Razón máx/mín de las escalas de la cola disponibles (≥ 2), con piso de un paso en T_(C); NaN si no hay dos."""
    disponibles = [v for v in (max(T_C_h, D.PASO_H), corte_h, escala_h) if v is not None and np.isfinite(v)]
    return max(disponibles) / min(disponibles) if len(disponibles) >= 2 else float("nan")


def escalas_cola(serie: pd.Series, meses=S.INVIERNO, q: float = S.Q_UMBRAL) -> dict:
    """Las tres escalas de la cola de una serie, con estimación dentro de ``meses``, y su razón máx/mín."""
    u = float(np.nanquantile(serie.to_numpy(), q))
    bl = D.bloques_anuales(serie)
    dc = D.declustering_bloques(bl, u, meses)
    corte = D.corte_natural(D.huecos_bloques(bl, u, meses) * D.PASO_H)
    ex = D.extremograma_bloques(bl, u, meses, factor=FACTOR_EXTREMOGRAMA)
    sens = {f: D.escala_extremograma(ex.rho, ex.ref, f)[0] for f in SENSIBILIDADES_FACTOR}
    razon = razon_escalas(dc.T_C_horas, corte.corte_h, ex.escala_h)
    n_exc = sum(int(D.excedencias(b, u).sum()) for b in bl)
    return {
        "u": u,
        "theta": dc.theta,
        "n_int": dc.n_int,
        "n_exc_invierno": dc.n_exc,
        "frac_invierno": dc.n_exc / n_exc if n_exc else float("nan"),
        "T_C_h": dc.T_C_horas,
        "corte_h": float("nan") if corte.corte_h is None else corte.corte_h,
        "n_modas": corte.n_modas,
        "mas_de_dos_modas": corte.mas_de_dos_modas,
        "escala_extremograma_h": ex.escala_h,
        **{f"escala_extremograma_{f:g}_h": v for f, v in sens.items()},
        "ultimo_2x_h": ex.ultimo_2x_h,
        "rho_3h": float(ex.rho[1]),
        "ref_3h": float(ex.ref[1]),
        "razon": razon,
    }


def replica(proceso: str, parametro: float, ciclo: bool, i: int) -> dict:
    rng = semillas.generador("f06", "etapa0", proceso, f"{parametro:g}", str(ciclo), str(i))
    return escalas_cola(S.generar(proceso, parametro, ciclo, rng)) | {
        "config": nombre_config(proceso, parametro, ciclo),
        "proceso": proceso,
        "parametro": parametro,
        "ciclo": ciclo,
        "replica": i,
    }


def _trabajo(args) -> list[dict]:
    proceso, parametro, ciclo, indices = args
    return [replica(proceso, parametro, ciclo, i) for i in indices]


def tareas(replicas: int) -> list[tuple[str, tuple]]:
    out = []
    for proceso, parametro in PROCESOS:
        for ciclo in CICLOS:
            for a in range(0, replicas, TROZO):
                idx = list(range(a, min(a + TROZO, replicas)))
                out.append((f"{nombre_config(proceso, parametro, ciclo)}__{a:04d}", (proceso, parametro, ciclo, idx)))
    return out


def factor_concordancia(razones) -> float:
    r = np.asarray(razones, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.quantile(r, CUANTIL_FACTOR)) if r.size else float("nan")


def resumen_por_config(df: pd.DataFrame) -> list[dict]:
    out = []
    for cfg, g in df.groupby("config", sort=False):
        fila = {"config": cfg, "replicas": int(len(g)), "razon_no_calculable": int((~np.isfinite(g["razon"])).sum())}
        for c in ("theta", "frac_invierno", "T_C_h", "corte_h", "escala_extremograma_h", "razon"):
            v = g[c].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            fila[c] = {"n": int(v.size)}
            if v.size:
                fila[c] |= {"p5": float(np.quantile(v, 0.05)), "mediana": float(np.median(v))}
                fila[c]["p95"] = float(np.quantile(v, 0.95))
        fila["sin_corte"] = int((~np.isfinite(g["corte_h"])).sum())
        fila["p95_razon"] = factor_concordancia(g["razon"])
        out.append(fila)
    return out


def etapa0(raiz_datos=None, procesos_: int = 4, replicas: int = REPLICAS, previo: bool = False) -> Path | None:
    if previo:
        t0 = time.time()
        for proceso, parametro in PROCESOS:
            for ciclo in CICLOS:
                replica(proceso, parametro, ciclo, 0)
        seg = (time.time() - t0) / (len(PROCESOS) * len(CICLOS))
        total = seg * replicas * len(PROCESOS) * len(CICLOS) / max(procesos_, 1) / 60
        print(f"previo: {seg:.2f} s por réplica; etapa completa ≈ {total:.1f} min con {procesos_} procesos", flush=True)
        return None
    raiz = rutas.directorio_datos(raiz_datos)
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA0)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA0), replicas, TROZO)
    avance = ruta_avance(raiz) / clave
    print(f"Avance reanudable en {avance}", flush=True)
    with etapa.staging_protegido(salida, raices) as st:
        t0 = time.time()
        aviso = lambda t: print(f"  etapa 0: {t}", flush=True)  # noqa: E731
        partes = reanudable.ejecutar(_trabajo, tareas(replicas), avance, procesos_, aviso)
        df = pd.DataFrame([f for p in partes for f in p])
        df.to_csv(st / "replicas_etapa0.csv", index=False, float_format="%.6g", lineterminator="\n")
        factor = factor_concordancia(df["razon"])
        no_discrimina = bool(factor > UMBRAL_NO_DISCRIMINA)
        resumen = {
            "valores_del_sitio_leidos": False,
            "factor_concordancia": factor,
            "regla_no_discrimina": no_discrimina,
            "criterio": (
                f"Factor = P{int(100 * CUANTIL_FACTOR)} de la razón máx/mín de las escalas de la cola (T_(C) a P99, "
                f"corte natural, escala del extremograma con factor {FACTOR_EXTREMOGRAMA}) sobre toda la rejilla, "
                f"estimación dentro de mayo–octubre (contrato §3.5 y §8; v5, S-17). Si supera "
                f"{UMBRAL_NO_DISCRIMINA:g}, la regla no discrimina."
            ),
            "rejilla": {
                "procesos": [list(p) for p in PROCESOS],
                "ciclos": list(CICLOS),
                "replicas": replicas,
                "largo": S.LARGO,
                "inicio": S.INICIO,
                "cuantil_umbral": S.Q_UMBRAL,
                "amplitud_ciclo": {"ar1": S.amplitud("ar1"), "fs": S.amplitud("fs")},
                "fraccion_invierno_objetivo": S.FRACCION_INVIERNO,
            },
            "convenciones": [
                "T_(C) con piso de un paso (3 h) en la razón: T_(C) = 0 significa que cada excedencia es una tormenta.",
                "La razón usa las escalas disponibles (≥ 2): sin corte natural, la razón es entre T_(C) y el "
                "extremograma; una réplica con menos de dos escalas no entra en el P95 y se cuenta aparte.",
            ],
            "replicas_sin_razon": int((~np.isfinite(df["razon"])).sum()),
            "por_config": resumen_por_config(df),
            "segundos": round(time.time() - t0),
        }
        json_estricto.escribir(st / "etapa0.json", resumen)
        print(f"Factor de concordancia: {factor:.3f}{' (la regla NO discrimina)' if no_discrimina else ''}", flush=True)
        motivos = [] if np.isfinite(factor) else ["El factor de concordancia no es finito"]
        criterios = {"replicas": replicas, "configuraciones": len(PROCESOS) * len(CICLOS), "factor": factor}
        publicada = etapa.publicar(
            st, salida, etapa.Gate(not motivos, criterios, motivos), raices, {}, raiz, semillas.semilla("f06", "etapa0")
        )
        reanudable.limpiar(avance)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clima-etapa0", description="Etapa 0 sintética de la F6 (factor de concordancia)")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=REPLICAS)
    ap.add_argument("--previo", action="store_true", help="Cronometra una réplica por configuración; no publica.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = etapa0(a.datos, a.procesos, a.replicas, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    if salida:
        print(f"Publicado en {salida}")
    return 0
