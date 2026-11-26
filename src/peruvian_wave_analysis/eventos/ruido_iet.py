"""Ruido del RL100 entre IET vecinos, ``eventos-ruido-iet`` (contrato de la F7, §5.3 y §3.3).

Calibra la tolerancia de la regla del IET (§3.3) bajo el nulo: procesos sintéticos sin ningún cambio real con el IET,
en los que toda diferencia del RL100 entre IET vecinos es ruido. Del sitio solo usa números ya publicados (el conteo de
§5.0 y el P95 y el P99 de la F3) y la duración media de las tormentas, como comprobación; ninguna magnitud de los picos.

**Apartamientos de §5.3, anteriores a los resultados del IET (confirmados por el usuario el 2026-11-20):**

1. **Ferro y Segers en escala logarítmica.** log X tiene cola de Gumbel (ξ = 0, como Hs) y el mismo índice extremal θ
   (θ no cambia con una transformación creciente). La Fréchet unitaria tiene ξ = 1 y no tiene varianza.
2. **Escala de Hs** (``a_escala_hs``). Cada réplica se lleva a metros con la transformación afín que manda su P95 al
   P95 de Hs de la F3 (2.70 m) y su P99 al P99 (3.15 m). Así el 1 % se mide sobre un RL100 en la misma escala que en el
   sitio, u_id es el P95 y la bajada es la de 0.5 m. θ y ξ no cambian.
3. **Persistencia calibrada con el conteo de §5.0** (rotulado: usa el número de tormentas del sitio, como el N0).
   - Con los procesos de la etapa 0 de la F6 (AR(1) con τ_e de 24 a 72 h; Ferro y Segers con θ de 0.25 a 0.75) salían
     de 17 a 55 eventos por año con 48 h, frente a 12.4 en el sitio: tormentas más cortas, casi sin separaciones entre
     48 y 120 h que agrupar, y una calibración optimista por construcción.
   - Cada familia (AR(1) y Ferro y Segers, ambas con el ciclo anual de la F6) se calibra a tres tasas de eventos con
     48 h. La central es la de E0 con 48 h (la rama donde se aplica la regla). Las otras dos son un **margen** alrededor
     de ella, no una incertidumbre medida: la de E0 con 120 h y la de E1 con 48 h, que cambian a la vez el IET o el
     período.
   - La variedad queda en la forma de la dependencia (el AR(1) gaussiano, cuya dependencia extremal se desvanece en la
     cola, frente al de Ferro y Segers, con dependencia extremal real) y en el margen de tasas.
   - El parámetro se busca con 8 réplicas de semillas fijas (números aleatorios comunes) y se publica con la tasa
     lograda. Un proceso que no llegue a su tasa dentro del rango del parámetro se publica aparte, fuera del criterio.
4. **Criterio del 20 %, por proceso:** si en algún proceso el IET de 48 h no cumple con el 1 % en más del 20 % de las
   réplicas, la tolerancia pasa al P80 del máximo en 48 h de ese proceso, tomando el mayor; se publica qué proceso la
   decide.

**Comprobación (no es un objetivo):** la duración media de las tormentas del sitio con 48 h (E0, hasta 2025-12-31),
junto a la de cada proceso calibrado. Si el sitio queda fuera del rango de los procesos, se dice.

**En cada réplica (C-07):** rejilla de IET de 42 h (solo como vecino de abajo) a 120 h; umbrales de referencia en los
cuantiles 0.98, 0.99 y 0.995 de los estados; el máximo de las 6 |ΔRL100| relativas de cada IET (2 vecinos × 3
umbrales; 120 h solo tiene el de abajo). Se publica la probabilidad, bajo el nulo, de que la regla elija cada IET o
marque «IET sensible».

Reanudable; ``--previo`` cronometra. Salida en ``processed/F07-eventos/simulacion-previa/ruido-iet/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import rutas
from ..clima import sinteticos as S
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..datos import carga
from ..validacion import reanudable
from . import conteo_previo as CP
from . import identificacion as ID
from . import salidas as SA
from . import umbral as U

ETAPA = orquestador.Etapa("eventos-ruido-iet", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
FAMILIAS = ("ar1", "fs")
RANGO_PARAMETRO = {"ar1": (24.0, 1000.0), "fs": (0.01, 0.75)}  # τ_e en h; θ
TASAS = (("baja", "E0", 120.0), ("central", "E0", 48.0), ("alta", "E1", 48.0))  # (etiqueta, rama, IET del conteo)
IET_CALIBRACION_H = 48.0
REPLICAS_CALIBRACION = 8
TOL_TASA = 0.02  # |tasa lograda / objetivo − 1| para que el proceso entre en el criterio
IETS_H = (42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 84.0, 96.0, 120.0)
IET_MIN_H = 48.0
Q_ID = 0.95
Q_REFERENCIA = (0.98, 0.99, 0.995)
BAJADA_M = ID.BAJADA_M
TOL_BASE = 0.01
FRACCION_MAX_FALLO = 0.20
CUANTIL_TOL = 0.80
REPLICAS = 200
TROZO = 10
CICLO = True
ANIOS_SINTETICOS = S.LARGO * S.PASO_H / (365.25 * 24.0)


@dataclass(frozen=True)
class Proceso:
    familia: str
    parametro: float
    etiqueta: str  # p. ej. «ar1_central»


def ruta_salida(raiz: Path) -> Path:
    return SA.raiz_fase(raiz) / "simulacion-previa" / "ruido-iet"


def ruta_avance(raiz: Path) -> Path:
    return Path(raiz) / "interim" / FASE / "ruido-iet-avance"


# --- Series ----------------------------------------------------------------------------------------


def serie(familia: str, parametro: float, rng: np.random.Generator, n: int = S.LARGO) -> pd.Series:
    x = S.generar(familia, parametro, CICLO, rng, n)
    return np.log(x) if familia == "fs" else x


def a_escala_hs(x: pd.Series, p95: float, p99: float) -> pd.Series:
    """Afín: P95 de la réplica → P95 de Hs y P99 → P99 de Hs (F3)."""
    q95, q99 = np.quantile(x.to_numpy(), [0.95, 0.99])
    return p95 + (x - q95) * (p99 - p95) / (q99 - q95)


def eventos_48(
    familia: str, parametro: float, rng: np.random.Generator, p95: float, p99: float, n: int
) -> pd.DataFrame:
    x = a_escala_hs(serie(familia, parametro, rng, n), p95, p99)
    return ID.identificar(x, p95, IET_CALIBRACION_H, BAJADA_M)


# --- Calibración de la persistencia ----------------------------------------------------------------


def tasa_media(familia: str, parametro: float, p95: float, p99: float, n: int = S.LARGO) -> tuple[float, float]:
    """Eventos por año y duración media (h) con 48 h, promedio de las réplicas de calibración (semillas fijas)."""
    anios = n * S.PASO_H / (365.25 * 24.0)
    tasas, dur = [], []
    for j in range(REPLICAS_CALIBRACION):
        ev = eventos_48(familia, parametro, semillas.generador("f07", "ruido-iet", "calibracion", familia, str(j)),
                        p95, p99, n)  # fmt: skip
        tasas.append(len(ev) / anios)
        dur.append(float(ev["duracion_h"].mean()))
    return float(np.mean(tasas)), float(np.mean(dur))


def calibrar_parametro(familia: str, objetivo: float, p95: float, p99: float, n: int = S.LARGO) -> dict:
    """Parámetro de la familia que da ``objetivo`` eventos por año con 48 h (Brent en el logaritmo del parámetro)."""
    lo, hi = np.log(RANGO_PARAMETRO[familia])

    def f(z):
        return tasa_media(familia, float(np.exp(z)), p95, p99, n)[0] - objetivo

    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):  # no llega dentro del rango: el extremo más cercano, fuera del criterio
        z = lo if abs(flo) < abs(fhi) else hi
    else:
        z = brentq(f, lo, hi, xtol=1e-3)
    par = float(np.exp(z))
    t, d = tasa_media(familia, par, p95, p99, n)
    return {"familia": familia, "parametro": par, "tasa_objetivo": objetivo, "tasa_lograda": t, "duracion_media_h": d,
            "en_el_criterio": bool(abs(t / objetivo - 1.0) <= TOL_TASA)}  # fmt: skip


def calibrar_procesos(tasas: dict[str, float], p95: float, p99: float, n: int = S.LARGO) -> list[dict]:
    out = []
    for fam in FAMILIAS:
        for etiqueta, objetivo in tasas.items():
            c = calibrar_parametro(fam, objetivo, p95, p99, n)
            out.append(c | {"etiqueta": f"{fam}_{etiqueta}"})
            print(f"  calibración {fam}_{etiqueta}: parámetro {c['parametro']:.4g}, {c['tasa_lograda']:.2f} eventos "
                  f"por año (objetivo {objetivo:.2f})", flush=True)  # fmt: skip
    return out


# --- Una réplica -----------------------------------------------------------------------------------


def cambios(rl: np.ndarray) -> np.ndarray:
    """rl: (IET, umbral). Máximo de |RL(vecino)/RL(iet) − 1| sobre vecinos y umbrales, para cada IET.

    Un RL100 nulo da un cambio infinito: ese IET no puede cumplir."""
    out = []
    for j in range(rl.shape[0]):
        vec = [v for v in (j - 1, j + 1) if 0 <= v < rl.shape[0]]
        d = np.abs(rl[vec] / rl[j] - 1.0)
        out.append(float(np.max(np.where(np.isfinite(d), d, np.inf))))
    return np.array(out)


def regla_iet(maximos: np.ndarray, iets: Sequence[float], tol: float) -> tuple[float, bool]:
    """§3.3: el menor IET ≥ 48 h con cambio máximo ≤ tol; si ninguno, el de menor cambio y «IET sensible»."""
    iets = np.asarray(iets, dtype=float)
    cand = np.flatnonzero(iets >= IET_MIN_H)
    ok = [j for j in cand if maximos[j] <= tol]
    if ok:
        return float(iets[ok[0]]), False
    j = cand[int(np.argmin(maximos[cand]))]
    return float(iets[j]), True


def replica(proc: Proceso, i: int, p95: float, p99: float, n: int = S.LARGO) -> dict:
    rng = semillas.generador("f07", "ruido-iet", proc.etiqueta, str(i))
    x = a_escala_hs(serie(proc.familia, proc.parametro, rng, n), p95, p99)
    v = x.to_numpy()
    us = np.quantile(v, Q_REFERENCIA)
    anios = n * S.PASO_H / (365.25 * 24.0)
    rl = np.full((len(IETS_H), len(us)), np.nan)
    eventos, duracion = [], []
    for j, iet in enumerate(IETS_H):
        ev = ID.identificar(x, p95, iet, BAJADA_M)
        picos = v[ev["i_pico"].to_numpy()]
        eventos.append(int(picos.size))
        duracion.append(float(ev["duracion_h"].mean()))
        for k, u in enumerate(us):
            rl[j, k] = U.ajustar_en(picos, float(u), anios, (100.0,)).rl_valor(100.0)
    return {
        "proceso": proc.etiqueta,
        "familia": proc.familia,
        "parametro": proc.parametro,
        "replica": i,
        "eventos": eventos,
        "duracion_media_h": duracion,
        "rl100": rl.tolist(),
        "maximo_cambio": cambios(rl).tolist(),
    }


def _trabajo(args) -> list[dict]:
    proc, indices, p95, p99 = args
    return [replica(proc, i, p95, p99) for i in indices]


def tareas(procs: Sequence[Proceso], replicas: int, p95: float, p99: float) -> list[tuple[str, tuple]]:
    return [
        (f"{p.etiqueta}__{b:04d}", (p, list(range(b, min(b + TROZO, replicas))), p95, p99))
        for p in procs
        for b in range(0, replicas, TROZO)
    ]


# --- Agregados -------------------------------------------------------------------------------------


def calibrar(df: pd.DataFrame, en_criterio: Sequence[str]) -> dict:
    """§3.3, por proceso: fracción en que 48 h no cumple con el 1 % y P80 del máximo en 48 h. Rige el mayor P80 entre
    los procesos que superan el 20 % y están en el criterio; si ninguno lo supera, el 1 %."""
    j48 = IETS_H.index(48.0)
    por = {}
    for p, g in df.groupby("proceso", sort=False):
        x = np.array([m[j48] for m in g["maximo_cambio"]])
        por[str(p)] = {
            "replicas": int(x.size),
            "fallo_48_con_1": float(np.mean(x > TOL_BASE)),
            # sin interpolar: un cambio infinito (RL100 nulo) no da NaN
            "p80_maximo_48": float(np.quantile(x, CUANTIL_TOL, method="inverted_cdf")),
            "cambios_infinitos_48": int(np.sum(np.isinf(x))),
            "en_el_criterio": str(p) in en_criterio,
        }
    superan = {p: v for p, v in por.items() if v["en_el_criterio"] and v["fallo_48_con_1"] > FRACCION_MAX_FALLO}
    if superan:
        decide = max(superan, key=lambda p: superan[p]["p80_maximo_48"])
        tol = superan[decide]["p80_maximo_48"]
    else:
        decide, tol = None, TOL_BASE
    return {"tolerancia": tol, "proceso_que_decide": decide, "por_proceso": por}


def eleccion_bajo_el_nulo(df: pd.DataFrame, tol: float) -> dict:
    """Probabilidad de que la regla elija cada IET o marque «IET sensible», por proceso y con todos juntos."""

    def dist(g) -> dict:
        r = [regla_iet(np.array(m), IETS_H, tol) for m in g["maximo_cambio"]]
        out = {f"{iet:g}": float(np.mean([x[0] == iet and not x[1] for x in r])) for iet in IETS_H[1:]}
        return out | {"iet_sensible": float(np.mean([x[1] for x in r]))}

    por = {str(p): dist(g) for p, g in df.groupby("proceso", sort=False)}
    return {"tolerancia": tol, "juntos": dist(df), "por_proceso": por}


def comprobacion_duracion(sitio_h: float, calib: Sequence[dict]) -> dict:
    d = [c["duracion_media_h"] for c in calib if c["en_el_criterio"]]
    dentro = bool(d) and min(d) <= sitio_h <= max(d)
    return {"sitio_h": sitio_h, "procesos_min_h": min(d) if d else None, "procesos_max_h": max(d) if d else None,
            "sitio_dentro_del_rango": dentro}  # fmt: skip


# --- Datos del sitio (solo números ya publicados y la duración media de las tormentas) ---------------


def datos_del_sitio(raiz: Path) -> tuple[dict[str, Path], dict]:
    """Insumos y números del sitio: P95 y P99 de la F3, tasas del conteo de §5.0 y duración media con 48 h en E0."""
    ins = CP.rutas_insumos(raiz)
    d_conteo = SA.ruta_conteo_previo(raiz)
    etapa.exigir_publicada(d_conteo)
    ins = {"serie_nodo": ins["serie_nodo"], "resumen_candidatos": ins["resumen_candidatos"],
           "conteo_previo": d_conteo / "conteo_previo.json"}  # fmt: skip
    etapa.exigir_insumos(ins)
    rc = json_estricto.leer(ins["resumen_candidatos"])
    conteo = json_estricto.leer(ins["conteo_previo"])
    tasas = {}
    for etiqueta, rama, iet in TASAS:
        tasas[etiqueta] = next(
            float(c["eventos_por_anio"]) for c in conteo["conteos"] if c["rama"] == rama and c["iet_h"] == iet
        )
    serie_ = carga.cargar_serie(raiz, CP.NOMBRE)
    if "tiempo" in serie_.columns:
        serie_ = serie_.set_index("tiempo")
    hs = serie_["VHM0"].astype(float)
    p95 = float(rc["p95_hs_m"])
    ev = ID.en_periodo(ID.identificar(hs, p95, IET_CALIBRACION_H), hs.index[0], CP.FIN_MODELO)
    sitio = {"p95_hs_m": p95, "p99_hs_m": float(rc["p99_hs_m"]), "tasas": tasas,
             "duracion_media_sitio_h": float(ev["duracion_h"].mean()), "eventos_sitio_e0_48": int(len(ev))}  # fmt: skip
    return ins, sitio


# --- Etapa -----------------------------------------------------------------------------------------


def ejecutar(raiz_datos=None, procesos_: int = 4, replicas: int = REPLICAS, previo: bool = False) -> Path | None:
    raiz = rutas.directorio_datos(raiz_datos)
    ins, sitio = datos_del_sitio(raiz)
    p95, p99 = sitio["p95_hs_m"], sitio["p99_hs_m"]
    print(f"Sitio: {sitio['tasas']} eventos por año; duración media con 48 h {sitio['duracion_media_sitio_h']:.1f} h",
          flush=True)  # fmt: skip
    calib = calibrar_procesos(sitio["tasas"], p95, p99)
    procs = [Proceso(c["familia"], c["parametro"], c["etiqueta"]) for c in calib]
    if previo:
        t0 = time.time()
        for p in procs:
            replica(p, 0, p95, p99)
        seg = (time.time() - t0) / len(procs)
        total = seg * replicas * len(procs) / max(procesos_, 1) / 60
        print(f"previo: {seg:.1f} s por réplica; ≈ {total:.0f} min con {procesos_} procesos", flush=True)
        return None
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA), etapa.exigir_insumos(ins), replicas, TROZO)
    avance = ruta_avance(raiz) / clave
    print(f"Avance reanudable en {avance}", flush=True)
    with etapa.staging_protegido(salida, raices) as st:
        aviso = lambda t: print(f"  ruido del IET: {t}", flush=True)  # noqa: E731
        partes = reanudable.ejecutar(_trabajo, tareas(procs, replicas, p95, p99), avance, procesos_, aviso)
        df = pd.DataFrame([f for p in partes for f in p])
        filas = []
        for _, f in df.iterrows():
            for j, iet in enumerate(IETS_H):
                fila = {"proceso": f["proceso"], "replica": f["replica"], "iet_h": iet, "eventos": f["eventos"][j],
                        "duracion_media_h": f["duracion_media_h"][j],
                        "maximo_cambio": f["maximo_cambio"][j]}  # fmt: skip
                fila |= {f"rl100_q{q:g}": f["rl100"][j][k] for k, q in enumerate(Q_REFERENCIA)}
                filas.append(fila)
        pd.DataFrame(filas).to_csv(st / "replicas.csv", index=False, float_format="%.8g", lineterminator="\n")
        en_criterio = [c["etiqueta"] for c in calib if c["en_el_criterio"]]
        cal = calibrar(df, en_criterio)
        res = {
            "sitio": sitio,
            "calibracion_persistencia": calib,
            "comprobacion_duracion": comprobacion_duracion(sitio["duracion_media_sitio_h"], calib),
            "tolerancia": cal,
            "eleccion_con_1": eleccion_bajo_el_nulo(df, TOL_BASE),
            "eleccion_con_tolerancia": eleccion_bajo_el_nulo(df, cal["tolerancia"]),
            "parametros": {
                "escala_fs": "log",
                "ciclo": CICLO,
                "q_referencia": list(Q_REFERENCIA),
                "escala": "afín, P95 y P99 de la réplica → P95 y P99 de Hs (F3); bajada 0.5 m",
                "iets_h": list(IETS_H),
                "replicas": replicas,
                "replicas_calibracion": REPLICAS_CALIBRACION,
                "tol_tasa": TOL_TASA,
            },  # fmt: skip
            "valores_de_picos_leidos": False,
        }
        json_estricto.escribir(st / "ruido_iet.json", res)
        motivos = [] if len(df) == replicas * len(procs) else ["Faltan réplicas"]
        if not en_criterio:
            motivos.append("Ningún proceso llegó a su tasa: no hay criterio")
        gate = etapa.Gate(not motivos, {"replicas": replicas, "procesos": len(procs)}, motivos)
        publicada = etapa.publicar(st, salida, gate, raices, ins, raiz, semillas.semilla("f07", "ruido-iet"))
        reanudable.limpiar(avance)
        print(f"Tolerancia: {cal['tolerancia']:.4f} (decide: {cal['proceso_que_decide']})", flush=True)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-ruido-iet", description="Ruido del RL100 entre IET vecinos (§5.3)")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=REPLICAS)
    ap.add_argument("--previo", action="store_true", help="Cronometra una réplica por proceso; no publica.")
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
