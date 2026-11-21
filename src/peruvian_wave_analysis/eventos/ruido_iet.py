"""Ruido del RL100 entre IET vecinos, ``eventos-ruido-iet`` (contrato de la F7, §5.3 y §3.3; sin datos del sitio).

Calibra la tolerancia de la regla del IET (§3.3) bajo el nulo: procesos sintéticos sin ningún cambio real con el IET,
en los que toda diferencia del RL100 entre IET vecinos es ruido.

- **Procesos** (los de la etapa 0 de la F6, ``clima.sinteticos``, con el ciclo anual calibrado al 86 % de excedencias
  del P99 en mayo–octubre): AR(1) con τ_e ∈ {24, 48, 72} h y el proceso de Ferro y Segers con θ ∈ {0.25, 0.5, 0.75}.
  - **Ferro y Segers en escala logarítmica** (``ESCALA_FS = "log"``): log X tiene cola de Gumbel (ξ = 0, como Hs) y
    el mismo índice extremal θ, porque θ no cambia con una transformación monótona; el ciclo queda aditivo, como en el
    AR(1). Con la Fréchet unitaria (ξ = 1) el RL100 sería mucho más ruidoso que el de Hs. **Pendiente de confirmar.**
- **Escala de Hs** (``a_escala_hs``): cada réplica se lleva a metros con la transformación afín que manda su P95 a
  2.70 m y su P99 a 3.15 m, los de Hs de la F3 ya publicados (``resumen_candidatos.json``). Así:
  - el 1 % se mide sobre un RL100 en la misma escala que en el sitio (un cambio relativo depende de la ubicación);
  - u_id = cuantil 0.95 = 2.70 m y la bajada intermedia es la de 0.5 m, sin reescalar;
  - θ y ξ no cambian, porque la transformación es afín.

  El contrato decía «la bajada llevada a la desviación típica del proceso», pero la Fréchet no tiene varianza, y sin
  fijar la ubicación el 1 % no tiene sentido. **Pendiente de confirmar.**
- **Rejilla del IET:** 42 (solo como vecino de abajo), 48, 54, 60, 66, 72, 84, 96 y 120 h.
- **Umbrales de referencia:** cuantiles 0.98, 0.99 y 0.995 de los estados del proceso. RL100 con ``umbral.ajustar_en``.
- **En cada réplica (C-07):** el máximo de las 6 |ΔRL100| relativas de cada IET (2 vecinos × 3 umbrales; 120 h solo
  tiene el de abajo). Con eso se aplica después la regla completa con cualquier tolerancia.
- **Decide (§3.3):** si con el 1 % el IET de 48 h no cumple en más del 20 % de las réplicas, la tolerancia pasa al
  P80 del máximo en 48 h. Se publica por proceso y con todos juntos (``CRITERIO``: **pendiente de confirmar**).
- **Se publica:** la probabilidad, bajo el nulo, de que la regla elija cada IET o marque «IET sensible».

Reanudable; ``--previo`` cronometra. Salida en ``processed/F07-eventos/simulacion-previa/ruido-iet/``, hermana de la
simulación y del conteo.
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
from ..clima import sinteticos as S
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..validacion import reanudable
from . import identificacion as ID
from . import umbral as U

ETAPA = orquestador.Etapa("eventos-ruido-iet", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
PROCESOS = (("ar1", 24.0), ("ar1", 48.0), ("ar1", 72.0), ("fs", 0.25), ("fs", 0.5), ("fs", 0.75))
IETS_H = (42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 84.0, 96.0, 120.0)
IET_MIN_H = 48.0
Q_ID = 0.95
Q_REFERENCIA = (0.98, 0.99, 0.995)
BAJADA_M = 0.5
P95_HS_M, P99_HS_M = 2.70, 3.15  # F3, resumen_candidatos.json
TOL_BASE = 0.01
FRACCION_MAX_FALLO = 0.20
CUANTIL_TOL = 0.80
REPLICAS = 200
TROZO = 10
ESCALA_FS = "log"
CICLO = True


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "simulacion-previa" / "ruido-iet"


def ruta_avance(raiz: Path) -> Path:
    return Path(raiz) / "interim" / FASE / "ruido-iet-avance"


def serie(proceso: str, parametro: float, rng: np.random.Generator, n: int = S.LARGO) -> pd.Series:
    x = S.generar(proceso, parametro, CICLO, rng, n)
    if proceso == "fs" and ESCALA_FS == "log":
        x = np.log(x)
    return x


def a_escala_hs(x: pd.Series) -> pd.Series:
    """Afín: P95 de la réplica → 2.70 m y P99 → 3.15 m (Hs de la F3)."""
    q95, q99 = np.quantile(x.to_numpy(), [0.95, 0.99])
    return P95_HS_M + (x - q95) * (P99_HS_M - P95_HS_M) / (q99 - q95)


def cambios(rl: np.ndarray) -> np.ndarray:
    """rl: (IET, umbral). Máximo de |RL(vecino)/RL(iet) − 1| sobre vecinos y umbrales, para cada IET ≥ el piso.

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


def replica(proceso: str, parametro: float, i: int, n: int = S.LARGO) -> dict:
    rng = semillas.generador("f07", "ruido-iet", proceso, f"{parametro:g}", str(i))
    x = a_escala_hs(serie(proceso, parametro, rng, n))
    v = x.to_numpy()
    u_id = float(np.quantile(v, Q_ID))
    baj = BAJADA_M
    us = np.quantile(v, Q_REFERENCIA)
    anios = n * S.PASO_H / (365.25 * 24.0)
    rl = np.full((len(IETS_H), len(us)), np.nan)
    eventos = []
    for j, iet in enumerate(IETS_H):
        ev = ID.identificar(x, u_id, iet, baj)
        picos = v[ev["i_pico"].to_numpy()]
        eventos.append(int(picos.size))
        for k, u in enumerate(us):
            rl[j, k] = U.ajustar_en(picos, float(u), anios, (100.0,)).rl_valor(100.0)
    return {
        "proceso": proceso,
        "parametro": parametro,
        "replica": i,
        "bajada": baj,
        "eventos": eventos,
        "rl100": rl.tolist(),
        "maximo_cambio": cambios(rl).tolist(),
    }


def _trabajo(args) -> list[dict]:
    proceso, parametro, indices = args
    return [replica(proceso, parametro, i) for i in indices]


def tareas(replicas: int) -> list[tuple[str, tuple]]:
    return [
        (f"{p}_{a:g}__{b:04d}", (p, a, list(range(b, min(b + TROZO, replicas)))))
        for p, a in PROCESOS
        for b in range(0, replicas, TROZO)
    ]


def calibrar(df: pd.DataFrame) -> dict:
    """§3.3: fracción de réplicas en que 48 h no cumple con el 1 %, P80 del máximo en 48 h y la regla resultante."""
    j48 = IETS_H.index(48.0)
    m48 = np.array([m[j48] for m in df["maximo_cambio"]])

    def bloque(mask) -> dict:
        x = m48[mask]
        fallo = float(np.mean(x > TOL_BASE))
        tol = float(np.quantile(x, CUANTIL_TOL)) if fallo > FRACCION_MAX_FALLO else TOL_BASE
        p80 = float(np.quantile(x, CUANTIL_TOL))
        return {"replicas": int(mask.sum()), "fallo_48_con_1": fallo, "p80_maximo_48": p80, "tolerancia": tol}

    por = {f"{p}_{a:g}": bloque(((df["proceso"] == p) & (df["parametro"] == a)).to_numpy()) for p, a in PROCESOS}
    return {"juntos": bloque(np.ones(len(df), bool)), "por_proceso": por}


def eleccion_bajo_el_nulo(df: pd.DataFrame, tol: float) -> dict:
    """Probabilidad de que la regla elija cada IET o marque «IET sensible», por proceso y con todos juntos."""

    def dist(g) -> dict:
        r = [regla_iet(np.array(m), IETS_H, tol) for m in g["maximo_cambio"]]
        out = {f"{iet:g}": float(np.mean([x[0] == iet and not x[1] for x in r])) for iet in IETS_H[1:]}
        return out | {"iet_sensible": float(np.mean([x[1] for x in r]))}

    por = {f"{p}_{a:g}": dist(g) for (p, a), g in df.groupby(["proceso", "parametro"], sort=False)}
    return {"tolerancia": tol, "juntos": dist(df), "por_proceso": por}


def ejecutar(raiz_datos=None, procesos_: int = 4, replicas: int = REPLICAS, previo: bool = False) -> Path | None:
    raiz = rutas.directorio_datos(raiz_datos)
    if previo:
        t0 = time.time()
        for p, a in PROCESOS:
            replica(p, a, 0)
        seg = (time.time() - t0) / len(PROCESOS)
        total = seg * replicas * len(PROCESOS) / max(procesos_, 1) / 60
        print(f"previo: {seg:.1f} s por réplica; ≈ {total:.0f} min con {procesos_} procesos", flush=True)
        return None
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA), replicas, TROZO)
    avance = ruta_avance(raiz) / clave
    print(f"Avance reanudable en {avance}", flush=True)
    with etapa.staging_protegido(salida, raices) as st:
        aviso = lambda t: print(f"  ruido del IET: {t}", flush=True)  # noqa: E731
        partes = reanudable.ejecutar(_trabajo, tareas(replicas), avance, procesos_, aviso)
        df = pd.DataFrame([f for p in partes for f in p])
        filas = []
        for _, f in df.iterrows():
            for j, iet in enumerate(IETS_H):
                fila = {"proceso": f["proceso"], "parametro": f["parametro"], "replica": f["replica"], "iet_h": iet,
                        "eventos": f["eventos"][j], "maximo_cambio": f["maximo_cambio"][j]}  # fmt: skip
                fila |= {f"rl100_q{q:g}": f["rl100"][j][k] for k, q in enumerate(Q_REFERENCIA)}
                filas.append(fila)
        pd.DataFrame(filas).to_csv(st / "replicas.csv", index=False, float_format="%.8g", lineterminator="\n")
        cal = calibrar(df)
        res = {
            "calibracion": cal,
            "eleccion_con_1": eleccion_bajo_el_nulo(df, TOL_BASE),
            "eleccion_con_tolerancia_juntos": eleccion_bajo_el_nulo(df, cal["juntos"]["tolerancia"]),
            "parametros": {
                "escala_fs": ESCALA_FS,
                "ciclo": CICLO,
                "q_id": Q_ID,
                "q_referencia": list(Q_REFERENCIA),
                "escala": "afín, P95 → 2.70 m y P99 → 3.15 m (F3); bajada 0.5 m",
                "iets_h": list(IETS_H),
                "replicas": replicas,
            },  # fmt: skip
        }
        json_estricto.escribir(st / "ruido_iet.json", res)
        motivos = [] if len(df) == replicas * len(PROCESOS) else ["Faltan réplicas"]
        gate = etapa.Gate(not motivos, {"replicas": replicas, "procesos": len(PROCESOS)}, motivos)
        publicada = etapa.publicar(st, salida, gate, raices, {}, raiz, semillas.semilla("f07", "ruido-iet"))
        reanudable.limpiar(avance)
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
