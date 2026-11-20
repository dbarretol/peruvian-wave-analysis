"""Piloto de ADR-0034, ``eventos-piloto-r3``: R3 con el ajuste rápido y con ``ajustar_gpd`` sobre las mismas réplicas.

Demuestra que el atajo no cambia la **regla**, no solo el ajuste: en cada réplica de 6 celdas de §5.1 (100 réplicas
por celda), R3 se calcula dos veces con la misma semilla de bootstrap en cada u_k; solo cambia la función de los
reajustes. Se publica, por celda y en total: réplicas con el mismo u*, mayor |Δp| entre los dos p de un mismo u_k,
número de u_k con p distinto y el estado de la regla en cada versión.

Celdas (N0 y años del conteo previo, §5.0): A con ξ = −0.1 (N_mín) y +0.1 (N_máx); B+ con ξ0 = −0.1 (N_mín); B− con
ξ0 = −0.1 (N_mín; ξ = −0.25 bajo u0); C con k = 1.5 (N_máx) y k = 2 (N_mín). Incluyen las más difíciles para el ajuste
rápido: el N0 menor, cuyo umbral más alto deja ≈ 30 picos, con la forma más negativa de A, de B− y de C (donde más
delega y donde una diferencia mínima en p podría resolver distinto el empate de R3). Sin datos del sitio más allá del
conteo y de u_id.

Reanudable (trozos de 10 réplicas). ``--previo`` cronometra una réplica por celda sin publicar. Salida en
``processed/F07-eventos/simulacion-previa/piloto-r3/``, hermana del conteo y de la simulación.
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
from ..extremos import gpd, gpd_perfil
from ..validacion import reanudable
from . import casos as CS
from . import salidas as SA
from . import umbral as U

ETAPA = orquestador.Etapa("eventos-piloto-r3", Path(__file__).resolve(), Path("."))
REPLICAS = 100
TROZO = 10
CELDAS = (("A", -0.1, 0), ("A", 0.1, 1), ("B+", -0.1, 0), ("B-", -0.1, 0), ("C", 1.5, 1), ("C", 2.0, 0))


def ruta_salida(raiz: Path) -> Path:
    return SA.raiz_fase(raiz) / "simulacion-previa" / "piloto-r3"


def ruta_avance(raiz: Path) -> Path:
    return Path(raiz) / "interim" / SA.FASE / "piloto-r3-avance"


def celdas_piloto(conteo: dict) -> list[CS.Celda]:
    n0 = conteo["n0_simulacion"]
    u_id = float(conteo["u_id_m"])
    return [CS.Celda(c, f, int(n0["n0"][i]), float(n0["anios_simulados"][i]), u_id) for c, f, i in CELDAS]


def r3_dos_ajustes(picos: np.ndarray, celda: CS.Celda, i: int) -> dict:
    rej = U.rejilla(picos)
    p_rap, p_ref = [], []
    for k, u in enumerate(rej.u):
        exc = picos[picos > u] - u
        a = gpd.ajustar_gpd(exc)
        if not a.valido:
            p_rap.append(np.nan)
            p_ref.append(np.nan)
            continue
        nombre = ("f07", "piloto-r3", "bootstrap", celda.nombre, str(i), str(k))
        p_rap.append(U.p_valor_r3(exc, a, u, semillas.generador(*nombre), reajustar=gpd_perfil.ajustar_rapido).p)
        p_ref.append(U.p_valor_r3(exc, a, u, semillas.generador(*nombre), reajustar=gpd.ajustar_gpd).p)
    r_rap, r_ref = U.regla_r3(p_rap), U.regla_r3(p_ref)
    dp = np.abs(np.array(p_rap) - np.array(p_ref))
    return {
        "celda": celda.nombre,
        "replica": i,
        "n_picos": int(picos.size),
        "indice_rapido": r_rap.indice,
        "indice_referencia": r_ref.indice,
        "estado_rapido": r_rap.estado,
        "estado_referencia": r_ref.estado,
        "mismo_u": r_rap.indice == r_ref.indice and r_rap.estado == r_ref.estado,
        "max_dp": float(np.nanmax(dp)) if np.any(np.isfinite(dp)) else 0.0,
        "u_con_p_distinto": int(np.sum(dp > 1e-12)),
    }


def replica(celda: CS.Celda, i: int) -> dict:
    rep = CS.replica(celda, semillas.generador("f07", "piloto-r3", celda.nombre, str(i)))
    return r3_dos_ajustes(rep.picos, celda, i)


def _trabajo(args) -> list[dict]:
    celda, indices = args
    return [replica(celda, i) for i in indices]


def tareas(cs: list[CS.Celda], replicas: int) -> list[tuple[str, tuple]]:
    return [
        (f"{c.nombre}__{a:04d}", (c, list(range(a, min(a + TROZO, replicas)))))
        for c in cs
        for a in range(0, replicas, TROZO)
    ]


def resumen(df: pd.DataFrame) -> dict:
    por = []
    for nombre, g in df.groupby("celda", sort=False):
        por.append({
            "celda": nombre,
            "replicas": int(len(g)),
            "mismo_u": int(g["mismo_u"].sum()),
            "max_dp": float(g["max_dp"].max()),
            "replicas_con_algun_p_distinto": int((g["u_con_p_distinto"] > 0).sum()),
        })  # fmt: skip
    return {
        "replicas": int(len(df)),
        "mismo_u": int(df["mismo_u"].sum()),
        "max_dp": float(df["max_dp"].max()),
        "replicas_con_algun_p_distinto": int((df["u_con_p_distinto"] > 0).sum()),
        "por_celda": por,
    }


def ejecutar(raiz_datos=None, procesos_: int = 4, replicas: int = REPLICAS, previo: bool = False) -> Path | None:
    raiz = rutas.directorio_datos(raiz_datos)
    d_conteo = SA.ruta_conteo_previo(raiz)
    etapa.exigir_publicada(d_conteo)
    ins = {"conteo_previo": d_conteo / "conteo_previo.json"}
    conteo = json_estricto.leer(ins["conteo_previo"])
    cs = celdas_piloto(conteo)
    if previo:
        t0 = time.time()
        for c in cs:
            replica(c, 0)
        seg = (time.time() - t0) / len(cs)
        total = seg * replicas * len(cs) / max(procesos_, 1) / 60
        print(
            f"previo: {seg:.1f} s por réplica; piloto completo ≈ {total:.0f} min con {procesos_} procesos", flush=True
        )
        return None
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA), etapa.exigir_insumos(ins), replicas, TROZO)
    avance = ruta_avance(raiz) / clave
    print(f"Avance reanudable en {avance}", flush=True)
    with etapa.staging_protegido(salida, raices) as st:
        aviso = lambda t: print(f"  piloto R3: {t}", flush=True)  # noqa: E731
        partes = reanudable.ejecutar(_trabajo, tareas(cs, replicas), avance, procesos_, aviso)
        df = pd.DataFrame([f for p in partes for f in p])
        df.to_csv(st / "piloto_r3.csv", index=False, float_format="%.6g", lineterminator="\n")
        res = resumen(df) | {
            "criterio": (
                "ADR-0034: R3 con los reajustes por perfil (gpd_perfil.ajustar_rapido) y con ajustar_gpd, misma "
                "semilla de bootstrap en cada u_k. El atajo se acepta si R3 elige el mismo u* (y estado) en todas las "
                "réplicas."
            ),
            "celdas": [c.nombre for c in cs],
        }
        res["atajo_aceptado"] = bool(res["mismo_u"] == res["replicas"])
        json_estricto.escribir(st / "piloto_r3.json", res)
        print(f"Mismo u* en {res['mismo_u']}/{res['replicas']} réplicas; máx |Δp| = {res['max_dp']:.4f}", flush=True)
        if not res["atajo_aceptado"]:
            print("AVISO: R3 eligió otro u* en alguna réplica: el atajo cambia la regla (ADR-0034).", flush=True)
        motivos = [] if len(df) == replicas * len(cs) else ["Faltan réplicas del piloto"]
        criterios = {"replicas_por_celda": replicas, "celdas": len(cs)}
        gate = etapa.Gate(not motivos, criterios, motivos)
        publicada = etapa.publicar(st, salida, gate, raices, ins, raiz, semillas.semilla("f07", "piloto-r3"))
        reanudable.limpiar(avance)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-piloto-r3", description="Piloto de ADR-0034: R3 con dos ajustes")
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
