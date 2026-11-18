"""Réplicas de la etapa 3 con el motor de la etapa 2 (sin modificarlo).

1. **Potencia recalculada (§5, «después de leer valores», y §3.3):** si el SI o el ρ observados de algún lado son
   peores que los nominales, o si los filtros de la etapa 3 quitan más del 1 % de un lado, se repiten las réplicas de
   b = 0 con la geometría tras el control de calidad y el punto peor de los dos (por clase de SI). Vale la potencia
   menor. Solo puede convertir un «descartado» en «no concluyente».
2. **Sesgo del estimador con el σ medido (§20.5, punto 3):** las mismas réplicas del nominal, pero con el ruido de 1 Hz
   medido por misión y nivel. Rotulado y descriptivo.

Cada réplica es la de ``validacion.potencia.replica`` con otro nombre de semilla y, en (2), otra función de ruido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..comun import semillas
from ..validacion import potencia as P
from ..validacion import simulacion as S
from . import ruido

TROZO = 25


def replica(
    lados: dict[str, S.LadoSim], r: dict[str, float], punto: P.Punto, i: int, prefijo: str, tabla_sigma: dict | None
) -> dict:
    """Como ``potencia.replica`` (sesgo nulo), con la semilla ``(f05, prefijo, punto, i)`` y, si hay ``tabla_sigma``,
    el ruido de 1 Hz medido (σ̂ por misión y nivel / √n) en lugar del de la rejilla."""
    rng = semillas.generador("f05", prefijo, punto.nombre, str(i))
    claves = set()
    for L in lados.values():
        claves |= set(S._claves(L, lado_boya=True))
    verdad = S.sortear_errores(claves, punto.factor_cal, rng)
    boot = S.sortear_errores(claves, punto.factor_cal, rng, size=P.C.B_POTENCIA)
    med, reps, reps_lado = {}, {}, {}
    for n, L in lados.items():
        m, o = S.par(L, r[n], punto.rho, rng)
        o = S.aplicar_calibracion(L, o, {k: v for k, v in verdad.items() if not k.startswith("lado_")})
        if tabla_sigma is not None and L.fuente == "altimetría":
            mis = np.asarray(L.tramo, object)
            mision = np.where(np.char.startswith(mis.astype(str), "HY-2A"), "HY-2A", mis.astype(str))
            df = pd.DataFrame({"mision": mision, "hs_obs": o, "n": L.npts})
            sig = ruido.sigma_superobs(df, tabla_sigma)
        else:
            sig = S.sigma_ruido(L, o, punto.factor_cal)
        o_obs = o + np.nan_to_num(sig) * rng.standard_normal(L.n)
        med[n] = S.medir(L, o_obs, np.nan_to_num(sig), P.C.B_POTENCIA, rng, m=m)
        reps[n] = S.con_calibracion(L, med[n], boot, lado_boya=False)
        reps_lado[n] = S.con_calibracion(L, med[n], boot, lado_boya=True)
    lado_verdad = float(np.log1p(verdad["lado_T0b"][2]) - np.log1p(verdad["lado_T1b"][2]))
    out = {
        "V1a": P._diferencia(med["T0"], reps["T0"], med["A"], reps["A"], P.ESTS_ATRIB),
        "V1b": P._diferencia(med["T0b"], reps["T0b"], med["T1b"], reps["T1b"], P.ESTS_ATRIB),
        "V1b_lado": P._diferencia(
            med["T0b"], reps_lado["T0b"], med["T1b"], reps_lado["T1b"], P.ESTS_ATRIB, lado_verdad
        ),
        "V2a": P._diferencia(med["A1"], reps["A1"], med["A2"], reps["A2"], P.ESTS_ATRIB),
    }
    for con_max, suf in ((True, ""), (False, "_sinmax")):
        v3a, v3b, sub = P._v3(med["A"], reps["A"], con_max)
        out[f"V3a{suf}"], out[f"V3b{suf}"], out[f"subida{suf}"] = v3a, v3b, sub
    out["B_lados"] = {n: med[n].B.tolist() for n in lados}
    out["estimable"] = {n: med[n].estimable.tolist() for n in lados}
    return out


def trabajo(args) -> list[dict]:
    """Una tarea (proceso hijo): réplicas ``indices`` de un punto."""
    ruta_alt, ruta_boy, punto, r, indices, prefijo, tabla_sigma = args
    lados = P.preparar_lados(pd.read_parquet(ruta_alt), pd.read_parquet(ruta_boy))
    return [replica(lados, r, punto, i, prefijo, tabla_sigma) for i in indices]


def tareas(ruta_alt, ruta_boy, punto: P.Punto, r, n: int, prefijo: str, tabla_sigma) -> list[tuple[str, tuple]]:
    out = []
    for a in range(0, n, TROZO):
        idx = list(range(a, min(a + TROZO, n)))
        nombre = f"{prefijo}__{punto.nombre}__{idx[0]:05d}-{idx[-1]:05d}"
        out.append((nombre, (str(ruta_alt), str(ruta_boy), punto, r, idx, prefijo, tabla_sigma)))
    return out


def punto_observado(si_obs: dict[str, float], rho_obs: dict[str, float]) -> tuple[P.Punto, dict]:
    """El punto peor entre el nominal y el observado, por clase de SI (el mayor de los lados de cada clase) y con el
    mayor ρ (un ρ mayor reduce la muestra efectiva). Calibración y ruido × 1: no se miden (§20.6)."""
    clase = P.CLASE_SI
    peor = {}
    for c in ("asimilado", "T0"):
        obs = [v for n, v in si_obs.items() if clase.get(n) == c and np.isfinite(v)]
        peor[c] = max([P.C.SI_NOMINAL[c], *obs])
    rhos = [v for v in rho_obs.values() if np.isfinite(v)]
    rho = max([P.C.RHO_NOMINAL, *rhos])
    nombre = f"observado_SI{peor['asimilado']:.3f}-{peor['T0']:.3f}_rho{rho:.2f}"
    return P.Punto(nombre, peor["asimilado"], peor["T0"], rho, 1.0), {
        "si_asimilado": peor["asimilado"],
        "si_T0": peor["T0"],
        "rho": rho,
    }
