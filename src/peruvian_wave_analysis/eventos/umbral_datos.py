"""u\\* por rama, ``eventos-umbral`` (contrato de la F7, §3.4 y §3.6; ADR-0037).

Con el IET\\* publicado por ``eventos-iet`` y los picos de cada rama (``ramas_picos``: pico de la rama, precisión
n.º 24 y borde E-29):

1. **Rejilla** de 20 umbrales geométrica en el número de picos (``umbral.rejilla``, la misma de la simulación) y
   **barrido**: en cada u_k, ajuste GPD, ξ̂ con su IC de perfil, σ\\* = σ̂ − ξ̂u con su IC de perfil, RL_T (T = 1, 10,
   50, 100) con el IC de perfil de RL100, el p de R2 (en los v_i) y el de R3.
2. **Regla R1b (ADR-0037):** u\\* es el umbral más bajo del tramo estable más alto del RL100. Con las mismas
   funciones que la simulación previa. «Sin resultado» de R1b en una rama: la fase se detiene (G6).
3. **Marcas (no cambian u\\*):**
   - punto sensible (ADR-0030, punto 4): el RL100 cambia > 2 % entre u\\* y alguno de sus vecinos;
   - deriva de ξ (§3.4, §5.2): pendiente de ξ̂ frente a log n_k en los u_k ≥ u\\* fuera del intervalo central del 90 %
     de su nulo (``simulacion/eleccion.json``, ``nulo_pendiente_xi``, agrupado; los de cada celda, descriptivos).
4. **Sensibilidad (ADR-0037, punto 6):** u\\* y RL100 de R2 y de R3 (B = 99, semilla principal
   ("f07", "datos", "r3", rama)); d_RL y Δu frente a R1b, ubicados en el intervalo [P2.5, P97.5] de
   ``sesgo-regla/referencia_sensibilidad.json``. Un «sin resultado» de R2 o R3 es «no comparable».
   **Descriptivo:** el u\\* de R3 con 5 semillas más.
5. **Tabla de picos** sobre u\\* de cada rama (§3.6).

E2 toma el u\\* de E0 en cada variante (P5); no se calcula aparte.

**Gate:** G4 (ajuste válido en u\\* y en la ventana de R1b), G5 (≥ 30 picos en u\\*), G6 (R1b con resultado en las
cuatro ramas).

Salida en ``processed/F07-eventos/umbral/marcona-200m/``: ``barrido_umbral.csv``, ``umbral.json``,
``picos_<rama>.csv`` y ``figuras/``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..clima import insumos as CI
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..extremos import perfil as PF
from . import iet as IE
from . import r2
from . import ramas as R
from . import ramas_picos as RP
from . import sesgo_regla as SR
from . import simulacion_previa as SP
from . import umbral as U

ETAPA = orquestador.Etapa("eventos-umbral", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
NOMBRE = CI.NOMBRE
REGLA = "R1b"
SENSIBILIDAD = ("R2", "R3")
SEMILLAS_R3_DESCRIPTIVAS = 5
TOL_RELEVANCIA = 0.02  # ADR-0030, punto 4
PERIODOS = U.PERIODOS
NIVEL = 0.95


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "umbral" / NOMBRE


def semilla_r3(rama: str, descriptiva: int | None = None) -> tuple[str, ...]:
    base = ("f07", "datos", "r3", rama)
    return base if descriptiva is None else (*base, "descriptiva", str(descriptiva))


def p_r3(picos: np.ndarray, rej: U.Rejilla, aj: Sequence[U.AjusteUmbral], semilla: tuple[str, ...]) -> np.ndarray:
    """p de R3 en cada u_k, con el mismo esquema que ``simulacion_previa.aplicar_reglas``."""
    p = []
    for k, (u, a) in enumerate(zip(rej.u, aj, strict=True)):
        if not a.valido:
            p.append(np.nan)
            continue
        exc = picos[picos > u] - u
        p.append(U.p_valor_r3(exc, a.ajuste, float(u), semillas.generador(*semilla, str(k))).p)
    return np.array(p, dtype=float)


def _ic(picos: np.ndarray, a: U.AjusteUmbral, anios: float) -> dict:
    out = {"xi_lo": np.nan, "xi_hi": np.nan, "sigma_mod_lo": np.nan, "sigma_mod_hi": np.nan, "rl100_lo": np.nan,
           "rl100_hi": np.nan, "causa_ic": ""}  # fmt: skip
    if not a.valido:
        out["causa_ic"] = "ajuste_no_valido"
        return out
    exc = picos[picos > a.u] - a.u
    causas = []
    px = PF.perfil_xi(exc, a.ajuste, NIVEL)
    ps = IE.perfil_sigma_mod(exc, a.u, a.ajuste, NIVEL)
    out |= {"xi_lo": px.lo, "xi_hi": px.hi, "sigma_mod_lo": ps.lo, "sigma_mod_hi": ps.hi}
    causas += [f"xi:{c}" for c in (px.causa_lo, px.causa_hi) if c]
    causas += [f"sigma_mod:{c}" for c in (ps.causa_lo, ps.causa_hi) if c]
    try:
        pr = PF.perfil_rl(exc, a.n, anios, 100.0, a.u, a.ajuste, NIVEL)
        out |= {"rl100_lo": pr.lo, "rl100_hi": pr.hi}
        causas += [f"rl100:{c}" for c in (pr.causa_lo, pr.causa_hi) if c]
    except ValueError as e:
        causas.append(f"rl100:{e}")
    out["causa_ic"] = ";".join(causas)
    return out


def relevancia(rl100: np.ndarray, i: int) -> dict:
    vec = [j for j in (i - 1, i + 1) if 0 <= j < rl100.size]
    cambios = {str(j): float(rl100[j] / rl100[i] - 1.0) for j in vec}
    m = max((abs(c) for c in cambios.values() if np.isfinite(c)), default=np.nan)
    return {"cambios_frente_a_vecinos": cambios, "maximo": m, "punto_sensible": bool(m > TOL_RELEVANCIA)}


def analizar_rama(
    rama: str, picos: np.ndarray, anios: float, referencia: dict, nulo: Sequence[float], con_ic: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Barrido, reglas, marcas y sensibilidad de una rama."""
    rej = U.rejilla(picos)
    aj = U.barrido(picos, rej, anios, PERIODOS)
    rl = np.array([a.rl_valor(100.0) for a in aj])
    xi = np.array([a.ajuste.xi if a.valido else np.nan for a in aj])
    res = {REGLA: U.regla_r1(rl, REGLA)}
    pr2 = r2.pruebas_r2(picos, r2.umbrales_r2(rej.u), [a.ajuste for a in aj][0::2])
    res["R2"] = r2.regla_r2(pr2)
    p3 = p_r3(picos, rej, aj, semilla_r3(rama))
    res["R3"] = U.regla_r3(p3)
    filas = []
    for i, (k, a) in enumerate(zip(rej.k, aj, strict=True)):
        f = {"rama": rama, "posicion": i, "k": int(k), "n_objetivo": int(rej.n_objetivo[i]), "n": int(rej.n_real[i]),
             "u_m": float(rej.u[i]), "lambda_por_anio": a.tasa_por_anio, "valido": bool(a.valido),
             "causa_ajuste": a.ajuste.causa or "", "xi": xi[i], "sigma": a.ajuste.sigma if a.valido else np.nan,
             "sigma_mod": a.sigma_mod}  # fmt: skip
        for T in PERIODOS:
            r = a.rl[T]
            f[f"rl{T:g}"] = r.valor if (a.valido and r.existe) else np.nan
            f[f"causa_rl{T:g}"] = "" if (a.valido and r.existe) else (r.causa or "")
        f |= _ic(picos, a, anios) if con_ic else {}
        f["p_r2"] = pr2[i // 2].p if (i % 2 == 0 and i // 2 < len(pr2)) else np.nan
        f["p_r3"] = p3[i]
        filas.append(f)
    tabla = pd.DataFrame(filas)
    r1b = res[REGLA]
    out: dict = {"rama": rama, "N0": int(picos.size), "anios": anios, "con_repetidos": rej.con_repetidos,
                 "k_repetidos": list(rej.k_repetidos)}  # fmt: skip
    for nombre, r in res.items():
        ok = r.indice is not None and np.isfinite(rl[r.indice])
        out[nombre] = {"estado": r.estado, "posicion": r.indice, "ventana": list(r.ventana) if r.ventana else None,
                       "u_m": float(rej.u[r.indice]) if r.indice is not None else None,
                       "n": int(rej.n_real[r.indice]) if r.indice is not None else None,
                       "rl100": float(rl[r.indice]) if ok else None, "detalle": r.detalle}  # fmt: skip
    if r1b.indice is None:
        return tabla, out
    i = int(r1b.indice)
    out["relevancia"] = relevancia(rl, i)
    pend = SP.pendiente_xi(xi, rej.n_real, i)
    out["deriva_xi"] = {"pendiente": pend, "intervalo_nulo_90": list(nulo),
                        "marca": bool(np.isfinite(pend) and not (nulo[0] <= pend <= nulo[1]))}  # fmt: skip
    sens = {}
    for nombre in SENSIBILIDAD:
        r, ref = res[nombre], referencia["comparadas"][nombre]
        if r.indice is None or not np.isfinite(rl[r.indice]) or not np.isfinite(rl[i]):
            sens[nombre] = {"comparable": False, "lectura": SR.NO_COMPARABLE}
            continue
        d_rl = float(rl[r.indice] / rl[i] - 1.0)
        du = float(rej.u[r.indice] - rej.u[i])
        sens[nombre] = {"comparable": True, "d_rl": d_rl, "delta_u_m": du,
                        "intervalo_d_rl": ref["d_rl"], "intervalo_delta_u_m": ref["delta_u_m"],
                        "lectura_d_rl": SR.ubicar(d_rl, ref["d_rl"]),
                        "lectura_delta_u": SR.ubicar(du, ref["delta_u_m"])}  # fmt: skip
    out["sensibilidad"] = sens
    desc = []
    for s in range(1, SEMILLAS_R3_DESCRIPTIVAS + 1):
        rr = U.regla_r3(p_r3(picos, rej, aj, semilla_r3(rama, s)))
        u_s = float(rej.u[rr.indice]) if rr.indice is not None else None
        desc.append({"semilla": s, "estado": rr.estado, "u_m": u_s})
    out["r3_semillas_descriptivas"] = desc
    j0, j1 = r1b.ventana
    out["g4"] = bool(all(aj[j].valido for j in range(j0, j1 + 1)))
    out["g5"] = bool(rej.n_real[i] >= U.N_MIN_ARRIBA)
    return tabla, out


def tabla_picos(p: pd.DataFrame, u: float) -> pd.DataFrame:
    sobre = p[p["hs"] > u].copy()
    sobre["exceso"] = sobre["hs"] - u
    cols = ["id_evento", "id_racha", "t_pico", "t_inicio", "t_fin", "n_excedencias", "hs", "exceso", "tp", "vmdr",
            "vped", "particion_dominante", "estrato", "toca_hueco", "dividido_por_bajada", "pico_movido",
            "empate_resuelto", "dudoso_f3", "revision_visual_f6"]  # fmt: skip
    return sobre[cols]


def figuras(tabla: pd.DataFrame, res: dict, carpeta: Path) -> list[str]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    carpeta.mkdir(parents=True, exist_ok=True)
    hechas = []
    for rama, g in tabla.groupby("rama", sort=False):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (col, et) in zip(axs, (("xi", "ξ̂"), ("sigma_mod", "σ* = σ̂ − ξ̂u (m)"), ("rl100", "RL100 (m)")),
                                 strict=True):  # fmt: skip
            ax.plot(g["u_m"], g[col], marker="o", ms=3)
            if f"{col}_lo" in g:
                ax.fill_between(g["u_m"], g[f"{col}_lo"], g[f"{col}_hi"], alpha=0.15)
            u = res[rama].get(REGLA, {}).get("u_m")
            if u is not None:
                ax.axvline(u, color="k", ls="--", lw=0.8)
            ax.set_xlabel("Umbral u (m)")
            ax.set_ylabel(et)
        fig.suptitle(f"{rama}: estabilidad frente al umbral, IC de perfil del 95 % (línea: u* de R1b)")
        fig.tight_layout()
        nombre = f"estabilidad_{rama}.png"
        fig.savefig(carpeta / nombre, dpi=130)
        plt.close(fig)
        hechas.append(nombre)
    return hechas


def ejecutar(raiz_datos=None) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = RP.rutas_insumos(raiz)
    d_iet = IE.ruta_salida(raiz)
    d_sim, d_ses = SP.SA.ruta_simulacion(raiz), SR.ruta_salida(raiz)
    for d in (d_iet, d_sim, d_ses):
        etapa.exigir_publicada(d)
    ins |= {"iet": d_iet / "iet.json", "eleccion": d_sim / "eleccion.json",
            "referencia_sensibilidad": d_ses / "referencia_sensibilidad.json"}  # fmt: skip
    etapa.exigir_insumos(ins)
    datos = CI.cargar(raiz, ins, NOMBRE)
    p95 = float(json_estricto.leer(ins["resumen_candidatos"])["p95_hs_m"])
    iet = float(json_estricto.leer(ins["iet"])["iet_estrella_h"])
    referencia = json_estricto.leer(ins["referencia_sensibilidad"])
    nulo = json_estricto.leer(ins["eleccion"])["nulo_pendiente_xi"]["intervalo"]
    cand, rev = pd.read_csv(ins["candidatos"]), pd.read_csv(ins["declustering_dudoso"])
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        conteos, picos, variantes, bajo = RP.ramas_picos(datos, p95, p95, cand, rev, iets=(iet,))
        tablas, res = [], {}
        for rama in R.RAMAS:
            pr = picos[picos["rama"] == rama]
            anios = float(conteos.loc[conteos["rama"] == rama, "anios_exposicion"].iloc[0])
            print(f"  {rama}: {len(pr)} picos", flush=True)
            t, r = analizar_rama(rama, pr["hs"].to_numpy(dtype=float), anios, referencia, nulo)
            tablas.append(t)
            res[rama] = r
            if r[REGLA]["u_m"] is not None:
                tabla_picos(pr, r[REGLA]["u_m"]).to_csv(st / f"picos_{rama}.csv", index=False, float_format="%.6g",
                                                        lineterminator="\n", date_format="%Y-%m-%d %H:%M")  # fmt: skip
        tabla = pd.concat(tablas, ignore_index=True)
        tabla.to_csv(st / "barrido_umbral.csv", index=False, float_format="%.6g", lineterminator="\n")
        figs = figuras(tabla, res, st / "figuras")
        motivos = []
        for rama, r in res.items():
            if r[REGLA]["posicion"] is None:
                motivos.append(f"G6: R1b sin resultado en {rama}: la fase se detiene (§3.4)")
                continue
            if not r["g4"]:
                motivos.append(f"G4: ajuste no válido en la ventana de R1b en {rama}")
            if not r["g5"]:
                motivos.append(f"G5: menos de 30 picos en u* en {rama}")
        resumen = {
            "iet_estrella_h": iet,
            "regla": REGLA,
            "adr": "ADR-0037",
            "ramas": res,
            "e2": "E2 toma el u* de E0 en cada variante (P5).",
            "g8": {"picos_bajo_p95_f3": bajo},
            "u_id_en_variantes": variantes,
            "figuras": figs,
        }
        json_estricto.escribir(st / "umbral.json", resumen)
        gate = etapa.Gate(not motivos and not bajo, {"iet_estrella_h": iet}, motivos)
        return etapa.publicar(st, salida, gate, raices, ins, raiz, semillas.semilla("f07", "datos", "r3"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-umbral", description="u* por rama de la F7 (§3.4, ADR-0037)")
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
