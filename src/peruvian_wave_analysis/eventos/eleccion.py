"""Elección de la regla de umbral con la simulación previa (contrato de la F7, §5.4; ADR-0030, ADR-0032, ADR-0033).

Trabaja sobre la tabla larga de réplicas de ``simulacion_previa`` (una fila por réplica y regla, con ``error_log`` =
log RL100(regla) − log RL100(verdadero)). No genera nada: solo agrega, así que se prueba con casos dados a mano.

1. **Métrica (E-24):** en cada celda, RECM(regla) / RECM(k = 10); promedio con igual peso sobre las 18 celdas de A,
   B+, B− y C. El control no entra. Una réplica «sin resultado» ya viene con el error de k = 10 (o 11, o 9).
2. **Restricciones:**
   - «sin resultado» ≤ 10 % en cada una de las 18 celdas;
   - B−: sesgo(regla) − sesgo(oráculo) ≥ log 0.98 en cada celda, sobre las réplicas con oráculo;
   - C: sesgo(regla) ≥ log 0.98; si k = 10 incumple esa cota en la celda, rige sesgo(regla) ≥ sesgo(k = 10) + log 0.98,
     con la marca «restricción relativa»;
   - control de potencia: si k = 1 cumple la de B− en una celda, se marca «B− no discrimina» (no cambia nada).
3. **Filtro del control:** la de B− frente al oráculo, en las 4 celdas del control; si k = 1 la cumple, «control no
   discrimina».
4. **Elección:** entre las que cumplen restricciones y filtro, la de menor métrica. R1 entra con la mejor de sus dos
   variantes que cumpla (si ninguna cumple, con la de menor métrica, que queda fuera). Empatan con la mejor las que
   quedan a menos de 2 errores típicos de ella, con el error de la diferencia por bootstrap de las réplicas dentro de
   cada celda (los mismos índices para todas las reglas) propagado al promedio. Entre las empatadas: R1, R3, R2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..comun import semillas
from . import umbral as U

LOG98 = float(np.log(0.98))
MAX_SIN_RESULTADO = 0.10
REGLAS = ("R1a", "R1b", "R2", "R3")
PREFERENCIA = ("R1", "R3", "R2")
B_BOOT = 2000
Z_EMPATE = 2.0
TERCIOS_K0 = (4.0, 4.0 + 10.0 / 3.0, 4.0 + 20.0 / 3.0, 14.0)
CASOS_METRICA = ("A", "B+", "B-", "C")


def _recm(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if x.size else np.nan


def errores(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla ancha: índice (sigma, celda, replica), una columna de error por regla o referencia."""
    return df.pivot(index=["sigma", "celda", "replica"], columns="regla", values="error_log")


def resumen_celdas(df: pd.DataFrame) -> pd.DataFrame:
    """Por (sigma, celda, regla): RECM, razón frente a k = 10, sesgos, tasas y sesgo por tercio de k₀."""
    E = errores(df)
    filas = []
    for (sigma, celda), g in df.groupby(["sigma", "celda"], sort=False):
        e = E.loc[(sigma, celda)]
        info = g.iloc[0]
        recm_k10 = _recm(e["k10"])
        for regla, h in g.groupby("regla", sort=False):
            x = e[regla].to_numpy(float)
            fila = {
                "sigma": sigma,
                "celda": celda,
                "caso": info["caso"],
                "forma": info["forma"],
                "N0": int(info["N0"]),
                "regla": regla,
                "replicas": int(len(h)),
                "error_no_finito": int(np.sum(~np.isfinite(x))),
                "recm": _recm(x),
                "razon_k10": _recm(x) / recm_k10,
                "sesgo": float(np.nanmean(x)),
                "desviacion": float(np.nanstd(x, ddof=1)),
                "tasa_sin_resultado": float(np.mean(h["estado"] == U.SIN_RESULTADO)),
                "tasa_no_discrimino": float(np.mean(h["estado"] == U.NO_DISCRIMINO)),
                "sustituidas": int(np.sum(h["sustituto"].fillna("") != "")),
            }
            if "oraculo" in e.columns and np.any(np.isfinite(e["oraculo"])):
                o = e["oraculo"].to_numpy(float)
                ok = np.isfinite(o) & np.isfinite(x)
                fila["sesgo_frente_oraculo"] = float(np.mean(x[ok] - o[ok]))
                fila["replicas_con_oraculo"] = int(ok.sum())
                hh = h.set_index("replica").loc[e.index]
                fila["frac_bajo_u0"] = float(np.mean(hh["u_usado"].to_numpy(float) < hh["u0"].to_numpy(float)))
                k0 = hh["k0"].to_numpy(float)
                for t in range(3):
                    en = (k0 >= TERCIOS_K0[t]) & (k0 <= TERCIOS_K0[t + 1]) & np.isfinite(x)
                    fila[f"sesgo_tercio_k0_{t + 1}"] = float(np.mean(x[en])) if en.any() else np.nan
            filas.append(fila)
    return pd.DataFrame(filas)


def _de(res: pd.DataFrame, regla: str, casos: Sequence[str] | None = None) -> pd.DataFrame:
    r = res[res["regla"] == regla]
    return r[r["caso"].isin(casos)] if casos is not None else r


def metrica(res: pd.DataFrame, regla: str) -> float:
    return float(_de(res, regla, CASOS_METRICA)["razon_k10"].mean())


def restricciones(res: pd.DataFrame, regla: str) -> dict:
    """Restricciones de §5.4, punto 2, y filtro del control (punto 3), con las celdas que fallan."""
    sr = _de(res, regla, CASOS_METRICA)
    falla_sr = sr.loc[sr["tasa_sin_resultado"] > MAX_SIN_RESULTADO, "celda"].tolist()
    bm = _de(res, regla, ["B-"])
    falla_bm = bm.loc[~(bm["sesgo_frente_oraculo"] >= LOG98), "celda"].tolist()
    c = _de(res, regla, ["C"]).set_index("celda")
    k10 = _de(res, "k10", ["C"]).set_index("celda")
    falla_c, relativas = [], []
    for celda, fila in c.iterrows():
        cota = LOG98
        if k10.loc[celda, "sesgo"] < LOG98:
            cota = float(k10.loc[celda, "sesgo"]) + LOG98
            relativas.append(celda)
        if not fila["sesgo"] >= cota:
            falla_c.append(celda)
    ct = _de(res, regla, ["control"])
    falla_ct = ct.loc[~(ct["sesgo_frente_oraculo"] >= LOG98), "celda"].tolist()
    cumple = not (falla_sr or falla_bm or falla_c)
    return {
        "cumple_restricciones": cumple,
        "cumple_control": not falla_ct and len(ct) > 0,
        "elegible": cumple and not falla_ct and len(ct) > 0,
        "celdas_sin_resultado_mayor_10": falla_sr,
        "celdas_b_menos_fallan": falla_bm,
        "celdas_c_fallan": falla_c,
        "celdas_c_restriccion_relativa": relativas,
        "celdas_control_fallan": falla_ct,
    }


def marcas_potencia(res: pd.DataFrame) -> dict:
    """k = 1 cumple la restricción de B− (o del control) frente al oráculo → la celda no discrimina."""
    k1 = res[res["regla"] == "k1"]
    bm = k1[k1["caso"] == "B-"]
    ct = k1[k1["caso"] == "control"]
    return {
        "b_menos_no_discrimina": bm.loc[bm["sesgo_frente_oraculo"] >= LOG98, "celda"].tolist(),
        "control_no_discrimina": ct.loc[ct["sesgo_frente_oraculo"] >= LOG98, "celda"].tolist(),
    }


def metricas_bootstrap(df: pd.DataFrame, reglas: Sequence[str], B: int = B_BOOT, sigma=None) -> dict[str, np.ndarray]:
    """Métrica de cada regla en B remuestreos de las réplicas dentro de cada celda (mismos índices para todas)."""
    E = errores(df)
    info = df.drop_duplicates(["sigma", "celda"]).set_index(["sigma", "celda"])["caso"]
    out = {r: np.zeros(B) for r in reglas}
    n_celdas = 0
    for (s, celda), caso in info.items():
        if caso not in CASOS_METRICA or (sigma is not None and s != sigma):
            continue
        e = E.loc[(s, celda)]
        n = len(e)
        idx = semillas.generador("f07", "eleccion", "bootstrap", f"{s:g}", celda).integers(0, n, size=(B, n))

        def recm_boot(x, idx=idx):
            return np.sqrt(np.nanmean(x[idx] ** 2, axis=1))

        base = recm_boot(e["k10"].to_numpy(float))
        for r in reglas:
            out[r] += recm_boot(e[r].to_numpy(float)) / base
        n_celdas += 1
    return {r: v / n_celdas for r, v in out.items()}


def _grupo(regla: str) -> str:
    return "R1" if regla.startswith("R1") else regla


def elegir(df: pd.DataFrame, res: pd.DataFrame | None = None, B: int = B_BOOT) -> dict:
    """Aplica §5.4, puntos 1 a 4, sobre las réplicas de un σ."""
    if res is None:
        res = resumen_celdas(df)
    sigmas = df["sigma"].unique()
    if sigmas.size != 1:
        raise ValueError("elegir() trabaja sobre un solo σ")
    reglas = [r for r in REGLAS if r in set(df["regla"])]
    por_regla = {r: {"metrica": metrica(res, r)} | restricciones(res, r) for r in reglas}

    # R1: la mejor de sus variantes que cumpla; si ninguna cumple, la de menor métrica (que queda fuera)
    r1 = [r for r in reglas if r.startswith("R1")]
    r1_elegibles = [r for r in r1 if por_regla[r]["elegible"]]
    variante_r1 = min(r1_elegibles or r1, key=lambda r: por_regla[r]["metrica"]) if r1 else None
    candidatas = [r for r in reglas if not r.startswith("R1")] + ([variante_r1] if variante_r1 else [])

    solo_restr = [r for r in candidatas if por_regla[r]["cumple_restricciones"]]
    mejor_sin_filtro = min(solo_restr, key=lambda r: por_regla[r]["metrica"]) if solo_restr else None
    elegibles = [r for r in candidatas if por_regla[r]["elegible"]]
    out = {
        "sigma": float(sigmas[0]),
        "por_regla": por_regla,
        "variante_r1": variante_r1,
        "mejor_solo_con_restricciones": mejor_sin_filtro,
        "marcas": marcas_potencia(res),
        "elegida": None,
        "empatadas": [],
        "error_tipico_diferencia": {},
    }
    if not elegibles:
        out["motivo"] = "Ninguna regla cumple las restricciones y el filtro del control: la F7 se detiene (§5.4)."
        return out
    mejor = min(elegibles, key=lambda r: por_regla[r]["metrica"])
    boot = metricas_bootstrap(df, elegibles, B)
    empatadas = [mejor]
    for r in elegibles:
        if r == mejor:
            continue
        dif = por_regla[r]["metrica"] - por_regla[mejor]["metrica"]
        et = float(np.std(boot[r] - boot[mejor], ddof=1))
        out["error_tipico_diferencia"][r] = {"diferencia": dif, "error_tipico": et}
        if dif < Z_EMPATE * et:
            empatadas.append(r)
    elegida = min(empatadas, key=lambda r: PREFERENCIA.index(_grupo(r)))
    out |= {
        "mejor_metrica": mejor,
        "empatadas": empatadas,
        "elegida": elegida,
        "filtro_cambio_la_eleccion": mejor_sin_filtro is not None and mejor_sin_filtro != mejor,
    }
    return out


def nulo_pendiente(df: pd.DataFrame, regla: str, nivel: float = 0.90) -> dict:
    """§5.2: intervalo central de la pendiente de ξ̂ frente a log n_k sobre u*, en el caso A con la regla elegida."""
    a = df[(df["caso"] == "A") & (df["regla"] == regla)]
    x = a["pendiente_xi"].to_numpy(float)
    ok = np.isfinite(x)
    q = (1 - nivel) / 2
    por = {
        str(c): [float(v) for v in np.nanquantile(g["pendiente_xi"].to_numpy(float), [q, 1 - q])]
        for c, g in a.groupby("celda", sort=False)
    }
    return {
        "regla": regla,
        "nivel": nivel,
        "intervalo": [float(v) for v in np.quantile(x[ok], [q, 1 - q])] if ok.any() else None,
        "replicas": int(x.size),
        "sin_pendiente": int((~ok).sum()),
        "por_celda": por,
    }


def para_la_f9(res: pd.DataFrame, regla: str) -> list[dict]:
    """§5.4, punto 6: sesgo y dispersión de la regla elegida frente a k = 10, y el sesgo del oráculo."""
    cols = ["celda", "caso", "sesgo", "desviacion", "recm"]
    r = _de(res, regla)[cols].set_index("celda")
    k = _de(res, "k10")[cols].set_index("celda")
    o = _de(res, "oraculo")[cols].set_index("celda") if "oraculo" in set(res["regla"]) else None
    filas = []
    for celda, f in r.iterrows():
        d = {
            "celda": celda,
            "caso": f["caso"],
            "sesgo_regla": f["sesgo"],
            "desviacion_regla": f["desviacion"],
            "sesgo_k10": k.loc[celda, "sesgo"],
            "desviacion_k10": k.loc[celda, "desviacion"],
        }
        if o is not None and celda in o.index:
            d["sesgo_oraculo"] = o.loc[celda, "sesgo"]
        filas.append(d)
    return filas
