"""Tablas descriptivas de la F6 (contrato §3.1–§3.4, §3.6 y §3.7). No deciden nada.

Solo registros con ``valido`` (en la serie de la F3 los no válidos ya son NaN). Las ramas por cuantiles cambian solo
Hs: Tp, dirección y particiones son las de E0.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import clase_fisica as CF
from . import descriptivos as DS
from . import picos as P

ESCALON = pd.Timestamp("2016-03-01")
VENTANA_ANIOS = 10
VENTANAS_DESDE, VENTANAS_HASTA = 1993, 2015
ANCHO_TP_ESTADOS, MIN_TP_ESTADOS = 0.25, 30
ANCHO_TP_PICOS, MIN_TP_PICOS = 0.5, 15
RANGOS_HS = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf)
TP_INCOHERENCIA = {"swell": 8.0, "mar de viento": 10.0}
DOMINANTE = {"WW": "VTM01_WW", "SW1": "VTM01_SW1", "SW2": "VTM01_SW2"}
DIR_PARTICION = {"WW": "VMDR_WW", "SW1": "VMDR_SW1", "SW2": "VMDR_SW2"}


def validos(serie: pd.DataFrame) -> pd.DataFrame:
    return serie[serie["valido"].astype(bool)] if "valido" in serie else serie.dropna(subset=["VHM0"])


# --- §3.1 --------------------------------------------------------------------------------------------------


def clima_mensual(serie: pd.DataFrame, ramas: Mapping[str, pd.Series]) -> pd.DataFrame:
    partes = []
    for rama, hs in ramas.items():
        df = serie.loc[hs.index, ["VTPK", "VPED"]].copy()
        df["VHM0"] = hs
        df = df.dropna(subset=["VHM0"])
        partes.append(DS.clima_mensual(df).assign(rama=rama))
    t = pd.concat(partes, ignore_index=True)
    return t[["rama", *[c for c in t.columns if c != "rama"]]]


def rosas(serie: pd.DataFrame, u99: float) -> pd.DataFrame:
    v = validos(serie)
    a = DS.rosa(v["VPED"]).assign(poblacion="todos los estados")
    b = DS.rosa(v.loc[v["VHM0"] >= u99, "VPED"]).assign(poblacion="Hs ≥ P99")
    return pd.concat([a, b], ignore_index=True)


# --- §3.2 --------------------------------------------------------------------------------------------------


def clase_fisica(serie: pd.DataFrame) -> pd.DataFrame:
    v = validos(serie)
    cf = CF.clase_fisica(v, P.ausentes(v))
    if "marca_discrepancia_energia" in v:
        cf["discrepancia_energia"] = v["marca_discrepancia_energia"].astype(bool)
    cf["dominante"] = cf[[f"r_{p}" for p in CF.PARTICIONES]].idxmax(axis=1).str[2:].where(cf["r_WW"].notna())
    return cf


def comprobacion_por_clase(serie: pd.DataFrame, cf: pd.DataFrame) -> dict:
    """Mediana y P90 de Tp, del período de la partición dominante y de |dirección dominante − VPED| por clase."""
    v = validos(serie)
    out = {}
    for clase in CF.CLASES:
        idx = cf.index[cf["clase"] == clase]
        if not len(idx):
            out[clase] = {"n": 0}
            continue
        d = v.loc[idx]
        dom = cf.loc[idx, "dominante"]
        per = pd.Series(np.nan, index=idx)
        dif = pd.Series(np.nan, index=idx)
        for p, col in DOMINANTE.items():
            m = dom == p
            if col in d:
                per[m] = d.loc[m, col]
            if DIR_PARTICION[p] in d:
                dif[m] = (d.loc[m, DIR_PARTICION[p]] - d.loc[m, "VPED"] + 180) % 360 - 180
        fila = {"n": int(len(idx))}
        for nombre, x in (("tp", d["VTPK"]), ("periodo_dominante", per), ("dif_direccion_abs", dif.abs())):
            x = x.dropna()
            fila[nombre] = {"mediana": float(x.median()), "p90": float(x.quantile(0.9))} if len(x) else None
        out[clase] = fila
    medianas = {c: (out[c]["tp"] or {}).get("mediana", np.nan) if out[c].get("n") else np.nan for c in CF.CLASES}
    return {"por_clase": out, "incoherencia": CF.incoherencia_fisica(medianas)}


def swell_secundario(serie: pd.DataFrame) -> dict:
    v = validos(serie)
    presente = v["VHM0_SW2"].fillna(0) > 0
    if "marca_particion_ausente_SW2" in v:
        presente &= ~v["marca_particion_ausente_SW2"].astype(bool)
    dif = ((v.loc[presente, "VMDR_SW2"] - v.loc[presente, "VMDR_SW1"] + 180) % 360 - 180).abs() \
        if {"VMDR_SW1", "VMDR_SW2"} <= set(v.columns) else pd.Series(dtype=float)  # fmt: skip
    dm, disp = DS.media_circular(v.loc[presente, "VMDR_SW2"]) if "VMDR_SW2" in v else (np.nan, np.nan)
    return {"fraccion_con_sw2": float(presente.mean()), "direccion_media_sw2": dm, "dispersion_sw2": disp,
            "dif_sw2_sw1_mediana": float(dif.median()) if len(dif) else float("nan")}  # fmt: skip


def _fracciones(clases: pd.Series) -> dict:
    n = len(clases)
    return {"n": n} | {c: float((clases == c).mean()) if n else float("nan") for c in (*CF.CLASES, CF.SIN_CLASE)}


def dominancia(serie: pd.DataFrame, cf: pd.DataFrame, picos: pd.DataFrame) -> pd.DataFrame:
    v = validos(serie)
    filas = []
    for corte, col in (("1/3–2/3", "clase"), ("0.2–0.8", "clase_sens")):
        c = cf[col]
        base = {"cortes": corte}
        for m in range(1, 13):
            filas.append(base | {"grupo": "mes", "valor": str(m)} | _fracciones(c[c.index.month == m]))
        rango = pd.cut(v["VHM0"], RANGOS_HS, right=False)
        for r, g in c.groupby(rango.reindex(c.index), observed=True):
            filas.append(base | {"grupo": "rango_hs", "valor": str(r)} | _fracciones(g))
        if len(picos) and col in ("clase",):
            filas.append(base | {"grupo": "picos_provisionales", "valor": "todos"} | _fracciones(picos["clase"]))
        filas.append(base | {"grupo": "periodo", "valor": "antes de 2016-03"} | _fracciones(c[c.index < ESCALON]))
        filas.append(base | {"grupo": "periodo", "valor": "desde 2016-03"} | _fracciones(c[c.index >= ESCALON]))
        for a in range(VENTANAS_DESDE, VENTANAS_HASTA - VENTANA_ANIOS + 2):
            sel = (c.index.year >= a) & (c.index.year < a + VENTANA_ANIOS)
            filas.append(base | {"grupo": "ventana_10_anios", "valor": f"{a}–{a + VENTANA_ANIOS - 1}"}
                         | _fracciones(c[sel]))  # fmt: skip
    return pd.DataFrame(filas)


# --- §3.3 y §3.4 ---------------------------------------------------------------------------------------------


def picos_tabla(pk: pd.DataFrame) -> pd.DataFrame:
    cols = ["Hs", "exceso", "VTPK", "VPED", "clase", "mes", "estrato_B", "sector", "t_inicio", "t_fin",
            "n_excedencias", "toca_hueco", "marca_tp_saturado", "dudoso_f3"]  # fmt: skip
    t = pk[[c for c in cols if c in pk.columns]].copy()
    t.index.name = "t_pico"
    return t.reset_index()


def origen_mayores(pk: pd.DataFrame, serie: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    top = pk.nlargest(n, "Hs").copy()
    d = serie.loc[top.index]
    cf = CF.clase_fisica(d, P.ausentes(d))
    r = cf[[f"r_{p}" for p in CF.PARTICIONES]]
    dom = r.idxmax(axis=1).str[2:]
    top["r_WW"] = cf["r_WW"]
    top["particion_dominante"] = dom
    top["periodo_dominante"] = [d.at[t, DOMINANTE[p]] if DOMINANTE[p] in d else np.nan for t, p in dom.items()]
    top["direccion_dominante"] = [d.at[t, DIR_PARTICION[p]] if DIR_PARTICION[p] in d else np.nan
                                  for t, p in dom.items()]  # fmt: skip
    top["fenomeno"] = [_fenomeno(c, p, di) for c, p, di in zip(top["clase"], dom, top["direccion_dominante"],
                                                                 strict=True)]  # fmt: skip
    top.insert(0, "orden", np.arange(1, len(top) + 1))
    top.index.name = "t_pico"
    cols = ["orden", "Hs", "VTPK", "VPED", "clase", "r_WW", "particion_dominante", "periodo_dominante",
            "direccion_dominante", "mes", "dudoso_f3", "fenomeno", "t_inicio", "t_fin"]  # fmt: skip
    return top[[c for c in cols if c in top.columns]].reset_index()


def _fenomeno(clase, particion, direccion) -> str:
    """Solo lo que respaldan las particiones (§3.3); el sistema meteorológico no se nombra."""
    if clase != "swell" or not np.isfinite(direccion):
        return f"{clase} (partición {particion}); sin atribución"
    sector = "sur" if 135 <= direccion % 360 < 247.5 else ("oeste" if 247.5 <= direccion % 360 < 315 else "otro")
    return f"swell del {sector} (partición {particion}); hipótesis: sistema lejano, sin datos meteorológicos"


def pares_cercanos(pk: pd.DataFrame, rachas: pd.DataFrame, sep_h: float, n: int = 20) -> pd.DataFrame:
    """Los n pares consecutivos con hueco > separación más cercano a ella (casi fusionados) y los n eventos con ≥ 2
    rachas cuyo hueco interno más largo está más cerca de la separación (casi partidos). Hueco en horas."""
    p = pk.sort_index()
    filas = []
    if len(p) > 1:
        hueco = (p["t_inicio"].iloc[1:].to_numpy() - p["t_fin"].iloc[:-1].to_numpy()) / np.timedelta64(1, "h")
        d = pd.DataFrame({"t_pico": p.index[:-1], "t_pico_siguiente": p.index[1:], "hueco_h": hueco})
        d = d[d["hueco_h"] > sep_h].nsmallest(n, "hueco_h")
        filas.append(d.assign(tipo="casi fusionados"))
    if len(rachas):
        internos = []
        for t, ev in p.iterrows():
            r = rachas[(rachas["inicio"] >= ev["t_inicio"]) & (rachas["fin"] <= ev["t_fin"])]
            if len(r) >= 2:
                h = r["hueco_siguiente_h"].iloc[:-1].max()
                internos.append({"t_pico": t, "t_pico_siguiente": pd.NaT, "hueco_h": float(h),
                                 "n_rachas": int(len(r))})  # fmt: skip
        if internos:
            d = pd.DataFrame(internos)
            d = d.iloc[np.argsort(np.abs(d["hueco_h"].to_numpy() - sep_h), kind="stable")[:n]]
            filas.append(d.assign(tipo="casi partidos"))
    if not filas:
        return pd.DataFrame(columns=["tipo", "t_pico", "t_pico_siguiente", "hueco_h", "n_rachas"])
    out = pd.concat(filas, ignore_index=True)
    return out[["tipo", *[c for c in out.columns if c != "tipo"]]]


# --- §3.6 y §3.7 ---------------------------------------------------------------------------------------------


def tp_condicional(serie: pd.DataFrame, pk: pd.DataFrame) -> pd.DataFrame:
    v = validos(serie)
    cens = v["marca_tp_saturado"].astype(bool) if "marca_tp_saturado" in v else pd.Series(False, index=v.index)
    a = DS.tp_condicional(v["VHM0"], v["VTPK"], cens, ANCHO_TP_ESTADOS, MIN_TP_ESTADOS).assign(poblacion="estados")
    cp = pk["marca_tp_saturado"].astype(bool) if "marca_tp_saturado" in pk else pd.Series(False, index=pk.index)
    b = DS.tp_condicional(pk["Hs"], pk["VTPK"], cp, ANCHO_TP_PICOS, MIN_TP_PICOS).assign(poblacion="picos")
    t = pd.concat([a, b], ignore_index=True)
    return t[["poblacion", *[c for c in t.columns if c != "poblacion"]]]


def tp_discretos(serie: pd.DataFrame, u99: float) -> list[float]:
    v = validos(serie)
    return sorted(float(x) for x in np.unique(v.loc[v["VHM0"] >= u99, "VTPK"].dropna()))


def peralte(serie: pd.DataFrame, u99: float) -> tuple[pd.DataFrame, dict]:
    v = validos(serie).dropna(subset=["VHM0", "VTPK"])
    sp = DS.peralte(v["VHM0"], v["VTPK"])
    lim = DS.limite_peralte_dnv(v["VTPK"])
    q = np.quantile(sp, [0.5, 0.9, 0.99, 0.999, 1.0])
    alto = v["VHM0"].to_numpy() >= u99
    tp10 = float(np.quantile(v.loc[alto, "VTPK"], 0.1)) if alto.any() else float("nan")
    sobre = sp > lim
    tabla = v.loc[sobre | (alto & (v["VTPK"].to_numpy() <= tp10)), ["VHM0", "VTPK"]].copy()
    tabla["peralte"] = sp[sobre | (alto & (v["VTPK"].to_numpy() <= tp10))]
    tabla["limite_dnv"] = lim[sobre | (alto & (v["VTPK"].to_numpy() <= tp10))]
    tabla["sobre_limite"] = tabla["peralte"] > tabla["limite_dnv"]
    tabla["hs_alto_tp_bajo"] = (tabla["VHM0"] >= u99) & (tabla["VTPK"] <= tp10)
    tabla.index.name = "tiempo"
    resumen = {"n": int(sp.size), "p50": q[0], "p90": q[1], "p99": q[2], "p999": q[3], "max": q[4],
               "sobre_limite_dnv": int(sobre.sum()), "tp_p10_con_hs_alto": tp10,
               "hs_alto_tp_bajo": int(tabla["hs_alto_tp_bajo"].sum()),
               "nota": "Límite de DNV-RP-C205 (2017), §3.5.4, con datos de la plataforma noruega."}  # fmt: skip
    return tabla.reset_index(), resumen
