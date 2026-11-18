"""Armado de las colocaciones geométricas y del escalón del modelo por fuente (etapa 2, parte A; contrato v6).

Solo modelo y geometría: ningún valor observado (G4). Lo usa la etapa ``validacion-potencia``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import manifiesto
from ..sitio import costa as costa_mod
from ..sitio import observaciones
from ..sitio.ficha import _gebco, _profundidad
from . import colocacion as K
from . import config as C
from . import escalon as E
from . import ndbc

GEBCO = "batimetria/gebco/gebco_2026_n-12.5_s-17.5_w-78.5_e-73.0.nc"  # el recorte de la F1 que cubre la región


def rutas_altimetria(raw: Path) -> dict[str, Path]:
    """Teselas de las misiones de la F5 registradas en el manifiesto de raw (clave = ruta relativa)."""
    ent = manifiesto.leer(raw)
    return {
        r: raw / r
        for r in sorted(ent)
        if r.startswith("altimetria/ribal_young/") and r.split("/")[2] in C.MISIONES and r.endswith(".nc")
    }


def rutas_ndbc(raw: Path) -> dict[str, Path]:
    ent = manifiesto.leer(raw)
    return {r: raw / r for r in sorted(ent) if r.startswith("boyas/ndbc/")}


def efectos_nodo(pruebas_cambio: dict) -> dict[str, float]:
    """Efecto del nodo de cada familia, log(1 + δ_rel corregido) del lado que fija §4.4."""
    out = {}
    for fam, (familia_f4, contraste, lado) in C.EFECTO_NODO_RUTA.items():
        d = pruebas_cambio[familia_f4][contraste]["delta_rel_corregido"]["por_lado"][lado]["delta"]
        out[fam] = float(np.log1p(abs(float(d))))
    return out


def colocaciones_altimetria(
    archivos: list[Path],
    ruta_gebco: Path,
    tierra,
    sectores: pd.DataFrame,
    region: pd.DataFrame,
    marcas: pd.DataFrame,
    nodo_hs: pd.Series,
) -> tuple[pd.DataFrame, list[dict]]:
    """Superobservaciones (principal ≥ 50 km y sensibilidad ≥ 25 km) con el modelo en la celda y en el nodo."""
    med = observaciones.leer(archivos)  # tiempo, posición, bandera, profundidad: no lee Hs
    cst = observaciones.Costa.desde_tierra(tierra, *C.NODO)
    prof = _profundidad(_gebco(ruta_gebco))
    d, qc = K.filtrar_altimetria(med, cst, prof, sectores, region.columns)
    so = pd.concat([K.superobservaciones(d, C.COSTA_MIN_KM), K.superobservaciones(d, C.COSTA_MIN_SENS_KM)])
    so = so.reset_index(drop=True)
    qc.append(
        {
            "filtro": "superobservaciones ≥ 3 puntos (≥ 50 km)",
            "quedan": int((so["costa_min_km"] == C.COSTA_MIN_KM).sum()),
        }
    )
    so["hs_mod"] = K.interpolar(region, so["celda"], so["tiempo"])
    so["hs_nodo"] = K.interpolar(nodo_hs.to_frame("nodo"), np.full(len(so), "nodo"), so["tiempo"])
    so["marca_detector"] = K.marca_en(marcas, so["celda"], so["tiempo"])
    sin_modelo = ~np.isfinite(so["hs_mod"]) | ~np.isfinite(so["hs_nodo"])
    qc.append(
        {
            "filtro": "con modelo en la celda y en el nodo (≥ 50 km)",
            "quedan": int(((so["costa_min_km"] == C.COSTA_MIN_KM) & ~sin_modelo).sum()),
        }
    )
    so = so[~sin_modelo].reset_index(drop=True)
    so["unidad"] = so["pasada"]
    return so, qc


def colocaciones_boyas(archivos: dict[str, Path], modelo_boyas: pd.DataFrame, nodo_hs: pd.Series) -> pd.DataFrame:
    """Pasos de 3 h de cada boya con al menos 2 registros de WVHT presentes, con el modelo en su celda y en el nodo."""
    partes = []
    for b in ndbc.BOYAS:
        rs = [p for r, p in archivos.items() if r.split("/")[2] == b.estacion]
        if not rs:
            continue
        pres = pd.concat([ndbc.presencia(p) for p in rs], ignore_index=True)
        pasos = K.pasos_boya(pres, "WVHT")
        mb = modelo_boyas[modelo_boyas["boya"] == b.estacion].set_index("tiempo")
        pasos["boya"] = b.estacion
        pasos["hs_mod"] = mb["VHM0"].reindex(pasos["tiempo"]).to_numpy(float)
        pasos["marca_detector"] = mb["marca_detector"].reindex(pasos["tiempo"]).fillna(False).to_numpy(bool)
        pasos["hs_nodo"] = nodo_hs.reindex(pasos["tiempo"]).to_numpy(float)
        pasos["unidad"] = b.estacion + "-" + pd.DatetimeIndex(pasos["tiempo"]).strftime("%Y-%m-%d")
        partes.append(pasos[np.isfinite(pasos["hs_mod"]) & np.isfinite(pasos["hs_nodo"])])
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def colocaciones_de_lado(alt: pd.DataFrame, boy: pd.DataFrame, nombre: str) -> pd.DataFrame:
    lado = C.LADOS[nombre]
    if lado.fuente == "altimetría":
        x = alt[alt["costa_min_km"] == C.COSTA_MIN_KM]
        return x[K.en_lado(x, lado, "mision")]
    return boy[K.en_lado(boy, lado, "boya")]


def resumen_lado(df: pd.DataFrame) -> dict:
    t = pd.DatetimeIndex(df["tiempo"])
    out = {
        "colocaciones": int(len(df)),
        "unidades": int(df["unidad"].nunique()) if len(df) else 0,
        "meses": int(pd.Series(t.to_period("M")).nunique()) if len(df) else 0,
        "anios_con_dato": sorted({int(a) for a in t.year}) if len(df) else [],
        "inicio": str(t.min()) if len(df) else None,
        "fin": str(t.max()) if len(df) else None,
    }
    if "mision" in df:
        out["por_mision"] = {k: int(v) for k, v in df["mision"].value_counts().sort_index().items()}
    return out


def escalones(alt: pd.DataFrame, boy: pd.DataFrame, efectos: dict[str, float], B: int, rng) -> pd.DataFrame:
    """Para cada comparación: S_mod y S_nodo por estadístico, IC de la diferencia, regla del ⅓ y traslado (§3.7)."""
    filas = []
    for comp in C.COMPARACIONES:
        l1, l2 = colocaciones_de_lado(alt, boy, comp.lado1), colocaciones_de_lado(alt, boy, comp.lado2)
        efecto = efectos[comp.familia]
        if len(l1) == 0 or len(l2) == 0:
            filas.append(
                {
                    "comparacion": comp.nombre,
                    "familia": comp.familia,
                    "fuente": comp.fuente,
                    "principal": comp.principal,
                    "estadistico": "P90",
                    "n1": len(l1),
                    "n2": len(l2),
                    "estado": "sin colocaciones",
                }
            )
            continue
        b1 = E.LadoBoot.desde(l1["tiempo"], l1["hs_mod"], l1["hs_nodo"])
        b2 = E.LadoBoot.desde(l2["tiempo"], l2["hs_mod"], l2["hs_nodo"])
        es = E.escalon(b1, b2, B, rng)
        for k, est in enumerate(E.ESTADISTICOS):
            s_mod, s_nodo = es.puntual[0, k], es.puntual[1, k]
            dif = es.replicas[:, 0, k] - es.replicas[:, 1, k]
            lo, hi = E.ic(dif)
            tol = C.TOLERANCIA_TRASLADO * efecto
            fila = {
                "comparacion": comp.nombre,
                "familia": comp.familia,
                "fuente": comp.fuente,
                "principal": comp.principal,
                "estadistico": est,
                "n1": len(l1),
                "n2": len(l2),
                "s_mod": s_mod,
                "s_nodo": s_nodo,
                "dif": s_mod - s_nodo,
                "dif_ic_lo": lo,
                "dif_ic_hi": hi,
                "efecto_nodo": efecto,
                "tolerancia_traslado": tol,
                "traslado": "válido" if (np.isfinite(lo) and -tol <= lo and hi <= tol) else "dudoso",
            }
            if est == "P90":  # §4.4: el efecto por fuente y la regla del ⅓ usan el P90
                fila["efecto_fuente"] = float(min(abs(s_mod), efecto))
                fila["atribuye"] = bool(abs(s_mod) >= C.FRACCION_MIN_ESCALON * efecto)
            filas.append(fila)
    return pd.DataFrame(filas)


def pesos_hy2(alt: pd.DataFrame) -> dict:
    """Peso de HY-2A y HY-2B en el lado A (por colocaciones), con la ventana principal y la de 2026."""
    out = {}
    for nombre in ("A", "A_hasta_2026"):
        x = colocaciones_de_lado(alt, pd.DataFrame(), nombre)
        n = x["mision"].value_counts()
        tot = int(n.sum())
        out[nombre] = {m: round(int(n.get(m, 0)) / tot, 4) if tot else None for m in ("HY-2A", "HY-2B")} | {"n": tot}
    return out


def tierra(raw: Path):
    """Costa de Natural Earth ya registrada en raw (la de la F1)."""
    return costa_mod.leer_tierra(costa_mod.obtener_natural_earth(raw))
