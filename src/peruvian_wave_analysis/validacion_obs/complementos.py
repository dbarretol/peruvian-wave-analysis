"""Salidas descriptivas de la etapa 3: períodos en las boyas (§3.6), saltos con pasada no asimilada (§3.8) y la
prueba de la serie corregida en los tres cambios (§6.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..comun import semillas
from ..homogeneidad import cuantiles as Q
from ..validacion import config as CV
from ..validacion import escalon as E
from . import config as C

# --- Períodos (§3.6) -------------------------------------------------------------------------------------


def _ic_mediana_dif(t1, d1, t2, d2, B: int, rng) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Mediana de cada lado y réplicas por bloques de un mes (independientes entre lados)."""

    def reps(t, d):
        if len(d) == 0:
            return np.full(B, np.nan)
        ym = pd.DatetimeIndex(t).year * 12 + pd.DatetimeIndex(t).month
        _, blo = np.unique(ym, return_inverse=True)
        K = blo.max() + 1
        out = np.empty(B)
        orden = [np.flatnonzero(blo == k) for k in range(K)]
        for i in range(B):
            sel = rng.integers(0, K, K)
            idx = np.concatenate([orden[k] for k in sel])
            out[i] = np.median(d[idx])
        return out

    return (
        float(np.median(d1)) if len(d1) else np.nan,
        float(np.median(d2)) if len(d2) else np.nan,
        reps(t1, d1),
        reps(t2, d2),
    )


def periodos(boy: pd.DataFrame, B: int) -> pd.DataFrame:
    """Por boya y umbral (Hs de la boya ≥ P90 y ≥ P99 de cada lado): diferencia modelo − boya de APD frente a VTM02
    (principal) y de DPD frente a VTPK (descriptivo), mediana, P10–P90 e IC; y el cambio entre lados."""
    filas = []
    pares = (("32302", "T0b", "T1b"), ("32012", "A1b", "A2b"))
    from ..validacion import colocacion as K  # noqa: PLC0415

    for estacion, n1, n2 in pares:
        lados = {n: boy[K.en_lado(boy, CV.LADOS[n], "boya")] for n in (n1, n2)}
        for umbral, q in C.CUANTILES_PERIODO.items():
            for lectura, obs, mod in (("principal", "apd_obs", "tm02_mod"), ("descriptivo", "dpd_obs", "tp_mod")):
                datos = {}
                for n, x in lados.items():
                    if x.empty:
                        datos[n] = (pd.DatetimeIndex([]), np.array([]))
                        continue
                    w = Q.pesos_mensuales(x["tiempo"])
                    u = Q.cuantil_ponderado(x["hs_obs"].to_numpy(float), w, q)[0]
                    s = x[(x["hs_obs"] >= u) & x[obs].notna() & x[mod].notna()]
                    datos[n] = (pd.DatetimeIndex(s["tiempo"]), (s[mod] - s[obs]).to_numpy(float))
                rng = semillas.generador("f05", "observada", "periodos", estacion, umbral, lectura)
                m1, m2, r1, r2 = _ic_mediana_dif(*datos[n1], *datos[n2], B, rng)
                for n, (_t, d), med, r in ((n1, datos[n1], m1, r1), (n2, datos[n2], m2, r2)):
                    filas.append(
                        {
                            "boya": estacion,
                            "umbral_hs": umbral,
                            "lectura": lectura,
                            "comparacion": "APD frente a VTM02" if lectura == "principal" else "DPD frente a VTPK",
                            "lado": n,
                            "pasos": len(d),
                            "mediana_modelo_menos_boya_s": med,
                            "p10_s": float(np.quantile(d, 0.1)) if len(d) else np.nan,
                            "p90_s": float(np.quantile(d, 0.9)) if len(d) else np.nan,
                            "ic95_lo": E.ic(r)[0],
                            "ic95_hi": E.ic(r)[1],
                        }
                    )
                cambio = r2 - r1
                lo, hi = E.ic(cambio)
                if not np.isfinite(lo):
                    clase = "no disponible"
                elif lo > C.TOL_PERIODO_S or hi < -C.TOL_PERIODO_S:
                    clase = "desplazamiento de Tp del producto, medido en la cuenca"
                elif -C.TOL_PERIODO_S <= lo and hi <= C.TOL_PERIODO_S:
                    clase = "desplazamiento de Tp del clima o de la cuenca"
                else:
                    clase = "indeterminado (el IC corta ±0.3 s)"
                if lectura == "descriptivo":
                    clase = f"descriptivo; resolución de banda no disponible ({clase})"
                else:  # §20.9 (S-30): APD frente a VTM02 mide el período medio, no Tp (rótulo posterior a los datos)
                    clase = clase.replace("de Tp", "del período medio (Tm02)")
                filas.append(
                    {
                        "boya": estacion,
                        "umbral_hs": umbral,
                        "lectura": lectura,
                        "comparacion": "APD frente a VTM02" if lectura == "principal" else "DPD frente a VTPK",
                        "lado": f"{n2} − {n1}",
                        "pasos": len(datos[n1][1]) + len(datos[n2][1]),
                        "mediana_modelo_menos_boya_s": m2 - m1,
                        "p10_s": np.nan,
                        "p90_s": np.nan,
                        "ic95_lo": lo,
                        "ic95_hi": hi,
                        "clase": clase,
                    }
                )
    return pd.DataFrame(filas)


# --- Saltos (§3.8) ---------------------------------------------------------------------------------------------


def saltos_validados(serie_sd: pd.Series, c_t0: float, so: pd.DataFrame) -> pd.DataFrame:
    """Saltos relativos (umbral c_T0 de la F4) con una superobservación de HY-2A o HY-2B a ±1.5 h. Descriptivo."""
    from ..homogeneidad import altimetros as A  # noqa: PLC0415

    hs = serie_sd.to_numpy(float)
    salto = A.saltos_relativos(hs, c_t0)
    t = pd.DatetimeIndex(serie_sd.index)
    hy = so[so["mision"].isin(["HY-2A", "HY-2B"]) & (so["costa_min_km"] == CV.COSTA_MIN_KM)]
    th = pd.DatetimeIndex(hy["tiempo"]).to_numpy()
    filas = []
    for i in np.flatnonzero(salto):
        cerca = np.abs(th - t[i].to_datetime64()) <= np.timedelta64(90, "m")
        for r in hy[cerca].itertuples():
            filas.append(
                {
                    "tiempo_salto": t[i],
                    "hs_nodo_antes": hs[i - 1],
                    "hs_nodo_despues": hs[i],
                    "mision": r.mision,
                    "tiempo_pasada": r.tiempo,
                    "celda": r.celda,
                    "dist_nodo_km": r.dist_nodo_km,
                    "hs_modelo_celda": r.hs_mod,
                    "hs_observada": r.hs_obs,
                    "posterior_2023_03": bool(pd.Timestamp(r.tiempo) > pd.Timestamp("2023-03-31 23:59:59")),
                }
            )
    cols = ["tiempo_salto", "hs_nodo_antes", "hs_nodo_despues", "mision", "tiempo_pasada", "celda", "dist_nodo_km",
            "hs_modelo_celda", "hs_observada", "posterior_2023_03"]  # fmt: skip
    return pd.DataFrame(filas, columns=cols)


# --- Serie corregida sin escalón nuevo (§6.2) --------------------------------------------------------------------


def _estadisticos(t: pd.DatetimeIndex, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    lb = E.LadoBoot.desde(t[ok], x[ok])
    return E.puntual(lb)[0]


def escalones_serie(t: pd.DatetimeIndex, orig: np.ndarray, corr: np.ndarray, cambios=C.CAMBIOS_SERIE) -> list[dict]:
    """S(q) = log(Q_lado2/Q_lado1) en cada cambio, sin corregir y corregida. Marca si la corrección lo agranda."""
    filas = []
    for nombre, (a1, b1), (a2, b2) in cambios:
        s1 = (t >= pd.Timestamp(a1)) & (t < pd.Timestamp(b1) + pd.Timedelta(days=1))
        s2 = (t >= pd.Timestamp(a2)) & (t < pd.Timestamp(b2) + pd.Timedelta(days=1))
        if not s1.any() or not s2.any():
            continue
        so = np.log(_estadisticos(t[s2], orig[s2]) / _estadisticos(t[s1], orig[s1]))
        sc = np.log(_estadisticos(t[s2], corr[s2]) / _estadisticos(t[s1], corr[s1]))
        for k, e in enumerate(E.ESTADISTICOS):
            filas.append(
                {
                    "cambio": nombre,
                    "estadistico": e,
                    "S_sin_corregir": float(so[k]),
                    "S_corregida": float(sc[k]),
                    "agranda": bool(abs(sc[k]) > abs(so[k]) + C.TOL_ESCALON_CORREGIDO),
                }
            )
    return filas
