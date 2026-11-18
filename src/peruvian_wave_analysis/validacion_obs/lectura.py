"""Lectura de los valores observados y su control de calidad (contrato v6, §3.2 y §3.3). **Solo etapa 3.**

La geometría se arma con el mismo código de la etapa 2 (``validacion.colocacion``): los identificadores de pasada,
las celdas y los tramos de calibración salen idénticos. Encima se añaden:
- **altimetría:** ``SWH_KU_CAL`` de cada punto de 1 Hz, el filtro 2 de §3.3 (Hs en (0, 25]) y la superobservación
  (media de los puntos válidos, al menos 3);
- **boyas:** los registros horarios de WVHT, DPD y APD, con los centinelas como ausentes, los rangos de §3.3 y la
  prueba de pico; el paso de 3 h es la media de los registros de su ventana (al menos 2).

``sintetico`` (solo para el ``--previo``): mismo armado, pero los valores salen de la geometría y del modelo, sin abrir
ninguna variable observada (``sitio.observaciones.leer`` y ``ndbc.presencia``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..sitio import observaciones
from ..sitio.ficha import _gebco, _profundidad
from ..validacion import colocacion as K
from ..validacion import config as CV
from ..validacion import ndbc
from . import config as C

# --- Altimetría ---------------------------------------------------------------------------------------------


def leer_altimetria(rutas: list[Path]) -> pd.DataFrame:
    """Lo mismo que ``observaciones.leer`` (bandera = 1) más ``hs``, la Hs calibrada de la banda del archivo."""
    partes = []
    for r in rutas:
        with xr.open_dataset(r) as ds:
            banda = next((b for b in ("KU", "KA") if f"SWH_{b}_quality_control" in ds), None)
            if banda is None or f"SWH_{banda}_CAL" not in ds:
                raise ValueError(f"{Path(r).name}: sin Hs calibrada ni bandera de calidad.")
            ok = np.asarray(ds[f"SWH_{banda}_quality_control"].values) == 1
            partes.append(
                pd.DataFrame(
                    {
                        "mision": Path(r).parent.name,
                        "banda": banda,
                        "tiempo": pd.to_datetime(ds["TIME"].values[ok]),
                        "lat": ds["LATITUDE"].values[ok].astype(float),
                        "lon": (ds["LONGITUDE"].values[ok].astype(float) + 180.0) % 360.0 - 180.0,
                        "prof_archivo_m": ds["BOT_DEPTH"].values[ok].astype(float) if "BOT_DEPTH" in ds else np.nan,
                        "hs": ds[f"SWH_{banda}_CAL"].values[ok].astype(float),
                    }
                )
            )
    if not partes:
        return pd.DataFrame(columns=["mision", "banda", "tiempo", "lat", "lon", "prof_archivo_m", "hs"])
    return pd.concat(partes, ignore_index=True).sort_values(["mision", "tiempo"]).reset_index(drop=True)


def superobservaciones(d: pd.DataFrame, costa_min_km: float) -> pd.DataFrame:
    """Como ``colocacion.superobservaciones`` y además ``hs_obs`` (media de los puntos) e ``hs_sd`` (su desviación)."""
    x = d[d["dist_costa_km"] >= costa_min_km]
    cols = ["mision", "pasada", "celda", "tiempo", "lat", "lon", "dist_costa_km", "dist_nodo_km", "n", "tramo_cal"]
    if x.empty:
        return pd.DataFrame(columns=[*cols, "hs_obs", "costa_min_km"])
    g = x.groupby(["pasada", "celda"], sort=True)
    s = g.agg(
        mision=("mision", "first"),
        tiempo=("tiempo", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        dist_costa_km=("dist_costa_km", "mean"),
        dist_nodo_km=("dist_nodo_km", "mean"),
        n=("tiempo", "size"),
        tramo_cal=("tramo_cal", "first"),
        hs_obs=("hs", "mean"),
    ).reset_index()
    s = s[s["n"] >= CV.MIN_PUNTOS_SUPEROBS]
    return s[[*cols, "hs_obs"]].assign(costa_min_km=costa_min_km).reset_index(drop=True)


def _con_modelo(so: pd.DataFrame, region: pd.DataFrame, marcas: pd.DataFrame, nodo_hs: pd.Series) -> pd.DataFrame:
    so = so.copy()
    so["hs_mod"] = K.interpolar(region, so["celda"], so["tiempo"])
    so["hs_nodo"] = K.interpolar(nodo_hs.to_frame("nodo"), np.full(len(so), "nodo"), so["tiempo"])
    so["marca_detector"] = K.marca_en(marcas, so["celda"], so["tiempo"])
    ok = np.isfinite(so["hs_mod"]) & np.isfinite(so["hs_nodo"])
    so = so[ok].reset_index(drop=True)
    so["unidad"] = so["pasada"]
    return so


def colocaciones_altimetria(
    med: pd.DataFrame,
    ruta_gebco: Path,
    tierra,
    sectores: pd.DataFrame,
    region: pd.DataFrame,
    marcas: pd.DataFrame,
    nodo_hs: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    """(superobservaciones con ``hs_obs``, puntos de 1 Hz válidos, recuento de filtros, claves antes del filtro 2).

    ``med``: salida de ``leer_altimetria`` (o sintética, con la columna ``hs``). Los identificadores de pasada se
    calculan antes del filtro de Hs, igual que en la etapa 2.
    """
    cst = observaciones.Costa.desde_tierra(tierra, *CV.NODO)
    prof = _profundidad(_gebco(ruta_gebco))
    d, qc = K.filtrar_altimetria(med, cst, prof, sectores, region.columns)
    antes = pd.concat([superobservaciones(d, CV.COSTA_MIN_KM), superobservaciones(d, CV.COSTA_MIN_SENS_KM)])
    antes = _con_modelo(antes, region, marcas, nodo_hs)
    ok = (d["hs"] > C.HS_RANGO[0]) & (d["hs"] <= C.HS_RANGO[1])
    qc.append({"filtro": "Hs calibrada en (0, 25] m (§3.3, filtro 2; solo etapa 3)", "quedan": int(ok.sum())})
    d = d[ok].copy()
    so = pd.concat([superobservaciones(d, CV.COSTA_MIN_KM), superobservaciones(d, CV.COSTA_MIN_SENS_KM)])
    so = _con_modelo(so.reset_index(drop=True), region, marcas, nodo_hs)
    qc.append(
        {
            "filtro": "superobservaciones ≥ 3 puntos con modelo (≥ 50 km), tras el filtro 2",
            "quedan": int((so["costa_min_km"] == CV.COSTA_MIN_KM).sum()),
        }
    )
    claves = antes[["pasada", "celda", "costa_min_km", "mision", "tiempo"]]
    return so, d.reset_index(drop=True), qc, claves


def sintetico_altimetria(med_sin_hs: pd.DataFrame, region: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """``--previo``: Hs sintética de cada punto (modelo en su celda × un factor por pasada y por celda + ruido de 1 Hz).

    Solo usa la geometría y el modelo; ninguna variable observada.
    """
    d = med_sin_hs.copy()
    celda = K.nombre_celda(d["lat"], d["lon"])
    m = K.interpolar(region, celda, d["tiempo"])
    m = np.where(np.isfinite(m), m, np.nanmedian(region.to_numpy(float)))
    pasada = K.pasadas(d.assign(mision=d["mision"].astype(str)))
    _, ip = np.unique(pasada, return_inverse=True)
    _, ic = np.unique(np.char.add(pasada.astype(str), celda.astype(str)), return_inverse=True)
    f = np.exp(0.10 * rng.standard_normal(ip.max() + 1)[ip] + 0.06 * rng.standard_normal(ic.max() + 1)[ic])
    hs = m * f
    a, b = CV.RUIDO_1HZ
    d["hs"] = hs + (a + b * hs) * rng.standard_normal(len(d))
    return d


# --- Boyas -----------------------------------------------------------------------------------------------


def _pico(hs: pd.Series) -> np.ndarray:
    """§3.3, filtro 3: |Hs − mediana de ±3 h| > 4·(1.4826·MAD de ±12 h). Serie con índice de tiempo ordenado."""
    x = hs.dropna()
    if x.empty:
        return np.zeros(len(hs), bool)
    med3 = x.rolling(pd.Timedelta(hours=2 * C.PICO_MEDIANA_H), center=True, closed="both").median()
    med12 = x.rolling(pd.Timedelta(hours=2 * C.PICO_MAD_H), center=True, closed="both").median()
    dev = (x - med12).abs()
    mad = dev.rolling(pd.Timedelta(hours=2 * C.PICO_MAD_H), center=True, closed="both").median()
    malo = (x - med3).abs() > C.PICO_K * 1.4826 * mad
    malo &= mad > 0
    return malo.reindex(hs.index, fill_value=False).to_numpy(bool)


def control_boya(reg: pd.DataFrame, estacion: str) -> tuple[pd.DataFrame, list[dict]]:
    """Registros horarios de una boya tras el control de §3.3. ``reg``: tiempo, WVHT, DPD, APD (centinelas ya NaN)."""
    r = reg.drop_duplicates("tiempo").sort_values("tiempo").set_index("tiempo")
    qc = [{"fuente": estacion, "filtro": "registros (sin duplicados de borde)", "quedan": int(r["WVHT"].notna().sum())}]
    fuera = ~((r["WVHT"] > C.HS_RANGO[0]) & (r["WVHT"] <= C.HS_RANGO[1])) & r["WVHT"].notna()
    r.loc[fuera, "WVHT"] = np.nan
    qc.append({"fuente": estacion, "filtro": "WVHT en (0, 25] m", "quedan": int(r["WVHT"].notna().sum())})
    for v in ("DPD", "APD"):
        f = ~((r[v] > C.PERIODO_RANGO[0]) & (r[v] <= C.PERIODO_RANGO[1])) & r[v].notna()
        r.loc[f, v] = np.nan
    pico = _pico(r["WVHT"])
    r.loc[pico, ["WVHT", "DPD", "APD"]] = np.nan
    qc.append({"fuente": estacion, "filtro": "prueba de pico (manual de NDBC)", "quedan": int(r["WVHT"].notna().sum())})
    return r.reset_index(), qc


def pasos(reg: pd.DataFrame) -> pd.DataFrame:
    """Pasos de 3 h con al menos 2 registros de WVHT en [t − 1.5 h, t + 1.5 h): media de Hs, de DPD y de APD.

    DPD y APD solo con al menos 2 registros válidos de esa variable en la ventana.
    """
    r = reg[reg["WVHT"].notna()].copy()
    r["paso"] = (pd.DatetimeIndex(r["tiempo"]) + pd.Timedelta(hours=CV.VENTANA_BOYA_H)).floor("3h")
    g = r.groupby("paso")
    out = pd.DataFrame({"hs_obs": g["WVHT"].mean(), "k": g["WVHT"].size()})
    for v in ("DPD", "APD"):
        n = g[v].count()
        out[f"{v.lower()}_obs"] = g[v].mean().where(n >= CV.MIN_REGISTROS_PASO_BOYA)
    out = out[out["k"] >= CV.MIN_REGISTROS_PASO_BOYA]
    return out.reset_index().rename(columns={"paso": "tiempo"})


def colocaciones_boyas(
    registros: dict[str, pd.DataFrame], modelo_boyas: pd.DataFrame, nodo_hs: pd.Series
) -> tuple[pd.DataFrame, list[dict]]:
    """Pasos de cada boya con su valor observado, el modelo (Hs, Tm02 y Tp) en la celda y el nodo en el mismo paso."""
    partes, qc = [], []
    for estacion, reg in registros.items():
        r, q = control_boya(reg, estacion)
        qc += q
        p = pasos(r)
        mb = modelo_boyas[modelo_boyas["boya"] == estacion].set_index("tiempo")
        p["boya"] = estacion
        for col, src in (("hs_mod", "VHM0"), ("tm02_mod", "VTM02"), ("tp_mod", "VTPK")):
            p[col] = mb[src].reindex(p["tiempo"]).to_numpy(float)
        p["marca_detector"] = mb["marca_detector"].reindex(p["tiempo"]).fillna(False).to_numpy(bool)
        p["hs_nodo"] = nodo_hs.reindex(p["tiempo"]).to_numpy(float)
        p["unidad"] = estacion + "-" + pd.DatetimeIndex(p["tiempo"]).strftime("%Y-%m-%d")
        p = p[np.isfinite(p["hs_mod"]) & np.isfinite(p["hs_nodo"])]
        qc.append({"fuente": estacion, "filtro": "pasos de 3 h con ≥ 2 registros y modelo", "quedan": int(len(p))})
        partes.append(p)
    return (pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()), qc


def leer_boyas(archivos: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Registros horarios (tiempo, WVHT, DPD, APD) de cada boya, con los centinelas como NaN. **Lee valores.**"""
    out = {}
    for b in ndbc.BOYAS:
        rs = [p for r, p in sorted(archivos.items()) if r.split("/")[2] == b.estacion]
        if rs:
            out[b.estacion] = pd.concat(
                [ndbc._leer(p)[0][["tiempo", *ndbc.VARIABLES_F5]] for p in rs], ignore_index=True
            )
    return out


def sintetico_boyas(archivos: dict[str, Path], modelo_boyas: pd.DataFrame, rng) -> dict[str, pd.DataFrame]:
    """``--previo``: registros sintéticos donde la máscara de presencia dice que hay dato (sin abrir los valores)."""
    out = {}
    for b in ndbc.BOYAS:
        rs = [p for r, p in sorted(archivos.items()) if r.split("/")[2] == b.estacion]
        if not rs:
            continue
        pres = pd.concat([ndbc.presencia(p) for p in rs], ignore_index=True).drop_duplicates("tiempo")
        mb = modelo_boyas[modelo_boyas["boya"] == b.estacion].set_index("tiempo")
        paso = (pd.DatetimeIndex(pres["tiempo"]) + pd.Timedelta(hours=CV.VENTANA_BOYA_H)).floor("3h")
        dia = pd.DatetimeIndex(pres["tiempo"]).floor("D")
        _, idia = np.unique(dia, return_inverse=True)
        f = np.exp(0.10 * rng.standard_normal(idia.max() + 1)[idia] + 0.06 * rng.standard_normal(len(pres)))
        hs = mb["VHM0"].reindex(paso).to_numpy(float) * f
        apd = mb["VTM02"].reindex(paso).to_numpy(float) * np.exp(0.05 * rng.standard_normal(len(pres)))
        dpd = mb["VTPK"].reindex(paso).to_numpy(float)
        out[b.estacion] = pd.DataFrame(
            {
                "tiempo": pres["tiempo"].to_numpy(),
                "WVHT": np.where(pres["WVHT"], hs, np.nan),
                "DPD": np.where(pres["DPD"], dpd, np.nan),
                "APD": np.where(pres["APD"], apd, np.nan),
            }
        )
    return out


def fraccion_perdida(antes: pd.DataFrame, despues: pd.DataFrame, lados: list[str], col: str) -> dict[str, float]:
    """Por lado: fracción de las colocaciones de la etapa 2 que la etapa 3 pierde por sus filtros (§3.3)."""
    out = {}
    for n in lados:
        lado = CV.LADOS[n]
        a = antes[K.en_lado(antes, lado, col)]
        b = despues[K.en_lado(despues, lado, col)]
        out[n] = float(1 - len(b) / len(a)) if len(a) else 0.0
    return out
