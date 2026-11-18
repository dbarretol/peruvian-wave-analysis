"""Colocalización **geométrica** de la F5 (contrato v6, §3.2 y §3.3): no lee ningún valor observado (G4).

- **Altimetría:** de los archivos de Ribal y Young solo se leen hora, posición, bandera de calidad y profundidad
  (``sitio.observaciones.leer``); nunca ``SWH_KU_CAL``. Filtros geométricos de §3.2 y §3.3 (bandera, ≤ 200 km del
  nodo, ≥ 50 km de la costa —25 km en la sensibilidad—, GEBCO > 100 m, sector abierto o parcial, celda con dato y
  tramo de calibración), pasadas, celda y superobservaciones de al menos 3 puntos.
- **Boyas:** del archivo solo se lee el tiempo y la máscara de «dato presente» (comparación de texto con los
  centinelas; ``ndbc.presencia``). Un paso de 3 h se forma con al menos 2 registros en [t − 1.5 h, t + 1.5 h).
- **Modelo en la colocación:** la serie de la celda (etapa 1), interpolada linealmente en el tiempo entre los dos
  pasos de 3 h (altimetría) o en el paso (boyas). Lo mismo con la serie del nodo, para el escalón en los mismos pasos
  (§3.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sitio.exposicion import distancia_km, rumbo, sector_de
from . import config as C

PASO = pd.Timedelta(hours=3)


def nombre_celda(lat, lon) -> np.ndarray:
    """Nombre de la celda de WAVERYS (0.2°) que contiene cada punto: la del nodo de malla más cercano."""
    la = np.round(np.asarray(lat, float) / 0.2) * 0.2
    lo = np.round(np.asarray(lon, float) / 0.2) * 0.2
    return np.array([f"{a:+.1f}_{b:+.1f}" for a, b in zip(la + 0.0, lo + 0.0, strict=True)], dtype=object)


def tramo_calibracion(mision: pd.Series, tiempo: pd.Series) -> np.ndarray:
    """Tramo de calibración de §3.5, o "" si la medición cae fuera de todos (HY-2A entre 2013-04-01 y 2013-04-15)."""
    out = np.full(len(mision), "", dtype=object)
    m, t = mision.to_numpy(), pd.DatetimeIndex(tiempo)
    for mis, tramo, a, b in C.TRAMOS_CALIBRACION:
        sel = (m == mis) & (t >= pd.Timestamp(a)) & (t < pd.Timestamp(b) + pd.Timedelta(days=1))
        out[sel & (out == "")] = tramo
    return out


def pasadas(d: pd.DataFrame, corte_s: float = C.CORTE_PASADA_S) -> np.ndarray:
    """Identificador de pasada: dentro de cada misión, una nueva pasada si el hueco supera ``corte_s``."""
    ds = d.sort_values(["mision", "tiempo"], kind="stable")
    nuevo = ds.groupby("mision")["tiempo"].diff().dt.total_seconds().fillna(np.inf) > corte_s
    num = nuevo.groupby(ds["mision"]).cumsum().astype(int)
    return (ds["mision"] + "-" + num.astype(str)).reindex(d.index).to_numpy()


def filtrar_altimetria(
    med: pd.DataFrame, costa, prof_gebco, sectores: pd.DataFrame, celdas_validas, nodo=C.NODO
) -> tuple[pd.DataFrame, list[dict]]:
    """Mediciones que pasan los filtros geométricos, con su celda, tramo de calibración y pasada.

    ``med``: salida de ``observaciones.leer`` (ya con la bandera = 1). Devuelve también el recuento de cada filtro.
    """
    qc = [{"filtro": "bandera de calidad = 1 (al leer)", "quedan": len(med)}]
    d = med.copy()
    d["mision"] = d["mision"].map(C.MISIONES)
    d = d[d["mision"].notna()].copy()
    qc.append({"filtro": "misiones de la F5", "quedan": len(d)})
    d["dist_nodo_km"] = distancia_km(nodo[0], nodo[1], d["lat"], d["lon"])
    d = d[d["dist_nodo_km"] <= C.RADIO_KM].copy()
    qc.append({"filtro": f"≤ {C.RADIO_KM:g} km del nodo", "quedan": len(d)})
    if d.empty:
        return d.assign(pasada=[], celda=[], tramo_cal=[], dist_costa_km=[]), qc
    d["pasada"] = pasadas(d)
    d["dist_costa_km"] = costa.distancia_km(d["lat"], d["lon"])
    d = d[d["dist_costa_km"] >= C.COSTA_MIN_SENS_KM].copy()
    qc.append({"filtro": f"≥ {C.COSTA_MIN_SENS_KM:g} km de la costa (sensibilidad)", "quedan": len(d)})
    qc.append(
        {
            "filtro": f"≥ {C.COSTA_MIN_KM:g} km de la costa (principal)",
            "quedan": int((d["dist_costa_km"] >= C.COSTA_MIN_KM).sum()),
        }
    )
    la, lo = d["lat"].to_numpy(float), d["lon"].to_numpy(float)
    pg = np.asarray(prof_gebco(la, lo), float)
    d["prof_m"] = np.where(np.isfinite(pg), pg, -d["prof_archivo_m"].astype(float))
    d = d[d["prof_m"] > C.PROF_MIN_M].copy()
    qc.append({"filtro": f"profundidad > {C.PROF_MIN_M:g} m", "quedan": len(d)})
    clase = sectores.set_index("sector")["clase"]
    d["clase_sector"] = pd.Series(sector_de(rumbo(nodo[0], nodo[1], d["lat"], d["lon"])), index=d.index).map(clase)
    d = d[d["clase_sector"].isin(C.SECTORES_VALIDOS)].copy()
    qc.append({"filtro": "sector abierto o parcial", "quedan": len(d)})
    d["celda"] = nombre_celda(d["lat"], d["lon"])
    d = d[d["celda"].isin(set(celdas_validas))].copy()
    qc.append({"filtro": "celda del modelo con dato", "quedan": len(d)})
    d["tramo_cal"] = tramo_calibracion(d["mision"], d["tiempo"])
    d = d[d["tramo_cal"] != ""].copy()
    qc.append({"filtro": "dentro de un tramo de calibración", "quedan": len(d)})
    return d, qc


def superobservaciones(d: pd.DataFrame, costa_min_km: float) -> pd.DataFrame:
    """Una por (pasada, celda) con al menos 3 puntos a ≥ ``costa_min_km`` de la costa. Sin filtro MAD (§3.3)."""
    x = d[d["dist_costa_km"] >= costa_min_km]
    cols = ["mision", "pasada", "celda", "tiempo", "lat", "lon", "dist_costa_km", "dist_nodo_km", "n", "tramo_cal"]
    if x.empty:
        return pd.DataFrame(columns=[*cols, "costa_min_km"])
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
    ).reset_index()
    s = s[s["n"] >= C.MIN_PUNTOS_SUPEROBS]
    return s[cols].assign(costa_min_km=costa_min_km).reset_index(drop=True)


def interpolar(serie: pd.DataFrame, columnas, tiempos) -> np.ndarray:
    """Valor de ``serie[columna]`` (rejilla de 3 h) interpolado linealmente en cada tiempo. NaN fuera de la rejilla
    o si alguno de los dos pasos que lo rodean es NaN (registro excluido por la etapa 1)."""
    t = pd.DatetimeIndex(tiempos)
    idx = pd.DatetimeIndex(serie.index)
    i0 = idx.get_indexer(t.floor("3h"))
    w = ((t - t.floor("3h")) / PASO).to_numpy(float)
    col = serie.columns.get_indexer(pd.Index(columnas))
    v = serie.to_numpy(float)
    ok = (i0 >= 0) & (col >= 0) & ((i0 + 1 < len(idx)) | (w == 0))
    out = np.full(len(t), np.nan)
    i1 = np.minimum(i0 + 1, len(idx) - 1)
    out[ok] = (1 - w[ok]) * v[i0[ok], col[ok]] + w[ok] * v[i1[ok], col[ok]]
    return out


def marca_en(marcas: pd.DataFrame, columnas, tiempos) -> np.ndarray:
    """True si alguno de los dos pasos que rodean el tiempo lleva la marca del detector."""
    t = pd.DatetimeIndex(tiempos)
    idx = pd.DatetimeIndex(marcas.index)
    i0 = idx.get_indexer(t.floor("3h"))
    i1 = np.minimum(i0 + 1, len(idx) - 1)
    col = marcas.columns.get_indexer(pd.Index(columnas))
    m = marcas.to_numpy(bool)
    ok = (i0 >= 0) & (col >= 0)
    out = np.zeros(len(t), bool)
    out[ok] = m[i0[ok], col[ok]] | m[i1[ok], col[ok]]
    return out


def pasos_boya(presencia: pd.DataFrame, variable: str = "WVHT") -> pd.DataFrame:
    """Pasos de 3 h con al menos 2 registros con dato de ``variable`` en [t − 1.5 h, t + 1.5 h). Sin valores.

    ``presencia``: salida de ``ndbc.presencia`` de uno o varios archivos. Los registros repetidos en el borde entre
    archivos anuales cuentan una vez.
    """
    p = presencia.drop_duplicates("tiempo").sort_values("tiempo")
    t = pd.DatetimeIndex(p.loc[p[variable].to_numpy(bool), "tiempo"])
    paso = (t + pd.Timedelta(hours=C.VENTANA_BOYA_H)).floor("3h")
    k = pd.Series(1, index=paso).groupby(level=0).sum()
    k = k[k >= C.MIN_REGISTROS_PASO_BOYA]
    return pd.DataFrame({"tiempo": k.index, "k": k.to_numpy(int)})


def en_lado(df: pd.DataFrame, lado: C.Lado, columna: str = "mision") -> np.ndarray:
    """Máscara de las colocaciones de un lado (misión o boya y período, fechas inclusivas)."""
    t = pd.DatetimeIndex(df["tiempo"])
    sel = np.zeros(len(df), bool)
    for quien, a, b in lado.partes:
        sel |= (df[columna].to_numpy() == quien) & (t >= pd.Timestamp(a)) & (t < pd.Timestamp(b) + pd.Timedelta(days=1))
    return sel
