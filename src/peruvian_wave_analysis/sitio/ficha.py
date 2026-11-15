"""Ficha del sitio de la F1 (contrato §2–§4): exposición, obstáculos, observaciones, cronología y gate.

Salidas en ``WAVE-DATA/processed/F01-sitio-y-fuentes/ficha-sitio/<nombre>/``. La serie se descarga y se
verifica aparte (``serie-waverys``); aquí solo se exige que su informe de gate exista y cumpla.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import __version__, manifiesto, rutas
from . import batimetria, costa, exposicion as ex, observaciones as ob, waverys

RESPALDO = (-14.8, -76.0)
URL_IBTRACS = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.SP.list.v04r01.csv")
RADIO_CICLONES_KM = 1000.0

# Cronología de WAVERYS (contrato §3.6). Cada fila con su referencia, leída el 2026-11-14.
QUID = "CMEMS-GLO-QUID-001-032 v1.5 (31/05/2024)"
PUM = "CMEMS-GLO-PUM-001-032 v1.6 (02/2026)"
CRONOLOGIA = [
    ("1980-01-01", "inicio del registro", "Serie extendida hacia atrás; 1980–1992 sin asimilación y sin corrientes (configuración V1.2)", f"{QUID} §VI.4"),
    ("1980-01-01", "forzamiento", "Vientos y hielo de ERA5, cada 3 h, en todo el período", f"{QUID} §II"),
    ("1980-01-01", "batimetría del modelo", "ETOPO2 (NOAA), 2′", f"{QUID} §II"),
    ("1993-01-01", "transición", "Enero de 1993 se reconstruye combinando linealmente la serie sin asimilar y la asimilada", f"{QUID} §VI.4"),
    ("1993-01-01", "corrientes", "Corrientes de superficie de GLORYS12 (cada 3 h) desde 1993", f"{QUID} §II y §VI.4"),
    ("1993-01-01", "asimilación", "Asimilación de Hs de altímetros (ventana de 3 h); misiones y períodos en la tabla 3", f"{QUID} §II, tabla 3"),
    ("2016-03-01", "asimilación", "Espectros SAR de Sentinel-1A (partición del swell)", f"{QUID} §II"),
    ("2020-02-01", "asimilación", "Espectros de CFOSAT/SWIM", f"{QUID} §II, tabla 3"),
    ("2023-04-01", "tramo interino", "Producción interina mensual (M−4 en el QUID; M−2 en el PUM 1.6)", f"{QUID} §VI.3.2; {PUM}"),
    ("2025-05-30", "fusión de datasets", "El dataset multianual y el interino se fusionan en cmems_mod_glo_wav_my_0.2deg_PT3H-i", f"{PUM}, historial (v1.5)"),
]


def _bajar(url: str) -> bytes:
    """Punto de sustitución en las pruebas."""
    import requests

    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


@dataclass
class Insumos:
    raiz: Path
    nodo: tuple[float, float]
    gebco: Path
    malla: Path


def _mascara_modelo(malla: Path):
    lat, lon, valido = waverys.leer_malla(malla, "VHM0", (-180, 180, -90, 90))
    h = float(np.median(np.diff(lat))) / 2

    def es_tierra(la, lo):
        i, j = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
        return not bool(valido[i, j])
    return es_tierra, (float(lon.min() - h), float(lon.max() + h), float(lat.min() - h), float(lat.max() + h))


def _gebco(ruta: Path) -> xr.DataArray:
    with xr.open_dataset(ruta) as ds:
        return batimetria.normalizar(ds, None, (-180, 180, -90, 90)).load()


def _ciclones(raiz_raw: Path, lat0: float, lon0: float) -> dict:
    destino = raiz_raw / "ciclones" / "ibtracs" / "ibtracs.SP.list.v04r01.csv"
    if destino.exists():
        manifiesto.verificar(raiz_raw, destino)
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(_bajar(URL_IBTRACS))
        manifiesto.registrar(raiz_raw, destino, {
            "fuente": "NOAA NCEI", "producto": "IBTrACS v04r01, cuenca del Pacífico Sur (SP)", "servicio": URL_IBTRACS,
            "modulo": "peruvian_wave_analysis.sitio.ficha"})
    df = pd.read_csv(destino, skiprows=[1], usecols=["SID", "SEASON", "NAME", "LAT", "LON", "NATURE"], low_memory=False)
    d = ex.distancia_km(lat0, lon0, pd.to_numeric(df["LAT"], errors="coerce"), pd.to_numeric(df["LON"], errors="coerce"))
    cerca = df[d <= RADIO_CICLONES_KM]
    return {"archivo": str(destino), "radio_km": RADIO_CICLONES_KM, "sistemas": int(cerca["SID"].nunique()),
            "puntos": int(len(cerca)), "temporadas": sorted(cerca["SEASON"].astype(str).unique().tolist()),
            "total_registros_SP": int(len(df))}


def generar(nombre: str = "marcona-200m", raiz_datos: str | None = None) -> dict:
    raiz = Path(rutas.directorio_datos(raiz_datos))
    raiz_raw = rutas.raw(raiz)
    final = raiz / "processed" / "F01-sitio-y-fuentes" / "mapa-sitio-final" / nombre
    ficha_nodo = json.loads((final / "ficha_nodo.json").read_text(encoding="utf-8"))
    lat0, lon0 = float(ficha_nodo["nodo_elegido"]["lat"]), float(ficha_nodo["nodo_elegido"]["lon"])
    gebco = next(iter(sorted((raiz_raw / "batimetria" / "gebco").glob("*.nc"))))
    malla = next(iter(sorted((raiz_raw / "waverys" / "malla").glob("*.nc"))))
    destino = raiz / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / nombre
    gate_serie = destino / "serie_gate.json"

    # Exposición y obstáculos (§3.1, §3.2).
    rutas_ne = costa.obtener_natural_earth(raiz_raw)
    tierra = costa.leer_tierra(rutas_ne)
    es_tierra_m, ext_m = _mascara_modelo(malla)
    expo = ex.calcular(tierra, lat0, lon0, mascara_modelo=(es_tierra_m, ext_m))
    elev = _gebco(gebco)
    islas_ne = [g for g in json_geoms(rutas_ne[1])]
    obst = ex.obstaculos(elev, islas_ne, lat0, lon0, expo.sectores, es_tierra_m)

    # Observaciones (§3.4) para el nodo y el respaldo.
    rutas_alt, faltan = ob.descargar(raiz_raw, lat0, lon0)
    rutas_alt2, faltan2 = ob.descargar(raiz_raw, *RESPALDO)
    todas = sorted(set(rutas_alt) | set(rutas_alt2))
    med = ob.leer(todas)

    def prof(la, lo):
        if not (elev["lat"].min() <= la <= elev["lat"].max() and elev["lon"].min() <= lo <= elev["lon"].max()):
            return np.nan
        return float(-elev.sel(lat=la, lon=lo, method="nearest"))

    resultados = {}
    for etiqueta, (la, lo) in {"principal": (lat0, lon0), "respaldo": RESPALDO}.items():
        expo_n = expo if etiqueta == "principal" else ex.calcular(tierra, la, lo)
        cst = ob.Costa.desde_tierra(tierra, la, lo)
        d = ob.clasificar(med, la, lo, cst, prof, expo_n.sectores)
        res = ob.resumen(d)
        resultados[etiqueta] = {"mediciones": d, "resumen": res, "cobertura": ob.cobertura(res),
                                "dist_costa_ne_km": d.attrs["dist_costa_nodo_km"]}
    reglas = {reg: ob.regla_respaldo(resultados["principal"]["cobertura"], resultados["respaldo"]["cobertura"], reg)
              for reg in resultados["principal"]["cobertura"]["regimen"]}
    ciclones = _ciclones(raiz_raw, lat0, lon0)
    crono = pd.DataFrame(CRONOLOGIA, columns=["fecha", "tipo", "descripcion", "referencia"])
    asim = pd.DataFrame([{"mision": m, "inicio": a, "fin": b or "presente", "referencia": f"{QUID}, tabla 3"}
                         for m, (a, b) in ob.ASIMILADAS_QUID.items()])

    # Gate de la F1 (§4).
    fallas = []
    if not gate_serie.exists() or not json.loads(gate_serie.read_text(encoding="utf-8")).get("cumple"):
        fallas.append("falta el informe de gate de la serie o no cumple (correr `serie-waverys`)")
    for etiqueta in resultados:
        r = resultados[etiqueta]["resumen"]
        if len(r) and (r["util_para_sesgo"] & (r["asimilacion"] != "no asimilada")).any():
            fallas.append(f"{etiqueta}: una observación asimilada o no verificable quedó como útil para el sesgo")
    if crono["referencia"].isna().any():
        fallas.append("cronología sin referencia")
    if expo.sectores["clase"].isna().any():
        fallas.append("sector sin clase")

    # Publicación (staging + MANIFEST.json).
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        expo.rayos.to_csv(t / "exposicion_rayos.csv", index=False, float_format="%.3f")
        expo.sectores.to_csv(t / "exposicion.csv", index=False, float_format="%.3f")
        obst.to_csv(t / "obstaculos.csv", index=False, float_format="%.4f")
        pd.concat([r["resumen"].assign(nodo=k) for k, r in resultados.items()]).to_csv(
            t / "observaciones.csv", index=False)
        pd.concat([r["cobertura"].assign(nodo=k) for k, r in resultados.items()]).to_csv(
            t / "cobertura_observaciones.csv", index=False, float_format="%.2f")
        crono.to_csv(t / "cronologia_waverys.csv", index=False)
        asim.to_csv(t / "misiones_asimiladas_waverys.csv", index=False)
        resumen = {
            "nodo": {"lat": lat0, "lon": lon0}, "respaldo": {"lat": RESPALDO[0], "lon": RESPALDO[1]},
            "dist_costa_natural_earth_km": {k: r["dist_costa_ne_km"] for k, r in resultados.items()},
            "dist_costa_gebco_km": ficha_nodo["nodo_elegido"]["dist_costa_km"],
            "prof_nodo_gebco_m": ficha_nodo["nodo_elegido"]["prof_nodo_m"],
            "sectores": expo.sectores.to_dict("records"),
            "obstaculos": int(len(obst)), "teselas_inexistentes": sorted(set(faltan) | set(faltan2)),
            "regla_respaldo_por_regimen": reglas, "ciclones_tropicales": ciclones,
            "gate": {"cumple": not fallas, "fallas": fallas}, "version_codigo": __version__,
        }
        (t / "resumen_ficha.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str) + "\n",
                                             encoding="utf-8")
        destino.mkdir(parents=True, exist_ok=True)
        for p in t.iterdir():
            shutil.copy2(p, destino / p.name)
    salidas = {p.name: manifiesto.sha256(p) for p in sorted(destino.iterdir()) if p.name != "MANIFEST.json"}
    (destino / "MANIFEST.json").write_text(json.dumps({"version_codigo": __version__, "salidas_sha256": salidas},
                                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resumen["destino"] = str(destino)
    return resumen


def json_geoms(ruta: Path) -> list:
    from shapely.geometry import shape

    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return [shape(f["geometry"]) for f in datos["features"] if f.get("geometry")]


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="ficha-sitio", description="Ficha del sitio de la F1 y su gate (contrato §2–§4).")
    p.add_argument("--nombre", default="marcona-200m")
    p.add_argument("--datos")
    a = p.parse_args(argv)
    r = generar(a.nombre, a.datos)
    print(json.dumps({k: r[k] for k in ("dist_costa_natural_earth_km", "obstaculos", "regla_respaldo_por_regimen",
                                        "ciclones_tropicales", "gate", "destino")}, ensure_ascii=False, indent=2, default=str))
    return 0 if r["gate"]["cumple"] else 1
