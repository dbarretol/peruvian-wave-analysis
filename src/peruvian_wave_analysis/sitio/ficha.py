"""Ficha del sitio de la F1 (contrato §2–§4): exposición, obstáculos, observaciones, cronología y gate.

Salidas en ``WAVE-DATA/processed/F01-sitio-y-fuentes/ficha-sitio/<nombre>/``, publicadas con staging, gate y
manifiesto comunes: un gate fallido borra la ficha anterior. La serie se descarga y se verifica aparte
(``serie-waverys`` y ``serie-waverys-gate``); aquí se exige que su gate publicado exista y cumpla, y se registra
como insumo con su hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import __version__, manifiesto, rutas
from ..comun import etapa, orquestador
from ..comun.errores import GateFallido
from . import batimetria, costa, exposicion as ex, ficha_md, observaciones as ob, serie, waverys

RESPALDO = (-14.8, -76.0)
URL_IBTRACS = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.SP.list.v04r01.csv"
)
RADIO_CICLONES_KM = 1000.0

# Cronología de WAVERYS (contrato §3.6). Cada fila con su referencia, leída el 2026-11-14.
QUID = "CMEMS-GLO-QUID-001-032 v1.5 (31/05/2024)"
PUM = "CMEMS-GLO-PUM-001-032 v1.6 (02/2026)"
CRONOLOGIA = [
    ("1980-01-01", "resolución", "Malla de 1/5° (0.2°), salidas cada 3 h", f"{QUID} §I"),
    (
        "1980-01-01",
        "física",
        "MFWAM (Météo-France) con el código ECWAM-IFS-42R2 y la disipación ST4 (Ardhuin et al., 2010)",
        f"{QUID} §II",
    ),
    (
        "1980-01-01",
        "inicio del registro",
        "Serie extendida hacia atrás; 1980–1992 sin asimilación y sin corrientes (configuración V1.2)",
        f"{QUID} §VI.4",
    ),
    ("1980-01-01", "forzamiento", "Vientos y hielo de ERA5, cada 3 h, en todo el período", f"{QUID} §II"),
    ("1980-01-01", "batimetría del modelo", "ETOPO2 (NOAA), 2′", f"{QUID} §II"),
    (
        "1993-01-01",
        "transición",
        "Enero de 1993 se reconstruye combinando linealmente la serie sin asimilar y la asimilada",
        f"{QUID} §VI.4",
    ),
    ("1993-01-01", "corrientes", "Corrientes de superficie de GLORYS12 (cada 3 h) desde 1993", f"{QUID} §II y §VI.4"),
    (
        "1993-01-01",
        "asimilación",
        "Asimilación de Hs de altímetros (ventana de 3 h); misiones y períodos en la tabla 3",
        f"{QUID} §II, tabla 3",
    ),
    ("2016-03-01", "asimilación", "Espectros SAR de Sentinel-1A (partición del swell)", f"{QUID} §II"),
    ("2020-02-01", "asimilación", "Espectros de CFOSAT/SWIM", f"{QUID} §II, tabla 3"),
    (
        "2023-04-01",
        "tramo interino",
        "Producción interina mensual (M−4 en el QUID; M−2 en el PUM 1.6)",
        f"{QUID} §VI.3.2; {PUM}",
    ),
    (
        "2025-05-30",
        "fusión de datasets",
        "El dataset multianual y el interino se fusionan en cmems_mod_glo_wav_my_0.2deg_PT3H-i",
        f"{PUM}, historial (v1.5)",
    ),
]


# Tipos que exige el contrato §3.6 en la cronología (F01-K-01).
TIPOS_CRONOLOGIA = [
    "resolución",
    "física",
    "forzamiento",
    "corrientes",
    "batimetría del modelo",
    "asimilación",
    "tramo interino",
    "período",
    "fecha de corte",
    "versión",
]


def filas_catalogo(gate: dict | None) -> list[tuple[str, str, str, str]]:
    """Período, fecha de corte y versión, leídos del catálogo anotado en el gate de la serie (F01-K-01)."""
    if not gate or "catalogo" not in gate:
        causa = "no disponible: falta serie_gate.json con el catálogo (correr `serie-waverys-gate`)"
        return [("", t, causa, "no disponible") for t in ("período", "fecha de corte", "versión")]
    c = gate["catalogo"]
    ref = f"Catálogo de Copernicus Marine (`copernicusmarine describe`), consulta {c.get('consulta_utc', 'sin hora')}"
    return [
        (
            c["inicio"][:10],
            "período",
            f"Cobertura del dataset {c['dataset_id']}: {c['inicio']} a {c['fin']}, paso de {c.get('paso_h', 3):g} h",
            ref,
        ),
        (c["fin"][:10], "fecha de corte", f"Último paso disponible el día de la descarga: {c['fin']}", ref),
        (c["fin"][:10], "versión", f"Versión del dataset {c['version']}, fijada en cada pedido (ADR-0016)", ref),
    ]


def gate_ficha(
    resultados: dict, crono: pd.DataFrame, sectores: pd.DataFrame, gate_serie: dict | None, nombre_gebco: str
) -> list[str]:
    """Criterios del gate de la F1 (contrato §4) que dependen de la ficha. Devuelve las fallas (F01-K-02)."""
    fallas = []
    if not gate_serie or not gate_serie.get("cumple"):
        fallas.append("falta el informe de gate de la serie o no cumple (correr `serie-waverys-gate`)")
    for etiqueta, r in resultados.items():
        m = r["mediciones"]
        if len(m):
            util = m[m["util_para_sesgo"].astype(bool)]
            # Comprobación independiente de ob.asimilacion: útil solo si es de 1980–1992 o el QUID declara la misión
            # como no asimilada (contrato §3.4); nunca «asimilada» ni «no verificable».
            permitido = (util["tiempo"] < ob.INICIO_ASIMILACION) | util["mision"].isin(list(ob.NO_ASIMILADAS_QUID))
            if (~permitido).any() or util["asimilacion"].isin(["asimilada", "no verificable"]).any():
                malas = sorted(util.loc[~permitido, "mision"].unique())
                fallas.append(
                    f"{etiqueta}: mediciones asimiladas o no verificables marcadas como útiles para el sesgo "
                    f"({', '.join(malas)})"
                )
        if r["prof_archivo_200km"]:
            fallas.append(
                f"{etiqueta}: {r['prof_archivo_200km']} mediciones a 200 km o menos sin profundidad de GEBCO "
                f"(el recorte {nombre_gebco} no cubre la región; contrato §3.4)"
            )
    faltan = [t for t in TIPOS_CRONOLOGIA if t not in set(crono["tipo"])]
    if faltan:
        fallas.append(f"cronología sin las filas de §3.6: {', '.join(faltan)}")
    ref = crono["referencia"].fillna("").astype(str).str.strip()
    if (ref == "").any():
        fallas.append("cronología con filas sin referencia ni «no disponible»")
    if len(sectores) != 16 or not sectores["clase"].isin(["abierto", "parcial", "cerrado"]).all():
        fallas.append("exposición: no hay 16 sectores con clase abierto, parcial o cerrado")
    return fallas


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


def _recuadro_observaciones(nodos) -> tuple[float, float, float, float]:
    """(oeste, este, sur, norte) que cubre el círculo de clase «región» de todos los nodos."""
    o, e, s, n = [], [], [], []
    for la, lo in nodos:
        dla = ob.RADIO_REGION_KM / 111.195
        dlo = ob.RADIO_REGION_KM / (111.195 * np.cos(np.radians(la)))
        o.append(lo - dlo)
        e.append(lo + dlo)
        s.append(la - dla)
        n.append(la + dla)
    return min(o), max(e), min(s), max(n)


def _elegir_gebco(carpeta: Path, recuadro) -> Path:
    """El recorte de GEBCO más pequeño que cubre ``recuadro``; si ninguno lo cubre, el mayor (el gate lo señala)."""
    candidatos = []
    for r in sorted(carpeta.glob("*.nc")):
        with xr.open_dataset(r) as ds:
            e = batimetria.normalizar(ds, None, (-180, 180, -90, 90))
            lo, la = e["lon"].values, e["lat"].values
        cubre = (
            lo.min() <= recuadro[0] and lo.max() >= recuadro[1] and la.min() <= recuadro[2] and la.max() >= recuadro[3]
        )
        area = float(np.ptp(lo) * np.ptp(la))
        candidatos.append((not cubre, area if cubre else -area, r.name, r))
    return min(candidatos)[3]


def _profundidad(elev: xr.DataArray):
    """Profundidad positiva (m) de GEBCO en arreglos de puntos; NaN fuera del recorte."""
    lat_min, lat_max = float(elev["lat"].min()), float(elev["lat"].max())
    lon_min, lon_max = float(elev["lon"].min()), float(elev["lon"].max())

    def prof(la, lo):
        la, lo = np.atleast_1d(np.asarray(la, float)), np.atleast_1d(np.asarray(lo, float))
        dentro = (la >= lat_min) & (la <= lat_max) & (lo >= lon_min) & (lo <= lon_max)
        out = np.full(la.shape, np.nan)
        if dentro.any():
            v = elev.sel(lat=xr.DataArray(la[dentro]), lon=xr.DataArray(lo[dentro]), method="nearest")
            out[dentro] = -np.asarray(v.values, float)
        return out

    return prof


def _json_limpio(x):
    """NaN e infinitos a null, para que el JSON sea válido (revisión de control del 2026-11-15)."""
    if isinstance(x, dict):
        return {k: _json_limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_limpio(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _avance(msg: str) -> None:
    print(f"[ficha] {msg}", flush=True)


def _ciclones(raiz_raw: Path, lat0: float, lon0: float) -> dict:
    destino = raiz_raw / "ciclones" / "ibtracs" / "ibtracs.SP.list.v04r01.csv"
    if destino.exists():
        manifiesto.verificar(raiz_raw, destino)
    else:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(_bajar(URL_IBTRACS))
        manifiesto.registrar(
            raiz_raw,
            destino,
            {
                "fuente": "NOAA NCEI",
                "producto": "IBTrACS v04r01, cuenca del Pacífico Sur (SP)",
                "servicio": URL_IBTRACS,
                "modulo": "peruvian_wave_analysis.sitio.ficha",
            },
        )
    df = pd.read_csv(destino, skiprows=[1], usecols=["SID", "SEASON", "NAME", "LAT", "LON", "NATURE"], low_memory=False)
    d = ex.distancia_km(
        lat0, lon0, pd.to_numeric(df["LAT"], errors="coerce"), pd.to_numeric(df["LON"], errors="coerce")
    )
    cerca = df[d <= RADIO_CICLONES_KM]
    return {
        "archivo": str(destino),
        "radio_km": RADIO_CICLONES_KM,
        "sistemas": int(cerca["SID"].nunique()),
        "puntos": int(len(cerca)),
        "temporadas": sorted(cerca["SEASON"].astype(str).unique().tolist()),
        "total_registros_SP": int(len(df)),
    }


def generar(nombre: str = "marcona-200m", raiz_datos: str | None = None) -> dict:
    raiz = Path(rutas.directorio_datos(raiz_datos))
    raiz_raw = rutas.raw(raiz)
    final = raiz / "processed" / "F01-sitio-y-fuentes" / "mapa-sitio-final" / nombre
    ficha_nodo = json.loads((final / "ficha_nodo.json").read_text(encoding="utf-8"))
    lat0, lon0 = float(ficha_nodo["nodo_elegido"]["lat"]), float(ficha_nodo["nodo_elegido"]["lon"])
    recuadro_obs = _recuadro_observaciones([(lat0, lon0), RESPALDO])
    gebco = _elegir_gebco(raiz_raw / "batimetria" / "gebco", recuadro_obs)
    _avance(f"GEBCO: {gebco.name}")
    malla = next(iter(sorted((raiz_raw / "waverys" / "malla").glob("*.nc"))))
    destino = raiz / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / nombre
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(destino, raices) as t:
        return _generar(
            t, destino, raices, raiz, raiz_raw, final, ficha_nodo, lat0, lon0, recuadro_obs, gebco, malla, nombre
        )


ETAPA = orquestador.Etapa("ficha-sitio", Path(__file__).resolve(), Path("."))


def _generar(
    t: Path,
    destino: Path,
    raices,
    raiz: Path,
    raiz_raw: Path,
    final: Path,
    ficha_nodo: dict,
    lat0: float,
    lon0: float,
    recuadro_obs,
    gebco: Path,
    malla: Path,
    nombre: str,
) -> dict:
    leido = serie.leer_gate(raiz, nombre)
    gate = leido[1] if leido else None

    # Exposición y obstáculos (§3.1, §3.2).
    _avance("exposición y obstáculos")
    rutas_ne = costa.obtener_natural_earth(raiz_raw)
    tierra = costa.leer_tierra(rutas_ne)
    es_tierra_m, ext_m = _mascara_modelo(malla)
    expo = ex.calcular(tierra, lat0, lon0, mascara_modelo=(es_tierra_m, ext_m))
    elev = _gebco(gebco)
    islas_ne = [g for g in json_geoms(rutas_ne[1])]
    obst = ex.obstaculos(elev, islas_ne, lat0, lon0, expo.sectores, es_tierra_m)

    # Observaciones (§3.4) para el nodo y el respaldo.
    _avance("observaciones: descarga o verificación de teselas")
    rutas_alt, faltan = ob.descargar(raiz_raw, lat0, lon0)
    rutas_alt2, faltan2 = ob.descargar(raiz_raw, *RESPALDO)
    todas = sorted(set(rutas_alt) | set(rutas_alt2))
    med = ob.leer(todas)
    _avance(f"observaciones: {len(med)} mediciones válidas leídas")
    prof = _profundidad(elev)

    resultados = {}
    for etiqueta, (la, lo) in {"principal": (lat0, lon0), "respaldo": RESPALDO}.items():
        expo_n = expo if etiqueta == "principal" else ex.calcular(tierra, la, lo)
        cst = ob.Costa.desde_tierra(tierra, la, lo)
        d = ob.clasificar(med, la, lo, cst, prof, expo_n.sectores)
        res = ob.resumen(d)
        res_s = ob.resumen(d, "asimilacion_periodo")
        cerca = d["dist_nodo_km"] <= ob.RADIO_REGION_KM
        resultados[etiqueta] = {
            "mediciones": d,
            "resumen": res,
            "cobertura": ob.cobertura(res),
            "resumen_sens": res_s,
            "cobertura_sens": ob.cobertura(res_s),
            "prof_archivo_200km": int((cerca & (d["fuente_prof"] != "GEBCO_2026")).sum()),
            "dist_costa_ne_km": d.attrs["dist_costa_nodo_km"],
        }
        _avance(f"observaciones: {etiqueta} clasificado")
    reglas = {
        reg: ob.regla_respaldo(resultados["principal"]["cobertura"], resultados["respaldo"]["cobertura"], reg)
        for reg in resultados["principal"]["cobertura"]["regimen"]
    }
    _avance("ciclones (IBTrACS)")
    ciclones = _ciclones(raiz_raw, lat0, lon0)
    crono = pd.DataFrame(CRONOLOGIA + filas_catalogo(gate), columns=["fecha", "tipo", "descripcion", "referencia"])
    asim = pd.DataFrame(
        [
            {"mision": m, "inicio": a, "fin": b or "presente", "referencia": f"{QUID}, tabla 3"}
            for m, (a, b) in ob.ASIMILADAS_QUID.items()
        ]
    )

    # Gate de la F1 (§4).
    fallas = gate_ficha(resultados, crono, expo.sectores, gate, gebco.name)
    if gate:
        fallas += serie.corresponde(gate, raiz_raw)  # F01-K-03: el informe corresponde a los archivos actuales

    # Publicación con la etapa común: todo se escribe en el staging ``t``.
    expo.rayos.merge(expo.sectores[["sector", "clase"]], on="sector", how="left").to_csv(
        t / "exposicion_rayos.csv", index=False, float_format="%.3f"
    )  # F01-K-07: clase del sector en cada rayo
    expo.sectores.to_csv(t / "exposicion.csv", index=False, float_format="%.3f")
    obst.to_csv(t / "obstaculos.csv", index=False, float_format="%.4f")
    pd.concat([r["resumen"].assign(nodo=k) for k, r in resultados.items()]).to_csv(t / "observaciones.csv", index=False)
    pd.concat([r["cobertura"].assign(nodo=k) for k, r in resultados.items()]).to_csv(
        t / "cobertura_observaciones.csv", index=False, float_format="%.2f"
    )
    rotulo = (
        "SENSIBILIDAD posterior a ver los conteos (revisión de control 2026-11-15): los tramos de misiones "
        "de la tabla 3 fuera de su período de asimilación cuentan como no asimilados. No es la regla del contrato."
    )
    pd.concat([r["resumen_sens"].assign(nodo=k) for k, r in resultados.items()]).assign(rotulo=rotulo).to_csv(
        t / "observaciones_sensibilidad_periodo.csv", index=False
    )
    pd.concat([r["cobertura_sens"].assign(nodo=k) for k, r in resultados.items()]).assign(rotulo=rotulo).to_csv(
        t / "cobertura_sensibilidad_periodo.csv", index=False, float_format="%.2f"
    )
    crono.to_csv(t / "cronologia_waverys.csv", index=False)
    asim.to_csv(t / "misiones_asimiladas_waverys.csv", index=False)
    resumen = {
        "nodo": {"lat": lat0, "lon": lon0},
        "respaldo": {"lat": RESPALDO[0], "lon": RESPALDO[1]},
        "dist_costa_natural_earth_km": {k: r["dist_costa_ne_km"] for k, r in resultados.items()},
        "dist_costa_gebco_km": ficha_nodo["nodo_elegido"]["dist_costa_km"],
        "prof_nodo_gebco_m": ficha_nodo["nodo_elegido"]["prof_nodo_m"],
        "gebco_observaciones": {
            "archivo": gebco.name,
            "recuadro_requerido_oesn": list(recuadro_obs),
            "mediciones_200km_sin_gebco": {k: r["prof_archivo_200km"] for k, r in resultados.items()},
        },
        "sectores": expo.sectores.to_dict("records"),
        "obstaculos": int(len(obst)),
        "teselas_inexistentes": sorted(set(faltan) | set(faltan2)),
        "regla_respaldo_por_regimen": reglas,
        "ciclones_tropicales": ciclones,
        "gate": {"cumple": not fallas, "fallas": fallas},
        "version_codigo": __version__,
    }
    (t / "resumen_ficha.json").write_text(
        json.dumps(_json_limpio(resumen), ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _escribir_md(t, raiz_raw, ficha_nodo, gate)
    insumos = {
        "ficha_nodo": final / "ficha_nodo.json",
        "gebco": gebco,
        "malla": malla,
        "ibtracs": Path(ciclones["archivo"]),
        **{f"natural_earth:{p.name}": p for p in rutas_ne},
        **{f"altimetria:{p.parent.name}/{p.name}": p for p in todas},
    }
    if leido:
        insumos["serie_gate"] = leido[0]
    estatico = _deptho(raiz_raw)
    if estatico is not None:
        insumos["deptho"] = estatico
    etapa.publicar(t, destino, etapa.Gate(not fallas, {"fallas": len(fallas)}, fallas), raices, insumos, raiz, None)
    resumen["destino"] = str(destino)
    return resumen


def _deptho(raiz_raw: Path) -> Path | None:
    return next(iter(sorted((raiz_raw / "waverys" / "estatico").glob("deptho*.nc"))), None)


def _escribir_md(carpeta: Path, raiz_raw: Path, ficha_nodo: dict, gate: dict | None) -> None:
    n = ficha_nodo["nodo_elegido"]
    texto = ficha_md.componer(
        carpeta, ficha_nodo, gate, ficha_md.deptho_en(_deptho(raiz_raw), n["lat"], n["lon"]), __version__
    )
    (carpeta / "ficha_sitio.md").write_text(texto, encoding="utf-8")


def regenerar_md(nombre: str = "marcona-200m", raiz_datos: str | None = None) -> Path:
    """Rehace solo ``ficha_sitio.md`` desde las salidas ya publicadas, sin recalcular, y vuelve a publicar la ficha
    con los mismos insumos (verificados de nuevo por hash)."""
    import shutil

    raiz = Path(rutas.directorio_datos(raiz_datos))
    final = raiz / "processed" / "F01-sitio-y-fuentes" / "mapa-sitio-final" / nombre
    ficha_nodo = json.loads((final / "ficha_nodo.json").read_text(encoding="utf-8"))
    destino = raiz / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / nombre
    man = etapa.exigir_publicada(destino)
    leido = serie.leer_gate(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(destino, raices) as t:
        for rel in man["salidas"]:
            if rel != "ficha_sitio.md":
                shutil.copy2(destino / rel, t / rel)
        _escribir_md(t, rutas.raw(raiz), ficha_nodo, leido[1] if leido else None)
        insumos = {k: raiz / v["ruta"] for k, v in man["insumos"].items()}
        etapa.publicar(t, destino, etapa.Gate(True, man["gate"]["criterios"]), raices, insumos, raiz, None)
    return destino / "ficha_sitio.md"


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
    p.add_argument("--solo-md", action="store_true", help="Rehace solo ficha_sitio.md desde las salidas publicadas.")
    a = p.parse_args(argv)
    if a.solo_md:
        print(regenerar_md(a.nombre, a.datos))
        return 0
    try:
        r = generar(a.nombre, a.datos)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "dist_costa_natural_earth_km",
                    "obstaculos",
                    "regla_respaldo_por_regimen",
                    "ciclones_tropicales",
                    "gate",
                    "destino",
                )
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if r["gate"]["cumple"] else 1
