"""Inventario de observaciones de altimetría alrededor del nodo (contrato de la F1 §3.4; F01-C-02, C-03, N1, N2).

Fuente: Ribal y Young (2019), altimetría calibrada multimisión, en AODN (IMOS), en archivos por misión
y tesela de 1°. La clasificación usa **solo la geometría** (posición, hora, profundidad y distancia a la
costa); no se miran los valores de Hs.

Independencia respecto de WAVERYS (QUID 1.5, tabla 3 y §VI.4):
- **asimilada:** misión y período en la tabla 3, o período desde 1993 con la misión en esa lista;
- **no asimilada:** período 1980–1992 (sin asimilación), o el QUID declara que no se asimila (HY-2A);
- **no verificable:** la misión no figura en el QUID; por precaución se trata como asimilada.

Sensibilidad (rotulada «posterior a ver los conteos», revisión de control del 2026-11-15): ``asimilacion_por_periodo``
trata como no asimilado el tramo de una misión de la tabla 3 que cae fuera de su período de asimilación
(p. ej. SARAL antes de 2016-03, GFO después de 2006-08, CFOSAT antes de 2020-02). No cambia la regla del contrato.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import xarray as xr

from .. import manifiesto
from .exposicion import distancia_km, rumbo, sector_de

URL_AODN = ("https://thredds.aodn.org.au/thredds/fileServer/IMOS/SRS/Surface-Waves/Wave-Wind-Altimetry-DM00/"
            "{mision}/{bloque}/IMOS_SRS-Surface-Waves_MW_{mision}_FV02_{tesela}-DM00.nc")
MISIONES_AODN = ["GEOSAT", "ERS-1", "TOPEX", "ERS-2", "GFO", "JASON-1", "ENVISAT", "JASON-2", "CRYOSAT-2", "HY-2",
                 "SARAL", "JASON-3", "SENTINEL-3A", "HY-2B", "CFOSAT", "SENTINEL-3B", "SENTINEL-6A", "SWOT"]

# QUID 1.5, tabla 3 («List of assimilated satellite data and their period of use»).
ASIMILADAS_QUID = {
    "ERS-1": ("1993-01-01", "1996-06-02"), "TOPEX": ("1993-01-01", "2005-10-08"), "ERS-2": ("1995-05-15", "2011-07-04"),
    "GFO": ("2000-01-07", "2006-08-01"), "JASON-1": ("2002-01-15", "2013-06-21"), "ENVISAT": ("2002-05-14", "2012-04-08"),
    "JASON-2": ("2008-07-04", None), "JASON-3": ("2008-07-04", None), "CRYOSAT-2": ("2010-07-14", None),
    "SARAL": ("2016-03-01", None), "SENTINEL-3A": ("2016-03-01", None), "CFOSAT": ("2020-02-01", None),
}
NO_ASIMILADAS_QUID = {"HY-2": "QUID 1.5 §I.2 y §IV.1.1: HY-2A, «non-assimilated», usada para validar 2014–2018"}
INICIO_ASIMILACION = pd.Timestamp("1993-01-01")

RADIO_SITIO_KM, RADIO_REGION_KM = 25.0, 200.0
COSTA_MIN_REGION_KM, PROF_MIN_M = 10.0, 100.0
CORTE_PASADA_S = 60.0
MIN_COLOCALIZACIONES_P90 = 30


def teselas(lat0: float, lon0: float, radio_km: float = RADIO_REGION_KM) -> list[str]:
    """Teselas de 1° (nombre por su esquina suroeste, longitud en 0–360 E) del recuadro del círculo."""
    dlat = radio_km / 111.195
    dlon = radio_km / (111.195 * np.cos(np.radians(lat0)))
    lon360 = lon0 % 360
    nombres = []
    for la in range(int(np.floor(lat0 - dlat)), int(np.floor(lat0 + dlat)) + 1):
        for lo in range(int(np.floor(lon360 - dlon)), int(np.floor(lon360 + dlon)) + 1):
            nombres.append(f"{abs(la):03d}{'S' if la < 0 else 'N'}-{lo % 360:03d}E")
    return nombres


def _bloque(tesela: str) -> str:
    la = int(tesela[:3]) * (-1 if tesela[3] == "S" else 1)
    lo = int(tesela[5:8])
    b_la = int(np.floor(la / 20.0) * 20)
    return f"{abs(b_la):03d}{'S' if b_la < 0 else 'N'}_{int(lo // 20 * 20):03d}E"


def _bajar(url: str) -> bytes | None:
    """Punto de sustitución en las pruebas. Devuelve None si la tesela no existe (404)."""
    import requests

    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def descargar(raiz_raw: Path, lat0: float, lon0: float, misiones=MISIONES_AODN) -> tuple[list[Path], list[str]]:
    """Descarga (o verifica) las teselas de cada misión. Devuelve (rutas, teselas inexistentes)."""
    rutas, faltan = [], []
    for m in misiones:
        for t in teselas(lat0, lon0):
            destino = raiz_raw / "altimetria" / "ribal_young" / m / f"IMOS_SRS-Surface-Waves_MW_{m}_FV02_{t}-DM00.nc"
            if destino.exists():
                manifiesto.verificar(raiz_raw, destino)
                rutas.append(destino)
                continue
            url = URL_AODN.format(mision=m, bloque=_bloque(t), tesela=t)
            contenido = _bajar(url)
            if contenido is None:
                faltan.append(f"{m}/{t}")
                continue
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(contenido)
            manifiesto.registrar(raiz_raw, destino, {
                "fuente": "IMOS / AODN", "producto": "Ribal y Young (2019), altimetría calibrada multimisión (DM00, FV02)",
                "mision": m, "tesela": t, "servicio": url, "licencia": "CC BY 4.0",
                "modulo": "peruvian_wave_analysis.sitio.observaciones"})
            rutas.append(destino)
    return rutas, faltan


def leer(rutas: list[Path]) -> pd.DataFrame:
    """Posiciones y horas de las mediciones válidas (bandera de calidad de Hs = 1). No devuelve Hs.

    La banda es Ku en casi todas las misiones y Ka en SARAL; se usa la que trae el archivo.
    """
    partes = []
    for r in rutas:
        with xr.open_dataset(r) as ds:
            banda = next((b for b in ("KU", "KA") if f"SWH_{b}_quality_control" in ds), None)
            if banda is None:
                raise ValueError(f"{r}: no tiene bandera de calidad de Hs en banda Ku ni Ka.")
            ok = np.asarray(ds[f"SWH_{banda}_quality_control"].values) == 1
            partes.append(pd.DataFrame({
                "mision": Path(r).parent.name, "banda": banda, "tiempo": pd.to_datetime(ds["TIME"].values[ok]),
                "lat": ds["LATITUDE"].values[ok].astype(float),
                "lon": (ds["LONGITUDE"].values[ok].astype(float) + 180.0) % 360.0 - 180.0,
                "prof_archivo_m": ds["BOT_DEPTH"].values[ok].astype(float) if "BOT_DEPTH" in ds else np.nan}))
    if not partes:
        return pd.DataFrame(columns=["mision", "tiempo", "lat", "lon", "prof_archivo_m"])
    return pd.concat(partes, ignore_index=True).sort_values(["mision", "tiempo"]).reset_index(drop=True)


def asimilacion(mision: str, t: pd.Timestamp) -> str:
    if t < INICIO_ASIMILACION:
        return "no asimilada"
    if mision in NO_ASIMILADAS_QUID:
        return "no asimilada"
    if mision in ASIMILADAS_QUID:
        return "asimilada"
    return "no verificable"


def asimilacion_por_periodo(mision: str, t: pd.Timestamp) -> str:
    """Sensibilidad: asimilada solo dentro del período de la tabla 3 del QUID (ver docstring del módulo)."""
    if mision in ASIMILADAS_QUID and t >= INICIO_ASIMILACION:
        a, b = ASIMILADAS_QUID[mision]
        dentro = pd.Timestamp(a) <= t and (b is None or t <= pd.Timestamp(b))
        return "asimilada" if dentro else "no asimilada"
    return asimilacion(mision, t)


def regimen(t: pd.Timestamp) -> str:
    return "1980-1992 (sin asimilación ni corrientes)" if t < INICIO_ASIMILACION else "desde 1993 (asimilado)"


@dataclass
class Costa:
    """Tierra en coordenadas locales (km) alrededor del nodo, para medir distancias a la costa."""
    lat0: float
    lon0: float
    geom: object

    @classmethod
    def desde_tierra(cls, tierra, lat0: float, lon0: float, margen_grados: float = 4.0) -> "Costa":
        recorte = shapely.clip_by_rect(tierra, lon0 - margen_grados, lat0 - margen_grados,
                                       lon0 + margen_grados, lat0 + margen_grados)
        k = 111.195
        local = shapely.transform(recorte, lambda xy: np.column_stack(
            [(xy[:, 0] - lon0) * k * np.cos(np.radians(lat0)), (xy[:, 1] - lat0) * k]))
        shapely.prepare(local)
        return cls(lat0, lon0, local)

    def distancia_km(self, lat, lon) -> np.ndarray:
        k = 111.195
        x = (np.asarray(lon, float) - self.lon0) * k * np.cos(np.radians(self.lat0))
        y = (np.asarray(lat, float) - self.lat0) * k
        return shapely.distance(self.geom, shapely.points(x, y))


def clasificar(df: pd.DataFrame, lat0: float, lon0: float, costa: Costa, prof_gebco, sectores: pd.DataFrame) -> pd.DataFrame:
    """Clase de cada medición (sitio, región o cuenca), independencia y régimen. ``prof_gebco(lat, lon)`` recibe
    arreglos y devuelve la profundidad positiva en m (o NaN fuera del recorte; ahí se usa la del archivo, que solo
    se admite a más de 200 km del nodo: lo verifica el gate de la ficha)."""
    d = df.copy()
    d["dist_nodo_km"] = distancia_km(lat0, lon0, d["lat"], d["lon"])
    d["dist_costa_km"] = costa.distancia_km(d["lat"], d["lon"])
    la, lo = d["lat"].to_numpy(float), d["lon"].to_numpy(float)
    pg = np.broadcast_to(np.asarray(prof_gebco(la, lo), float), la.shape).copy() if len(d) else np.array([])
    d["prof_m"] = np.where(np.isfinite(pg), pg, -d["prof_archivo_m"].astype(float))
    d["fuente_prof"] = np.where(np.isfinite(pg), "GEBCO_2026", "archivo (SRTM30+)")
    clase_sector = sectores.set_index("sector")["clase"]
    d["clase_sector"] = sector_de(rumbo(lat0, lon0, d["lat"], d["lon"])) if len(d) else []
    d["clase_sector"] = d["clase_sector"].map(clase_sector)
    expuesto = d["clase_sector"].isin(["abierto", "parcial"])
    costa_nodo = float(np.asarray(costa.distancia_km([lat0], [lon0]))[0])  # N2: mismo código y misma costa
    hondo = d["prof_m"] > PROF_MIN_M
    sitio = (d["dist_nodo_km"] <= RADIO_SITIO_KM) & (d["dist_costa_km"] >= costa_nodo) & hondo & expuesto
    region = ~sitio & (d["dist_nodo_km"] <= RADIO_REGION_KM) & (d["dist_costa_km"] >= COSTA_MIN_REGION_KM) & hondo & expuesto
    d["clase"] = np.select([sitio, region], ["sitio", "región"], "cuenca")
    d["asimilacion"] = [asimilacion(m, t) for m, t in zip(d["mision"], d["tiempo"])]
    d["asimilacion_periodo"] = [asimilacion_por_periodo(m, t) for m, t in zip(d["mision"], d["tiempo"])]
    d["regimen"] = [regimen(t) for t in d["tiempo"]]
    d["util_para_sesgo"] = d["asimilacion"] == "no asimilada"
    nuevo = d.groupby("mision")["tiempo"].diff().dt.total_seconds().fillna(np.inf) > CORTE_PASADA_S
    d["pasada"] = nuevo.groupby(d["mision"]).cumsum().astype(int)
    d.attrs["dist_costa_nodo_km"] = costa_nodo
    return d


def resumen(d: pd.DataFrame, columna: str = "asimilacion") -> pd.DataFrame:
    """Una fila por misión, régimen e independencia: período, pasadas y mediciones por clase. ``columna`` elige
    la regla de independencia: la del contrato (``asimilacion``) o la sensibilidad (``asimilacion_periodo``)."""
    filas = []
    for (m, reg, asim), g in d.groupby(["mision", "regimen", columna]):
        fila = {"mision": m, "regimen": reg, "asimilacion": asim, "util_para_sesgo": asim == "no asimilada",
                "inicio": g["tiempo"].min(), "fin": g["tiempo"].max()}
        for c in ("sitio", "región", "cuenca"):
            gc = g[g["clase"] == c]
            fila[f"pasadas_{c}"] = int(gc[["mision", "pasada"]].drop_duplicates().shape[0])
            fila[f"mediciones_{c}"] = int(len(gc))
        filas.append(fila)
    return pd.DataFrame(filas)


def cobertura(res: pd.DataFrame) -> pd.DataFrame:
    """Cobertura para la cola por régimen (N1), solo con misiones no asimiladas: N, 0.10·N, 0.01·N y si es útil."""
    filas = []
    for reg in ["1980-1992 (sin asimilación ni corrientes)", "desde 1993 (asimilado)"]:
        g = res[(res["regimen"] == reg) & res["util_para_sesgo"]] if len(res) else res
        n_sitio = int(g["pasadas_sitio"].sum()) if len(g) else 0
        n_region = int(g["pasadas_región"].sum()) if len(g) else 0
        filas.append({"regimen": reg, "pasadas_sitio_N": n_sitio, "esperadas_sobre_P90": 0.10 * n_sitio,
                      "esperadas_sobre_P99": 0.01 * n_sitio,
                      "cobertura_util_sitio": 0.10 * n_sitio >= MIN_COLOCALIZACIONES_P90,
                      "pasadas_region": n_region, "esperadas_sobre_P90_region": 0.10 * n_region})
    return pd.DataFrame(filas)


def regla_respaldo(cob_principal: pd.DataFrame, cob_respaldo: pd.DataFrame, regimen_f4: str) -> str:
    """Contrato §3.4: se pasa al respaldo solo si él tiene cobertura útil de clase «sitio» y el principal no."""
    p = bool(cob_principal.set_index("regimen").loc[regimen_f4, "cobertura_util_sitio"])
    r = bool(cob_respaldo.set_index("regimen").loc[regimen_f4, "cobertura_util_sitio"])
    return "pasar al respaldo (requiere ADR)" if (r and not p) else "mantener el nodo"
