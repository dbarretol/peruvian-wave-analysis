"""Zonas de exclusión y de colisión de usos (analysis/04 §3.3; ADR-0010 y ADR-0012).

Cada capa se pide a su servicio ArcGIS REST con el recuadro del sitio, paginando, en GeoJSON.
El archivo se guarda en ``raw/zonas/`` con la fecha de acceso en el nombre y se registra en
el manifiesto. Una capa que falla queda como **no evaluada**; nunca como «sin zonas».
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from requests.adapters import HTTPAdapter
import shapely
from shapely import make_valid
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from .. import manifiesto, rutas
from .batimetria import _fmt

SERNANP = "https://geoservicios.sernanp.gob.pe/arcgis/rest/services"
GEOCATMIN = "https://geocatmin.ingemmet.gob.pe/arcgis/rest/services"
EXCLUSION, COLISION, IGNORADA, DESCARTADA = "exclusion", "colision", "ignorada", "descartada"
# IGNORADA: no aporta (p. ej. zonificación que no es de protección estricta; la reserva ya cuenta).
# DESCARTADA: polígono sin respaldo verificable; se quita del mapa y se anota en los metadatos.
ANP_USO_INDIRECTO = {"parque nacional", "santuario nacional", "santuario historico", "santuario histórico"}


def _anp_definitiva(p: dict) -> str:
    return EXCLUSION if str(p.get("anp_cate", "")).strip().lower() in ANP_USO_INDIRECTO else COLISION


def _lote(p: dict) -> str:
    """Lotes con contrato = colisión. Sin decreto supremo ni tipo de contrato = descartado.

    Esos polígonos (LIV, LV, LVI, LVII de Repsol y XLV de Global Petroleum, firmados en 2017 frente a
    Ica) no tienen ninguna referencia pública; por su numeración romana parecen áreas de convenios de
    evaluación técnica, que no otorgan derechos y suelen durar 24 meses. Decisión del usuario, 2026-11-14.
    """
    q = {str(k).lower(): v for k, v in p.items()}
    con_contrato = any(str(q.get(c) or "").strip() for c in ("dec_suprem", "tipo_contr"))
    return COLISION if con_contrato else DESCARTADA


def _zonificacion(p: dict) -> str:
    return EXCLUSION if str(p.get("tz_cod", "")).strip().upper() == "PE" else IGNORADA


@dataclass(frozen=True)
class Capa:
    clave: str
    titulo: str
    entidad: str
    url: str
    campos_nombre: tuple[str, ...]
    clasificar: Callable[[dict], str]


def _fija(clase: str) -> Callable[[dict], str]:
    return lambda _p: clase


# Clasificación: ADR-0012 (propuesta).
CAPAS: tuple[Capa, ...] = (
    Capa("anp_definitiva", "ANP nacionales definitivas", "SERNANP",
         f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/0", ("anp_nomb", "anp_sect"), _anp_definitiva),
    Capa("zonificacion_anp", "Zonificación de ANP", "SERNANP",
         f"{SERNANP}/gestion_de_anp/peru_sernanp_0213/MapServer/0", ("anp_nomb", "tz_nomb", "zanp_sect"), _zonificacion),
    Capa("zonificacion_acr", "Zonificación de ACR", "SERNANP",
         f"{SERNANP}/gestion_de_anp/peru_sernanp_0213/MapServer/1", ("acr_nomb", "tz_nomb", "zacr_sect"), _zonificacion),
    Capa("anp_transitoria", "Zonas reservadas (ANP transitorias)", "SERNANP",
         f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/1", ("zr_nomb", "anp_sect"), _fija(EXCLUSION)),
    Capa("acr", "Áreas de conservación regional", "SERNANP",
         f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/2", ("acr_nomb", "acr_sect"), _fija(COLISION)),
    Capa("acp", "Áreas de conservación privada", "SERNANP",
         f"{SERNANP}/base_fisica/peru_sernanp_0102/MapServer/3", ("acp_nomb",), _fija(COLISION)),
    Capa("amortiguamiento", "Zonas de amortiguamiento", "SERNANP",
         f"{SERNANP}/gestion_de_anp/peru_sernanp_0214/MapServer/0", ("anp_nomb",), _fija(COLISION)),
    Capa("aavc_marina", "Áreas de alto valor para la conservación marina", "SERNANP",
         f"{SERNANP}/sernanp_visor/servicio_descarga/MapServer/6", ("aavc_nomb",), _fija(COLISION)),
    Capa("propuesta_anp", "Propuestas de ANP", "SERNANP",
         f"{SERNANP}/servicios_ogc/peru_sernanp_0103/MapServer/0", ("p_nomb",), _fija(COLISION)),
    Capa("lotes_contrato", "Lotes de hidrocarburos con contrato", "PERUPETRO vía GEOCATMIN-INGEMMET",
         f"{GEOCATMIN}/SERV_OTRAS_FUENTES/MapServer/37", ("nomb_lote", "nom_lote", "cia"), _lote),
    # La capa 35 («Hidrocarburos PERUPETRO») es un grupo de 36 y 37: no se consulta directamente.
    Capa("pozos_petroleros", "Pozos petroleros", "PERUPETRO vía GEOCATMIN-INGEMMET",
         f"{GEOCATMIN}/SERV_OTRAS_FUENTES/MapServer/36", ("nombre", "pozo", "nom_pozo", "lote"), _fija(EXCLUSION)),
)

# Familia de cada capa: decide el color en el mapa (no la clase, que decide ADR-0012).
FAMILIAS = {
    "areas_protegidas": "Áreas protegidas (SERNANP)",
    "amortiguamiento": "Zona de amortiguamiento (SERNANP)",
    "aavc": "Alto valor para la conservación marina (SERNANP)",
    "propuesta": "Propuesta de ANP (SERNANP)",
    "petroleo": "Hidrocarburos (PERUPETRO)",
}
FAMILIA_DE_CAPA = {
    "anp_definitiva": "areas_protegidas", "zonificacion_anp": "areas_protegidas", "zonificacion_acr": "areas_protegidas",
    "anp_transitoria": "areas_protegidas", "acr": "areas_protegidas", "acp": "areas_protegidas",
    "amortiguamiento": "amortiguamiento", "aavc_marina": "aavc", "propuesta_anp": "propuesta",
    "lotes_contrato": "petroleo", "pozos_petroleros": "petroleo",
}

NO_CUBIERTAS = (
    "zonas militares", "dispositivos de tráfico y fondeaderos", "cables y ductos",
    "franja de 5 millas", "caladeros", "acuicultura",
)


@dataclass
class Zona:
    capa: str
    nombre: str
    clase: str
    geometria: object  # shapely, lon/lat
    familia: str = "areas_protegidas"  # para la simbología del mapa (FAMILIAS)


@dataclass
class Zonas:
    zonas: list[Zona] = field(default_factory=list)
    capas: dict[str, dict] = field(default_factory=dict)  # clave → {titulo, estado, n, ruta, fecha, error}

    @property
    def no_evaluadas(self) -> list[str]:
        return [c["titulo"] for c in self.capas.values() if c["estado"] != "ok"]

    def de_clase(self, clase: str) -> list[Zona]:
        return [z for z in self.zonas if z.clase == clase]


# --- Descarga -----------------------------------------------------------------------------

class _AdaptadorSistema(HTTPAdapter):
    """Verifica TLS con el almacén de certificados del sistema operativo (``truststore``).

    Algunos servidores (GEOCATMIN) no envían la cadena intermedia completa; el sistema la
    completa y certifi no. La verificación sigue activa.
    """

    def init_poolmanager(self, *args, **kwargs):
        import ssl

        import truststore

        kwargs["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return super().init_poolmanager(*args, **kwargs)


def _sesion() -> requests.Session:
    s = requests.Session()
    s.mount("https://", _AdaptadorSistema())
    return s


def _pedir(sesion, url: str, params: dict) -> dict:
    r = sesion.get(f"{url}/query", params=params, timeout=180)
    r.raise_for_status()
    datos = r.json()
    if "error" in datos:
        raise RuntimeError(f"El servicio respondió con error: {datos['error']}")
    return datos


def consultar_arcgis(url: str, recuadro: tuple[float, float, float, float], lote: int = 200) -> dict:
    """Polígonos de una capa ArcGIS REST que tocan el recuadro, como FeatureCollection GeoJSON.

    Primero pide los identificadores (sin límite de registros) y después las geometrías por lotes
    de identificadores: funciona aunque el servidor no admita paginación (GEOCATMIN) y no se corta
    en el máximo de registros por respuesta.
    """
    o, e, s, n = recuadro
    sesion = _sesion()
    filtro = {"geometry": f"{o},{s},{e},{n}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
              "spatialRel": "esriSpatialRelIntersects", "where": "1=1"}
    ids = sorted(_pedir(sesion, url, {**filtro, "returnIdsOnly": "true", "f": "json"}).get("objectIds") or [])
    features = []
    for k in range(0, len(ids), lote):
        bloque = ids[k:k + lote]
        datos = _pedir(sesion, url, {"objectIds": ",".join(map(str, bloque)), "outFields": "*",
                                     "returnGeometry": "true", "outSR": 4326, "f": "geojson"})
        if datos.get("type") != "FeatureCollection":
            raise RuntimeError("La respuesta no es un FeatureCollection GeoJSON.")
        recibidos = datos.get("features", [])
        if len(recibidos) != len(bloque):
            raise RuntimeError(f"Se pidieron {len(bloque)} polígonos y llegaron {len(recibidos)}.")
        features += recibidos
    return {"type": "FeatureCollection", "features": features}


_consultar = consultar_arcgis  # punto de sustitución en las pruebas


def _descargar_capa(capa: Capa, recuadro, raiz_raw: Path, fecha: str) -> tuple[Path, dict]:
    o, e, s, n = recuadro
    destino = raiz_raw / "zonas" / capa.clave / f"{capa.clave}_{fecha}_lat{_fmt(s)}_{_fmt(n)}_lon{_fmt(o)}_{_fmt(e)}.geojson"
    if destino.exists():
        return destino, manifiesto.verificar(raiz_raw, destino)
    gj = _consultar(capa.url, recuadro)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    origen = {
        "fuente": capa.entidad, "capa": capa.titulo, "servicio": capa.url,
        "parametros": {"recuadro_oeste_este_sur_norte": list(recuadro), "spatialRel": "esriSpatialRelIntersects", "f": "geojson"},
        "n_poligonos": len(gj["features"]),
        "ultima_actualizacion_declarada": "el servicio no la declara",
        "modulo": "peruvian_wave_analysis.sitio.zonas",
    }
    return destino, manifiesto.registrar(raiz_raw, destino, origen)


def _nombre(props: dict, campos: tuple[str, ...]) -> str:
    """Nombre legible a partir de los campos indicados (sin distinguir mayúsculas); si no hay, el primer campo *nomb*."""
    p = {str(k).lower(): v for k, v in props.items()}
    partes = [str(p[c]).strip() for c in campos if p.get(c) is not None and str(p[c]).strip()]
    if not partes:
        partes = [str(v).strip() for k, v in p.items() if "nomb" in k and isinstance(v, str) and v.strip()][:1]
    return " · ".join(dict.fromkeys(partes)) or "sin nombre"


def leer_capa(capa: Capa, ruta: Path) -> tuple[list[Zona], list[str]]:
    """Zonas de la capa y nombres de los polígonos descartados (DESCARTADA)."""
    gj = json.loads(Path(ruta).read_text(encoding="utf-8"))
    zonas, descartadas = [], []
    for f in gj.get("features", []):
        if not f.get("geometry"):
            continue
        props = f.get("properties") or {}
        clase = capa.clasificar(props)
        if clase == IGNORADA:
            continue
        if clase == DESCARTADA:
            descartadas.append(_nombre(props, capa.campos_nombre))
            continue
        geom = make_valid(shape(f["geometry"]))
        if geom.is_empty:
            continue
        nombre = f"{capa.titulo}: {_nombre(props, capa.campos_nombre)}"
        zonas.append(Zona(capa=capa.titulo, nombre=nombre, clase=clase, geometria=geom,
                          familia=FAMILIA_DE_CAPA[capa.clave]))
    return zonas, descartadas


def obtener(recuadro: tuple[float, float, float, float], raiz_datos: Path, capas: tuple[Capa, ...] = CAPAS) -> Zonas:
    """Descarga (o reutiliza las del mismo día) y lee todas las capas para el recuadro."""
    raiz_raw = rutas.raw(raiz_datos)
    fecha = datetime.now(timezone.utc).strftime("%Y%m%d")
    res = Zonas()
    for capa in capas:
        try:
            ruta, origen = _descargar_capa(capa, recuadro, raiz_raw, fecha)
            zs, descartadas = leer_capa(capa, ruta)
            res.zonas += zs
            res.capas[capa.clave] = {"titulo": capa.titulo, "estado": "ok", "n_poligonos": origen.get("n_poligonos"),
                                     "descartadas": descartadas, "ruta": str(ruta), "sha256": origen["sha256"],
                                     "fecha_acceso_utc": origen["fecha_acceso_utc"]}
        except Exception as exc:  # noqa: BLE001 — una capa que falla se declara, no se oculta
            res.capas[capa.clave] = {"titulo": capa.titulo, "estado": "no evaluada", "error": str(exc)[:300]}
    return res


def desde_registro(capas_registradas: dict[str, dict], raiz_datos: Path) -> Zonas:
    """Relee las capas de una corrida anterior verificando su hash (sin descargar). Para el mapa definitivo."""
    raiz_raw = rutas.raw(raiz_datos)
    por_clave = {c.clave: c for c in CAPAS}
    res = Zonas()
    for clave, info in capas_registradas.items():
        if info["estado"] != "ok":
            res.capas[clave] = dict(info)
            continue
        ruta = Path(info["ruta"])
        manifiesto.verificar(raiz_raw, ruta)
        zs, descartadas = leer_capa(por_clave[clave], ruta)
        res.zonas += zs
        info = dict(info, descartadas=descartadas)
        res.capas[clave] = dict(info)
    return res


# --- Evaluación por celda -------------------------------------------------------------------

def transform(fn, geom):
    """Aplica ``fn(x, y) -> (x, y)`` a las coordenadas de ``geom``."""
    return shapely.transform(geom, fn, interleaved=False)


def _a_km(lon0: float, lat0: float):
    from .candidatos import RADIO_TIERRA_KM
    k = RADIO_TIERRA_KM * np.pi / 180
    c = np.cos(np.radians(lat0))
    return lambda x, y, z=None: ((np.asarray(x) - lon0) * k * c, (np.asarray(y) - lat0) * k)


@dataclass
class EvaluacionZonas:
    excluido: bool
    zonas_exclusion: list[str]
    zonas_colision: list[str]
    dist_exclusion_km: float


class Evaluador:
    """Evalúa celdas de la malla contra las zonas, en km de una proyección local."""

    def __init__(self, zonas: Zonas, lon0: float, lat0: float, resguardo_km: float):
        self._a_km = _a_km(lon0, lat0)
        self.resguardo_km = resguardo_km
        self._excl = [(z.nombre, transform(self._a_km, z.geometria)) for z in zonas.de_clase(EXCLUSION)]
        self._excl_amp = [(n, g.buffer(resguardo_km)) for n, g in self._excl]
        self._col = [(z.nombre, transform(self._a_km, z.geometria)) for z in zonas.de_clase(COLISION)]
        self._union_excl = unary_union([g for _, g in self._excl]) if self._excl else None

    def evaluar(self, lat: float, lon: float, paso: float) -> EvaluacionZonas:
        h = paso / 2
        celda = transform(self._a_km, box(lon - h, lat - h, lon + h, lat + h))
        excl = sorted({n for n, g in self._excl_amp if g.intersects(celda)})
        col = sorted({n for n, g in self._col if g.intersects(celda)})
        if self._union_excl is None:
            d = np.inf
        else:
            punto = transform(self._a_km, Point(lon, lat))
            d = float(self._union_excl.distance(punto))
        return EvaluacionZonas(excluido=bool(excl), zonas_exclusion=excl, zonas_colision=col, dist_exclusion_km=d)


def _de_km(lon0: float, lat0: float):
    from .candidatos import RADIO_TIERRA_KM
    k = RADIO_TIERRA_KM * np.pi / 180
    c = np.cos(np.radians(lat0))
    return lambda x, y, z=None: (np.asarray(x) / (k * c) + lon0, np.asarray(y) / k + lat0)


def exclusion_ampliada(zonas: Zonas, lon0: float, lat0: float, resguardo_km: float):
    """Unión de las zonas de exclusión ampliadas en ``resguardo_km``, en lon/lat (para dibujar)."""
    a_km = _a_km(lon0, lat0)
    geoms = [transform(a_km, z.geometria).buffer(resguardo_km) for z in zonas.de_clase(EXCLUSION)]
    return transform(_de_km(lon0, lat0), unary_union(geoms)) if geoms else None
