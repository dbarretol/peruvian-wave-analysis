"""Boyas NDBC 32302 y 32012: descarga de los archivos estándar (``stdmet``) y lector de los tres formatos.

Contrato de la F5 (v6): §1 (insumos), §3.2–§3.3 (boyas) y §8 (prueba del descargador, G4).

Formatos de ``stdmet`` que el lector reconoce por la primera línea:
- ``YY MM DD hh ...``: hasta 1998. Año con dos dígitos (19YY), sin minutos. Columnas ``WD`` y ``BAR``.
- ``YYYY MM DD hh ...``: 1999–2004. Año con cuatro dígitos, sin minutos.
- ``YYYY MM DD hh mm ...``: 2005–2006. Con minutos.
- ``#YY MM DD hh mm ...`` y una segunda línea de unidades (``#yr mo dy hr mn ...``): desde 2007.
Las columnas se nombran con la grafía actual (``WD`` → ``WDIR``, ``BAR`` → ``PRES``).

**G4 (contrato §2 y §8).** Las funciones públicas no devuelven valores observados:
- ``estructura``: formato, columnas, número de registros y fechas extremas;
- ``presencia``: el tiempo y una máscara booleana de «dato presente» (sin centinela), nunca el valor.
Los mensajes de error no citan el contenido del archivo. ``_leer`` devuelve los valores y queda para la etapa 3.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import manifiesto

URL_STDMET = "https://www.ndbc.noaa.gov/data/historical/stdmet/{estacion}h{anio}.txt.gz"
AGENTE = "Mozilla/5.0 (compatible; research-client)"  # NDBC responde 403 sin un agente de navegador
CENTINELAS = (99.0, 999.0, 9999.0)  # valores de dato ausente de NDBC (contrato §3.3)
CENTINELAS_TEXTO = frozenset({"MM", "99", "99.0", "99.00", "999", "999.0", "9999", "9999.0"})  # igual, como texto
_NUMERO = r"-?\d+(?:\.\d+)?|MM"  # forma de un campo de datos válido
VARIABLES_F5 = ("WVHT", "DPD", "APD")
_ALIAS = {"WD": "WDIR", "BAR": "PRES"}
BORDE = pd.Timedelta(hours=1)  # tolerancia en los bordes del año (registros de las hh:50)
_TIEMPO = ("YY", "YYYY", "MM", "DD", "hh", "mm")


@dataclass(frozen=True)
class Boya:
    estacion: str
    lat: float
    lon: float
    anios: tuple[int, ...]
    descripcion: str


# Posiciones de las páginas de estación de NDBC guardadas el 2026-11-17 (raw/documentos/ndbc/); años del listado
# histórico de stdmet del mismo día. 32012 es un fondeo con recambios: la posición es la última publicada.
BOYAS = (
    Boya("32302", -18.0, -85.1, tuple(range(1986, 1996)), "NDBC 32302 (atribución de C1)"),
    Boya("32012", -19.425, -85.078, tuple(range(2007, 2019)), "NDBC 32012, Woods Hole Stratus (atribución de C2)"),
)


class ErrorNDBC(RuntimeError):
    """Archivo de NDBC ausente, con formato no reconocido o inconsistente. El mensaje no cita valores."""


# --- Descarga ----------------------------------------------------------------------------------------


def _bajar(url: str) -> bytes | None:
    """Punto de sustitución en las pruebas. None si el archivo no existe (404)."""
    import requests  # noqa: PLC0415 (solo hace falta al descargar)

    r = requests.get(url, timeout=120, headers={"User-Agent": AGENTE})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def ruta_archivo(raiz_raw: Path, estacion: str, anio: int) -> Path:
    return Path(raiz_raw) / "boyas" / "ndbc" / estacion / f"{estacion}h{anio}.txt.gz"


def descargar(raiz_raw: Path, boya: Boya, anios: list[int] | None = None) -> tuple[list[Path], list[int]]:
    """Descarga (o verifica por hash) los archivos anuales. Devuelve (rutas, años que NDBC no tiene).

    Un archivo ya presente se verifica contra el manifiesto y no se vuelve a pedir. Un archivo nuevo se comprueba
    (formato y años) antes de registrarlo.
    """
    rutas, faltan = [], []
    for anio in anios or boya.anios:
        destino = ruta_archivo(raiz_raw, boya.estacion, anio)
        if destino.exists():
            manifiesto.verificar(raiz_raw, destino)
            rutas.append(destino)
            continue
        url = URL_STDMET.format(estacion=boya.estacion, anio=anio)
        contenido = _bajar(url)
        if contenido is None:
            faltan.append(anio)
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_name(destino.name + ".parcial")
        temporal.write_bytes(contenido)
        try:
            est = estructura(temporal, anio_esperado=anio)
        except ErrorNDBC:
            temporal.unlink()
            raise
        temporal.replace(destino)
        manifiesto.registrar(
            raiz_raw,
            destino,
            {
                "fuente": "NOAA National Data Buoy Center (NDBC)",
                "producto": "Datos meteorológicos y de oleaje estándar (stdmet), archivo histórico anual",
                "estacion": boya.estacion,
                "posicion_pagina_estacion": [boya.lat, boya.lon],
                "anio": anio,
                "servicio": url,
                "formato": est["formato"],
                "columnas": est["columnas"],
                "licencia": "Datos del Gobierno de EE. UU., dominio público (sin verificar)",
                "modulo": "peruvian_wave_analysis.validacion.ndbc",
            },
        )
        rutas.append(destino)
    return rutas, faltan


def comparar_hashes(rutas: list[Path], directorio: Path) -> dict[str, str]:
    """Compara el sha256 de cada archivo con el del mismo nombre en otro directorio (solo bytes; no se abre nada).

    Devuelve, por nombre: «igual», «distinto» o «ausente». Es un control: la fuente sigue siendo NDBC (ADR-0019).
    """
    res = {}
    for r in rutas:
        otro = Path(directorio) / Path(r).name
        if not otro.exists():
            res[Path(r).name] = "ausente"
            continue
        h = hashlib.sha256(otro.read_bytes()).hexdigest()
        res[Path(r).name] = "igual" if h == manifiesto.sha256(r) else "distinto"
    return res


# --- Lectura ---------------------------------------------------------------------------------------


def _lineas(ruta: Path) -> list[str]:
    ruta = Path(ruta)
    try:
        if ruta.suffix == ".gz" or ruta.name.endswith(".gz.parcial"):
            with gzip.open(ruta, "rt", encoding="ascii", errors="strict") as f:
                return f.read().splitlines()
        return ruta.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError, EOFError) as exc:
        raise ErrorNDBC(f"{ruta.name}: no se puede leer como texto ({type(exc).__name__}).") from None


def _encabezado(lineas: list[str], nombre: str) -> tuple[list[str], int, str]:
    """(columnas normalizadas, líneas de encabezado, formato)."""
    if not lineas:
        raise ErrorNDBC(f"{nombre}: archivo vacío.")
    cols = lineas[0].lstrip("#").split()
    saltar = 1
    if len(lineas) > 1 and lineas[1].startswith("#"):
        saltar = 2  # línea de unidades (desde 2007)
    if not cols or cols[0] not in ("YY", "YYYY") or cols[1:4] != ["MM", "DD", "hh"]:
        raise ErrorNDBC(f"{nombre}: encabezado no reconocido (columnas {cols[:5]}).")
    con_min = len(cols) > 4 and cols[4] == "mm"
    if lineas[0].startswith("#"):
        formato = "#YY con minutos y unidades" if saltar == 2 else "#YY con minutos"
        cols[0] = "YYYY"  # desde 2007 el año va con cuatro dígitos aunque la columna se llame #YY
    elif cols[0] == "YY":
        formato = "YY sin minutos"
    else:
        formato = "YYYY con minutos" if con_min else "YYYY sin minutos"
    cols = [_ALIAS.get(c, c) for c in cols]
    if len(set(cols)) != len(cols):
        raise ErrorNDBC(f"{nombre}: columnas repetidas en el encabezado.")
    return cols, saltar, formato


def _leer_texto(ruta: Path, validar: bool = True) -> tuple[pd.DatetimeIndex, pd.DataFrame, str]:
    """(tiempo, columnas de datos como **texto**, formato). Solo las columnas de tiempo se convierten a número.

    Las columnas de datos no se convierten. Con ``validar``, un campo que no tiene forma de número (comparación de
    texto con una expresión regular) es un error, y el mensaje no lo cita (G4). ``presencia`` no valida: funciona
    aunque los valores sean ilegibles (contrato §8).
    """
    nombre = Path(ruta).name.removesuffix(".parcial")
    lineas = _lineas(ruta)
    cols, saltar, formato = _encabezado(lineas, nombre)
    filas = [ln.split() for ln in lineas[saltar:] if ln.strip()]
    malas = [i + saltar + 1 for i, f in enumerate(filas) if len(f) != len(cols)]
    if malas:
        raise ErrorNDBC(
            f"{nombre}: {len(malas)} líneas con un número de campos distinto del encabezado (primera: {malas[0]})."
        )
    texto = pd.DataFrame(filas, columns=cols, dtype=object) if filas else pd.DataFrame(columns=cols, dtype=object)
    tcols = [c for c in _TIEMPO if c in texto]
    datos = texto.drop(columns=tcols)
    if validar and len(datos) and not datos.apply(lambda c: c.str.fullmatch(_NUMERO)).all().all():
        raise ErrorNDBC(f"{nombre}: hay campos no numéricos.")
    try:
        t = texto[tcols].astype(int)
        anio = t["YYYY"] if "YYYY" in t else 1900 + t["YY"]
        tiempo = pd.to_datetime(
            dict(year=anio, month=t["MM"], day=t["DD"], hour=t["hh"], minute=t["mm"] if "mm" in t else 0)
        )
    except (ValueError, OverflowError):
        raise ErrorNDBC(f"{nombre}: fechas inválidas.") from None
    return pd.DatetimeIndex(tiempo), datos.reset_index(drop=True), formato


def _leer(ruta: Path) -> tuple[pd.DataFrame, str]:
    """Tiempo (UTC, sin zona) y columnas numéricas, centinelas como NaN. **Lee valores:** solo etapa 3 y pruebas."""
    tiempo, datos, formato = _leer_texto(ruta)
    df = datos.replace("MM", np.nan).astype(float)
    df = df.mask(df.isin(CENTINELAS))
    df.insert(0, "tiempo", tiempo.to_numpy())
    return df, formato


def estructura(ruta: Path, anio_esperado: int | None = None) -> dict:
    """Formato, columnas, registros y fechas extremas. **Sin valores observados** (G4).

    Falla si faltan las columnas de la F5 (WVHT, DPD, APD) o si hay registros de un año ajeno al archivo. Se admite
    una hora en cada borde: desde 2009, los archivos de NDBC empiezan con el registro de las 23:50 del 31 de diciembre
    anterior (hora 00 del año del archivo). Esos registros de borde pueden repetirse entre archivos contiguos; los
    duplicados entre archivos se resuelven al unir la serie, no aquí.
    """
    t, datos, formato = _leer_texto(ruta)
    nombre = Path(ruta).name.removesuffix(".parcial")
    faltan = [v for v in VARIABLES_F5 if v not in datos]
    if faltan:
        raise ErrorNDBC(f"{nombre}: faltan las columnas {faltan}.")
    if anio_esperado is not None and len(t):
        desde = pd.Timestamp(f"{anio_esperado}-01-01") - BORDE
        hasta = pd.Timestamp(f"{anio_esperado + 1}-01-01") + BORDE
        ajenos = int(((t < desde) | (t > hasta)).sum())
        if ajenos:
            raise ErrorNDBC(f"{nombre}: {ajenos} registros fuera del año {anio_esperado}.")
    return {
        "archivo": nombre,
        "formato": formato,
        "columnas": list(datos.columns),
        "registros": int(len(datos)),
        "duplicados": int(t.duplicated().sum()),
        "inicio": str(t.min()) if len(t) else None,
        "fin": str(t.max()) if len(t) else None,
    }


def presencia(ruta: Path, variables=VARIABLES_F5) -> pd.DataFrame:
    """Tiempo y, por variable, True si hay dato. **Nunca el valor** (etapa 2; contrato §2).

    El campo se compara **como texto** con los centinelas (``CENTINELAS_TEXTO``); no se convierte a número.
    """
    t, datos, _ = _leer_texto(ruta, validar=False)
    out = pd.DataFrame({"tiempo": t})
    for v in variables:
        out[v] = (~datos[v].isin(CENTINELAS_TEXTO)).to_numpy() if v in datos else False
    return out
