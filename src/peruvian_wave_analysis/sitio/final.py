"""Mapa definitivo del sitio (ADR-0011): GEBCO e insumos ya guardados, sin descargas.

Recibe la corrida exploratoria (su ``metadatos.json``), el nodo elegido y el recorte de GEBCO
registrado. Verifica el hash de cada insumo, recalcula la tabla con GEBCO y exige que el nodo
elegido cumpla los criterios de ADR-0014 (gate: nodo con dato y fuera de exclusión). Si el gate falla, borra la salida
publicada anterior y termina con error.

Salida en ``WAVE-DATA/processed/F01-sitio-y-fuentes/mapa-sitio-final/<nombre>/``: ``mapa.png`` (contexto
regional, solo el nodo elegido), ``detalle_nodo.png`` (su celda y las ocho vecinas, batimetría a resolución
completa), ``nodos.csv`` (todos los nodos con GEBCO, sin numeración de candidatos), ``ficha_nodo.json``
(nodo y vecindario de 3 × 3) y ``MANIFEST.json``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .. import __version__, manifiesto, rutas
from . import batimetria, candidatos, waverys, zonas as zonas_mod
from .config import ConfigMapa
from .gebco import FUENTE_GEBCO
from .detalle import dibujar_detalle
from .mapa import FASE, ResultadoMapa, dibujar

TOL_COBERTURA = 0.01  # grados


class ErrorGate(RuntimeError):
    """El nodo elegido no cumple los criterios o falta un insumo."""


@dataclass
class ConfigFinal:
    exploratorio: str  # ruta al metadatos.json de la corrida exploratoria
    gebco: str  # ruta del recorte de GEBCO dentro de WAVE-DATA/raw (registrado con registrar-gebco)
    nodo_lat: float
    nodo_lon: float
    nombre: str | None = None  # None: el de la corrida exploratoria
    directorio_datos: str | None = None


def _cargar_exploratorio(cf: ConfigFinal) -> dict:
    p = Path(cf.exploratorio)
    if not p.exists():
        raise ErrorGate(f"No existe la corrida exploratoria {p}.")
    meta = json.loads(p.read_text(encoding="utf-8"))
    if meta.get("tipo") != "mapa exploratorio":
        raise ErrorGate(f"{p} no es el metadatos.json de un mapa exploratorio.")
    if not meta.get("zonas"):
        raise ErrorGate("La corrida exploratoria no evaluó zonas: repetirla con zonas antes del mapa definitivo.")
    return meta


def _verificar_gebco(ruta: Path, raiz: Path, cfg: ConfigMapa) -> dict:
    raiz_raw = rutas.raw(raiz)
    try:
        ruta.resolve().relative_to(raiz_raw.resolve())
    except ValueError as exc:
        raise ErrorGate(f"{ruta} no está en {raiz_raw}: registrarlo antes con `registrar-gebco`.") from exc
    if not ruta.exists():
        raise ErrorGate(f"No existe {ruta}.")
    entrada = manifiesto.verificar(raiz_raw, ruta)
    if entrada.get("fuente") != FUENTE_GEBCO:
        raise ErrorGate(f"{ruta} no está registrado como GEBCO (fuente: {entrada.get('fuente')}).")
    o, e, s, n = batimetria.recuadro_con_margen(cfg)
    with xr.open_dataset(ruta) as ds:
        lat = ds[batimetria._primer_nombre(batimetria._NOMBRES_LAT, ds.coords)].values
        lon = ds[batimetria._primer_nombre(batimetria._NOMBRES_LON, ds.coords)].values
    if (
        lon.min() > o + TOL_COBERTURA
        or lon.max() < e - TOL_COBERTURA
        or lat.min() > s + TOL_COBERTURA
        or lat.max() < n - TOL_COBERTURA
    ):
        raise ErrorGate(
            f"El recorte de GEBCO ({lon.min():.2f}…{lon.max():.2f}, {lat.min():.2f}…{lat.max():.2f}) no cubre "
            f"el recuadro necesario (O {o}, E {e}, S {s}, N {n})."
        )
    return entrada


def _publicar(destino: Path, archivos: dict[str, Path], insumos: dict) -> None:
    """Copia las salidas desde staging, con un MANIFEST.json de salidas e insumos."""
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    salidas = {}
    for nombre, p in archivos.items():
        shutil.copy2(p, destino / p.name)
        salidas[p.name] = manifiesto.sha256(destino / p.name)
    (destino / "MANIFEST.json").write_text(
        json.dumps(
            {"version_codigo": __version__, "salidas_sha256": salidas, "insumos": insumos},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def generar_mapa_final(cf: ConfigFinal) -> ResultadoMapa:
    meta = _cargar_exploratorio(cf)
    datos_cfg = dict(meta["configuracion"])
    if cf.directorio_datos:
        datos_cfg["directorio_datos"] = cf.directorio_datos
    if cf.nombre:
        datos_cfg["nombre"] = cf.nombre
    raiz = rutas.directorio_datos(datos_cfg.get("directorio_datos"))
    destino = Path(raiz) / "processed" / FASE / "mapa-sitio-final" / datos_cfg["nombre"]

    ruta_wav = meta["insumos"]["waverys"]["ruta"]
    datos_cfg["batimetria"] = {"fuente": "archivo", "archivo": str(cf.gebco)}
    datos_cfg["waverys"] = dict(datos_cfg["waverys"], fuente="archivo", archivo=ruta_wav)
    cfg = ConfigMapa(**datos_cfg)

    try:
        entrada_gebco = _verificar_gebco(Path(cf.gebco), raiz, cfg)
        bat = batimetria.obtener(cfg, raiz)
        bat.etiqueta = (
            f"Batimetría: {entrada_gebco.get('producto', 'GEBCO')},\n"
            f"  descarga {entrada_gebco['fecha_acceso_utc'][:10]}"
        )
        malla = waverys.obtener(cfg, raiz)  # verifica el hash: el archivo está en raw
        zonas = zonas_mod.desde_registro(meta["zonas"], raiz)
        tabla, _ = candidatos.tabla(cfg, bat.elevacion, malla, zonas)

        elegido = tabla[
            np.isclose(tabla["lat"], cf.nodo_lat, atol=1e-3) & np.isclose(tabla["lon"], cf.nodo_lon, atol=1e-3)
        ]
        if elegido.empty:
            raise ErrorGate(f"({cf.nodo_lat}, {cf.nodo_lon}) no es un nodo de WAVERYS dentro del recuadro.")
        fila = elegido.iloc[0]
        if not fila["cumple_criterios"]:
            raise ErrorGate(f"El nodo elegido no cumple los criterios con GEBCO: {fila['motivo_descarte']}.")
    except Exception:
        if destino.exists():  # un gate fallido invalida la salida publicada anterior (roadmap §8.1)
            shutil.rmtree(destino)
        raise

    # Ya hay un nodo elegido: el definitivo no numera candidatos (ni en el mapa ni en la tabla).
    tabla = tabla.drop(columns=["candidato"])
    fila = fila.drop(labels=["candidato"])
    fig = dibujar(
        cfg, bat, malla, tabla, zonas, modo="definitivo", nodo_elegido=(float(fila["lat"]), float(fila["lon"]))
    )
    nodo = {k: (v.item() if hasattr(v, "item") else v) for k, v in fila.items()}
    fig_detalle, vecinas = dibujar_detalle(cfg, bat, malla, zonas, nodo)
    res = ResultadoMapa(figura=fig, tabla=tabla, batimetria=bat, malla=malla, zonas=zonas)

    ficha = {
        "nodo_elegido": nodo,
        "decision": "ADR-0007",
        "vecindario_3x3": vecinas,  # de norte a sur y de oeste a este; incluye el nodo elegido
        "criterios": {
            "resguardo_km": cfg.resguardo_km,
            "isobata_principal_m": cfg.isobata_principal,
            "min_vecinas": cfg.min_vecinas,
            "max_tierra": cfg.max_tierra,
            "adr": ["ADR-0007", "ADR-0011", "ADR-0012", "ADR-0014"],
        },
        "no_evaluado": zonas.no_evaluadas + list(zonas_mod.NO_CUBIERTAS),
        "corrida_exploratoria": str(Path(cf.exploratorio)),
    }
    insumos = {
        "batimetria": bat.origen,
        "waverys": malla.origen,
        "zonas": zonas.capas,
        "corrida_exploratoria": {"ruta": str(cf.exploratorio), "sha256": manifiesto.sha256(Path(cf.exploratorio))},
    }

    with tempfile.TemporaryDirectory() as tmp:  # staging
        t = Path(tmp)
        for f, nombre in ((fig, "mapa.png"), (fig_detalle, "detalle_nodo.png")):
            f.savefig(
                t / nombre,
                facecolor=f.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.15,
                metadata={"Software": None},
            )
        tabla.to_csv(t / "nodos.csv", index=False, float_format="%.4f", encoding="utf-8")
        (t / "ficha_nodo.json").write_text(
            json.dumps(ficha, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        _publicar(
            destino,
            {
                "mapa": t / "mapa.png",
                "detalle": t / "detalle_nodo.png",
                "tabla": t / "nodos.csv",
                "ficha": t / "ficha_nodo.json",
            },
            insumos,
        )
    res.salidas = {
        "mapa": destino / "mapa.png",
        "detalle": destino / "detalle_nodo.png",
        "tabla": destino / "nodos.csv",
        "ficha": destino / "ficha_nodo.json",
        "manifiesto": destino / "MANIFEST.json",
    }
    res.figura_detalle = fig_detalle
    return res
