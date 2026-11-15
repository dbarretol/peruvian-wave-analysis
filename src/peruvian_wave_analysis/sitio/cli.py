"""Línea de comandos del mapa de sitio (ADR-0011).

- ``mapa-sitio``: mapa exploratorio (descarga GMRT, malla de WAVERYS y zonas).
- ``registrar-gebco``: registra el recorte de GEBCO descargado a mano.
- ``mapa-sitio-final``: mapa definitivo con GEBCO e insumos ya guardados, sin descargas.

Ejemplos::

    mapa-sitio --nombre marcona --lat -15.4 --lon -75.3 --isobatas 200 100 50
    registrar-gebco --archivo C:/Descargas/gebco_2025_recorte.nc --version GEBCO_2025
    mapa-sitio-final --exploratorio <WAVE-DATA>/interim/.../metadatos.json \\
        --gebco <WAVE-DATA>/raw/batimetria/gebco/gebco_2025_recorte.nc --nodo-lat -15.6 --nodo-lon -75.2
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

from .config import ConfigMapa

_COLUMNAS = ["candidato", "lat", "lon", "prof_nodo_m", "prof_media_celda_m", "dist_costa_km", "zonas_colision"]


def _imprimir(res, cfg: ConfigMapa) -> None:
    import pandas as pd

    cols = _COLUMNAS[:5] + [f"dist_iso_{cfg.isobata_principal:g}m_km"] + _COLUMNAS[5:]
    top = res.tabla[res.tabla["candidato"].notna()]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(top[cols].to_string(index=False) if len(top) else "Ningún nodo cumple los criterios.")
    print("\nMotivos de descarte:")
    print(res.tabla["motivo_descarte"].replace("", "cumple").value_counts().to_string())
    if res.zonas is not None and res.zonas.no_evaluadas:
        print(f"\nCapas NO evaluadas: {', '.join(res.zonas.no_evaluadas)}")
    print()
    for nombre, ruta in res.salidas.items():
        print(f"{nombre}: {ruta}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mapa-sitio", description="Mapa exploratorio: isóbatas, malla de WAVERYS y zonas alrededor de una coordenada.")
    p.add_argument("--config", help="Archivo YAML de configuración. Los demás argumentos lo sobrescriben.")
    p.add_argument("--nombre", help="Nombre corto del sitio (minúsculas; se usa en las rutas de salida).")
    p.add_argument("--lat", type=float, help="Latitud de referencia (°, negativa al sur).")
    p.add_argument("--lon", type=float, help="Longitud de referencia (°, negativa al oeste).")
    p.add_argument("--isobatas", type=float, nargs="+", help="Profundidades de las isóbatas (m). Por defecto 200 100 50.")
    p.add_argument("--isobata-principal", type=float, help="Isóbata con la que se ordenan los candidatos.")
    p.add_argument("--semiancho", type=float, help="Semiancho del recuadro (°). Por defecto 1.")
    p.add_argument("--n-candidatos", type=int, help="Candidatos numerados (ADR-0013: 3).")
    p.add_argument("--resguardo-km", type=float, help="Resguardo alrededor de las zonas de exclusión (ADR-0013: 1 km).")
    p.add_argument("--min-vecinas", type=int, help="Mínimo de vecinas con dato (0 a 8). ADR-0014: 0, no descartan.")
    p.add_argument("--max-tierra", type=float, help="Fracción de tierra tolerada en la celda (0 a 1). ADR-0014: 1, no descarta.")
    p.add_argument("--punto", nargs=3, action="append", metavar=("NOMBRE", "LAT", "LON"),
                   help="Punto que solo se dibuja; acepta grados decimales o 15°23′20″S. Se puede repetir.")
    p.add_argument("--sin-zonas", action="store_true", help="No descarga zonas (el mapa lo declara).")
    p.add_argument("--datos", help="Raíz de WAVE-DATA (si no, variable de entorno WAVE_DATA_DIR).")
    p.add_argument("--mostrar", action="store_true", help="Abre la figura en una ventana además de guardarla.")
    a = p.parse_args(argv)

    base = ConfigMapa.desde_yaml(a.config).como_dict() if a.config else {}
    for clave, valor in {
        "nombre": a.nombre, "latitud": a.lat, "longitud": a.lon, "isobatas": a.isobatas,
        "isobata_principal": a.isobata_principal, "semiancho_grados": a.semiancho,
        "n_candidatos": a.n_candidatos, "resguardo_km": a.resguardo_km, "min_vecinas": a.min_vecinas, "max_tierra": a.max_tierra,
        "directorio_datos": a.datos,
    }.items():
        if valor is not None:
            base[clave] = valor
    if a.isobatas is not None and a.isobata_principal is None:
        base["isobata_principal"] = None
    if a.sin_zonas:
        base["zonas"] = False
    if a.punto:
        base["puntos"] = [{"nombre": n, "latitud": la, "longitud": lo} for n, la, lo in a.punto]
    faltan = [k for k in ("nombre", "latitud", "longitud") if k not in base]
    if faltan:
        print(f"Faltan argumentos: {', '.join(faltan)} (o usa --config).", file=sys.stderr)
        return 2
    if not a.mostrar:
        matplotlib.use("Agg")
    from .mapa import generar_mapa  # después de fijar el backend

    cfg = ConfigMapa(**base)
    res = generar_mapa(cfg)
    _imprimir(res, cfg)
    o, e, s, n = res.recuadro_gebco
    print(f"\nMAPA EXPLORATORIO. Para el mapa definitivo, descargar GEBCO en https://download.gebco.net/ "
          f"(NetCDF) con N {n:.1f}, S {s:.1f}, O {o:.1f}, E {e:.1f}; ver {res.salidas['siguiente_paso']}")
    if a.mostrar:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


def main_registrar_gebco(argv=None) -> int:
    p = argparse.ArgumentParser(prog="registrar-gebco", description="Copia a WAVE-DATA/raw y registra un recorte de GEBCO descargado a mano.")
    p.add_argument("--archivo", required=True, help="NetCDF descargado de https://download.gebco.net/.")
    p.add_argument("--version", required=True, help="Versión de la malla, p. ej. GEBCO_2025.")
    p.add_argument("--datos", help="Raíz de WAVE-DATA (si no, WAVE_DATA_DIR).")
    a = p.parse_args(argv)
    from .. import rutas
    from .gebco import registrar

    ruta, entrada = registrar(a.archivo, rutas.directorio_datos(a.datos), a.version)
    print(f"Registrado: {ruta}\nsha256: {entrada['sha256']}")
    print("Recordatorio: añadir la versión de GEBCO a WAVE-REPO-VAULT/fuentes-de-datos.md.")
    return 0


def main_final(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mapa-sitio-final", description="Mapa definitivo con GEBCO e insumos ya guardados (sin descargas).")
    p.add_argument("--exploratorio", required=True, help="metadatos.json de la corrida exploratoria.")
    p.add_argument("--gebco", required=True, help="Recorte de GEBCO dentro de WAVE-DATA/raw (registrar-gebco).")
    p.add_argument("--nodo-lat", type=float, required=True, help="Latitud del nodo elegido.")
    p.add_argument("--nodo-lon", type=float, required=True, help="Longitud del nodo elegido.")
    p.add_argument("--nombre", help="Nombre de salida (por defecto el de la corrida exploratoria).")
    p.add_argument("--datos", help="Raíz de WAVE-DATA (si no, la de la corrida exploratoria o WAVE_DATA_DIR).")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    from .final import ConfigFinal, ErrorGate, generar_mapa_final

    try:
        res = generar_mapa_final(ConfigFinal(exploratorio=a.exploratorio, gebco=a.gebco, nodo_lat=a.nodo_lat,
                                             nodo_lon=a.nodo_lon, nombre=a.nombre, directorio_datos=a.datos))
    except ErrorGate as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    for nombre, ruta in res.salidas.items():
        print(f"{nombre}: {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def main_serie(argv=None) -> int:
    """Serie de WAVERYS del bloque de 5 × 5 (ADR-0016): catálogo, descarga por año con reanudación y gate."""
    import json
    import time
    from pathlib import Path

    import xarray as xr

    from .. import rutas
    from . import serie, waverys

    p = argparse.ArgumentParser(prog="serie-waverys", description="Descarga la serie de WAVERYS por año y pasa el gate (ADR-0016).")
    p.add_argument("--anio", type=int, nargs="+", help="Solo estos años (prueba). Sin --anio: el registro completo y el gate.")
    p.add_argument("--nombre", default="marcona-200m", help="Nombre del sitio para el informe del gate.")
    p.add_argument("--datos", help="Raíz de WAVE-DATA (si no, WAVE_DATA_DIR).")
    a = p.parse_args(argv)
    raiz = Path(rutas.directorio_datos(a.datos))
    raiz_raw = rutas.raw(raiz)
    cfg = serie.ConfigSerie()
    cob = serie.consultar_catalogo(cfg)
    print(f"Catálogo: {cob.dataset_id} v{cob.version}, {cob.inicio} → {cob.fin}, {len(cob.variables)} variables.")
    pedidos = a.anio or serie.anios(cob)
    archivos = []
    for anio in pedidos:
        t0 = time.time()
        ruta, entrada = serie.descargar_anio(anio, cfg, cob, raiz_raw)
        archivos.append(ruta)
        print(f"{anio}: {ruta.name}  {entrada['bytes'] / 1e6:.2f} MB  {time.time() - t0:.1f} s")
    destino = raiz / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / a.nombre / "serie_gate.json"
    if a.anio:
        print("Prueba por años: no se evalúa el gate del registro completo.")
        return 0
    malla = next(iter(sorted((raiz_raw / "waverys" / "malla").glob("*.nc"))))
    lat, lon, valido = waverys.leer_malla(malla, "VHM0", (cfg.oeste, cfg.este, cfg.sur, cfg.norte))
    mascara = xr.DataArray(valido, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    try:
        informe = serie.verificar(archivos, cob, cfg, raiz_raw, mascara)
    except serie.ErrorSerie as exc:
        if destino.exists():
            destino.unlink()  # un gate fallido invalida el informe publicado anterior
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    destino.parent.mkdir(parents=True, exist_ok=True)
    informe["catalogo"] = cob.__dict__
    informe["malla_referencia"] = str(malla)
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Gate de la serie: CUMPLE. Informe en {destino}")
    return 0
