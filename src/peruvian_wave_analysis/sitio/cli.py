"""Línea de comandos del mapa de selección de sitio.

Ejemplos::

    mapa-sitio --config config/mapa_sitio.ejemplo.yaml
    mapa-sitio --nombre marcona --lat -15.4 --lon -75.3 --isobatas 200 100 50
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

from .config import ConfigMapa


def _argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mapa-sitio", description="Mapa de isóbatas y malla de WAVERYS alrededor de una coordenada.")
    p.add_argument("--config", help="Archivo YAML de configuración. Los demás argumentos lo sobrescriben.")
    p.add_argument("--nombre", help="Nombre corto del sitio (minúsculas; se usa en las rutas de salida).")
    p.add_argument("--lat", type=float, help="Latitud de referencia (°, negativa al sur).")
    p.add_argument("--lon", type=float, help="Longitud de referencia (°, negativa al oeste).")
    p.add_argument("--isobatas", type=float, nargs="+", help="Profundidades de las isóbatas (m). Por defecto 200 100 50.")
    p.add_argument("--isobata-principal", type=float, help="Isóbata con la que se ordenan los candidatos.")
    p.add_argument("--semiancho", type=float, help="Semiancho del recuadro (°). Por defecto 1.")
    p.add_argument("--n-candidatos", type=int, help="Candidatos numerados en el mapa. Por defecto 10.")
    p.add_argument("--datos", help="Raíz de WAVE-DATA (si no, variable de entorno WAVE_DATA_DIR).")
    p.add_argument("--mostrar", action="store_true", help="Abre la figura en una ventana además de guardarla.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _argumentos(argv)
    base = {}
    if a.config:
        base = ConfigMapa.desde_yaml(a.config).como_dict()
    for clave, valor in {
        "nombre": a.nombre, "latitud": a.lat, "longitud": a.lon, "isobatas": a.isobatas,
        "isobata_principal": a.isobata_principal, "semiancho_grados": a.semiancho,
        "n_candidatos": a.n_candidatos, "directorio_datos": a.datos,
    }.items():
        if valor is not None:
            base[clave] = valor
    if a.isobatas is not None and a.isobata_principal is None:
        base["isobata_principal"] = None
    faltan = [k for k in ("nombre", "latitud", "longitud") if k not in base]
    if faltan:
        print(f"Faltan argumentos: {', '.join(faltan)} (o usa --config).", file=sys.stderr)
        return 2
    if not a.mostrar:
        matplotlib.use("Agg")
    from .mapa import generar_mapa  # después de fijar el backend

    cfg = ConfigMapa(**base)
    res = generar_mapa(cfg)
    cols = ["candidato", "lat", "lon", "prof_nodo_m", "prof_media_celda_m",
            f"dist_iso_{cfg.isobata_principal:g}m_km", "dist_costa_km"]
    print(res.tabla[res.tabla["candidato"].notna()].head(10)[cols].to_string(index=False))
    for nombre, ruta in res.salidas.items():
        print(f"{nombre}: {ruta}")
    if a.mostrar:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
