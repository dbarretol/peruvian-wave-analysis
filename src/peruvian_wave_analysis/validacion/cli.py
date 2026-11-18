"""Línea de comandos de las descargas de la F5 (``validacion-descarga``). Escribe solo en ``raw/``; no publica nada.

G4 en los mensajes: se imprimen nombres de archivo, tamaños, dimensiones, variables, número de registros y fechas
extremas. Nunca un valor observado.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .. import rutas
from . import ndbc, waverys

FUENTES = ("ndbc", "waverys-boyas", "waverys-regional")


def _ndbc(raiz_raw: Path, anios: list[int] | None, comparar: Path | None) -> int:
    for boya in ndbc.BOYAS:
        sel = [a for a in anios if a in boya.anios] if anios else None
        if anios and not sel:
            continue
        rutas_b, faltan = ndbc.descargar(raiz_raw, boya, sel)
        for r in rutas_b:
            e = ndbc.estructura(r)
            print(
                f"{boya.estacion} {e['archivo']}: {e['formato']}; {e['registros']} registros "
                f"({e['duplicados']} duplicados), {e['inicio']} → {e['fin']}; columnas {', '.join(e['columnas'])}"
            )
        if faltan:
            print(f"{boya.estacion}: NDBC no tiene los años {faltan}.", file=sys.stderr)
        if comparar is not None:
            res = ndbc.comparar_hashes(rutas_b, comparar)
            resumen = {k: sum(v == k for v in res.values()) for k in ("igual", "distinto", "ausente")}
            print(f"{boya.estacion}, hash frente a {comparar}: {resumen}")
            for nombre, v in res.items():
                if v == "distinto":
                    print(f"  distinto: {nombre}")
    return 0


def _waverys(raiz_raw: Path, recs: list[waverys.Recuadro], anios: list[int] | None) -> int:
    tramos = waverys.tramos_del_nodo(raiz_raw)
    coberturas = {}
    for tr in tramos:
        coberturas[tr] = waverys.comprobar_catalogo(tr)
        print(f"Tramo del nodo: {tr.dataset_id} v{tr.version}, {tr.inicio} → {tr.fin} (el catálogo de hoy lo cubre)")
    for rec in recs:
        for anio, tr, ini, fin in waverys.pedidos(rec, tramos):
            if anios and anio not in anios:
                continue
            t0 = time.time()
            ruta, entrada = waverys.descargar(raiz_raw, rec, tr, coberturas[tr], anio, ini, fin)
            e = waverys.estructura(ruta)
            print(
                f"{rec.nombre} {anio} [{tr.dataset_id} v{tr.version}]: {ruta.name}  {entrada['bytes'] / 1e6:.2f} MB  "
                f"{e['dimensiones']}  {e['inicio']} → {e['fin']}  {time.time() - t0:.1f} s"
            )
    return 0


def main_descarga(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="validacion-descarga",
        description="Descargas de la F5: boyas NDBC 32302 y 32012 (stdmet) y WAVERYS en la región y en las boyas.",
    )
    p.add_argument("fuente", choices=(*FUENTES, "todo"))
    p.add_argument("--anio", type=int, nargs="+", help="Solo estos años.")
    p.add_argument("--datos", help="Raíz de WAVE-DATA (si no, WAVE_DATA_DIR).")
    p.add_argument(
        "--comparar-hash-con",
        type=Path,
        help="Solo ndbc: directorio con copias de los mismos archivos; compara el sha256 sin abrirlos.",
    )
    a = p.parse_args(argv)
    try:
        raiz_raw = rutas.raw(Path(rutas.directorio_datos(a.datos)))
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc} Ejemplo: --datos C:\\WORKSPACES\\TESIS-M5-WAVE\\WAVE-DATA", file=sys.stderr)
        return 2
    fuentes = FUENTES if a.fuente == "todo" else (a.fuente,)
    try:
        if "ndbc" in fuentes:
            _ndbc(raiz_raw, a.anio, a.comparar_hash_con)
        recs = []
        if "waverys-boyas" in fuentes:
            recs += [waverys.recuadro_boya(b) for b in ndbc.BOYAS]
        if "waverys-regional" in fuentes:
            recs.append(waverys.recuadro_regional())
        if recs:
            _waverys(raiz_raw, recs, a.anio)
    except (ndbc.ErrorNDBC, waverys.ErrorDescarga) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Descarga terminada. Registrar las fuentes nuevas en fuentes-de-datos.md.")
    return 0
