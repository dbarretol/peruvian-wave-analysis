"""Ramas y picos sobre el sitio, ``eventos-ramas-picos`` (contrato de la F7, §3.1, §3.2 puntos 5 y 6, §3.4 E-29,
§3.5 G8 y §3.6). Primer contacto con los picos del sitio (autorizado por el usuario el 2026-11-23).

**No elige nada.** El IET\\* todavía no está decidido (§3.3), así que la etapa corre en cada IET de la rejilla (48 a
120 h) e informa, por IET y por rama:
- eventos, picos con ``pico_movido`` y ventanas recortadas (E1);
- en las variantes por cuantiles, u_id llevado por su propio cuantil, estados discordantes y si se activa E-29;
- la protección de cola (G8): picos de rama en instantes con Hs de E0 sin corregir < P95 de la F3;
- exposición y λ con su IC de Garwood (con la nota de E0, ``ramas.NOTA_GARWOOD_E0``);
- banderas ``dudoso_f3`` y ``revision_visual_f6``.

No ajusta ninguna GPD ni calcula ξ, σ\\* ni RL. Las tablas definitivas (``eventos.csv`` y ``picos_<rama>.csv``, con
IET\\* y u\\*) las publica la etapa con datos en ``marcona-200m/``; esta publica en la carpeta hermana
``ramas-picos/``, que aquella no borra.

**Gate:** G1 (insumos publicados), rejilla completa (G2), ningún evento sin dato de la rama en su ventana, y G8: si
algún pico de una variante cae bajo el P95 de la F3, la etapa falla con «revisión requerida» y lista los picos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .. import rutas
from ..clima import insumos as CI
from ..comun import etapa, json_estricto, orquestador
from ..comun.errores import GateFallido
from ..datos.run import ruta_candidatos
from . import identificacion as ID
from . import ramas as R

ETAPA = orquestador.Etapa("eventos-ramas-picos", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
NOMBRE = CI.NOMBRE
IETS_H = (48.0, 54.0, 60.0, 66.0, 72.0, 84.0, 96.0, 120.0)  # §3.3 (42 h solo es vecino para el RL100)


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "ramas-picos" / NOMBRE


def ruta_f06(raiz: Path) -> Path:
    return Path(raiz) / "processed" / "F06-clima-de-oleaje" / NOMBRE


def rutas_insumos(raiz: Path) -> dict[str, Path]:
    ins = CI.rutas_insumos(raiz, NOMBRE)
    d6, dc = ruta_f06(raiz), ruta_candidatos(raiz, NOMBRE)
    etapa.exigir_publicada(d6)
    return {
        "serie_nodo": ins["serie_nodo"],
        "serie_corregida": ins["serie_corregida"],
        "resumen_f05": ins["resumen_f05"],
        "resumen_f04": ins["resumen_f04"],
        "candidatos": ins["candidatos"],
        "resumen_candidatos": dc / "resumen_candidatos.json",
        "declustering_dudoso": d6 / "declustering_dudoso.csv",
    }


def ramas_picos(datos: CI.Datos, u_id: float, p95_f3: float, candidatos: pd.DataFrame, revision: pd.DataFrame,
                iets=IETS_H) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:  # fmt: skip
    """(conteos por IET y rama, picos de todas las ramas e IET, u_id en las variantes, picos bajo el P95)."""
    serie = datos.serie
    hs0 = serie["VHM0"].astype(float)
    inicio_serie = pd.Timestamp(hs0.index[0])
    variantes = {}
    for rama in R.RAMAS:
        if R.es_variante(rama):
            ini = R.inicio_rama(rama, inicio_serie, datos.inicio_e1)
            base = datos.ramas[R.SIN_CORREGIR[rama]]
            en = (base.index >= ini) & (base.index <= R.FIN_MODELO)
            var = datos.ramas[rama]
            variantes[rama] = R.u_id_en_variante(base[en], var[(var.index >= ini) & (var.index <= R.FIN_MODELO)], u_id)
    conteos, tablas, bajo_p95 = [], [], []
    for iet in iets:
        ev = ID.identificar(hs0, u_id, iet)
        for rama in R.RAMAS:
            ini = R.inicio_rama(rama, inicio_serie, datos.inicio_e1)
            ev_r = R.eventos_de_rama(ev, ini)
            p = R.picos_de_rama(ev_r, datos.ramas[rama], serie)
            borde = {"activo": False, "N0": int(len(p)), "picos_antes": int(len(p)), "picos_despues": int(len(p))}
            if R.es_variante(rama):
                p, borde = R.borde_inferior(p, variantes[rama])
            g8 = R.proteccion_cola(p, p95_f3)
            for x in g8["lista"]:
                bajo_p95.append({"iet_h": iet, "rama": rama} | x)
            p = p.assign(
                dudoso_f3=R.marcar_dudosos_f3(p, candidatos), revision_visual_f6=R.marcar_revision_f6(p, revision)
            )
            expo = R.exposicion(datos.ramas[rama], ini, n_picos=int(len(p)), rama=rama)
            conteos.append({
                "iet_h": iet, "rama": rama, "eventos": int(len(ev_r)), "picos": int(len(p)),
                "pico_movido": int(p["pico_movido"].sum()), "empates_resueltos": int(p["empate_resuelto"].sum()),
                "recortados": int(p["recortado"].sum()),
                "e29_activo": bool(borde["activo"]), "picos_antes_de_e29": int(borde["picos_antes"]),
                "picos_bajo_p95_f3": int(g8["picos_bajo_p95"]), "dudoso_f3": int(p["dudoso_f3"].sum()),
                "revision_visual_f6": int((p["revision_visual_f6"] != "").sum()),
                "divididos_por_bajada": int(p["dividido_por_bajada"].sum()), "toca_hueco": int(p["toca_hueco"].sum()),
                "anios_exposicion": expo["anios"], "lambda_por_anio": expo["lambda_por_anio"],
                "ic_garwood_95_inf": expo["ic_garwood_95"][0], "ic_garwood_95_sup": expo["ic_garwood_95"][1],
                "nota_ic": expo["nota_ic"] or "",
            })  # fmt: skip
            tablas.append(p.drop(columns=["i_pico"]).assign(iet_h=iet, rama=rama))
    return pd.DataFrame(conteos), pd.concat(tablas, ignore_index=True), variantes, bajo_p95


def ejecutar(raiz_datos=None) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = rutas_insumos(raiz)
    etapa.exigir_insumos(ins)
    datos = CI.cargar(raiz, ins, NOMBRE)
    rc = json_estricto.leer(ins["resumen_candidatos"])
    p95 = float(rc["p95_hs_m"])
    u_id = p95  # §3.2, punto 1 (el conteo de §5.0 no lo revisó)
    candidatos = pd.read_csv(ins["candidatos"])
    revision = pd.read_csv(ins["declustering_dudoso"])
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        conteos, picos, variantes, bajo = ramas_picos(datos, u_id, p95, candidatos, revision)
        opciones = {"index": False, "float_format": "%.6g", "lineterminator": "\n"}
        conteos.to_csv(st / "conteos.csv", **opciones)
        picos.to_csv(st / "picos_por_iet.csv", **opciones, date_format="%Y-%m-%d %H:%M")
        resumen = {
            "u_id_m": u_id,
            "p95_f3_m": p95,
            "inicio_e1": str(datos.inicio_e1),
            "fin_modelo": str(R.FIN_MODELO),
            "iets_h": list(IETS_H),
            "u_id_en_variantes": variantes,
            "e29_activo_en_alguna_rama": bool(conteos["e29_activo"].any()),
            "g8": {"picos_bajo_p95_f3": bajo, "estado": R.REVISION_REQUERIDA if bajo else "cumple"},
            "nota_ic_e0": R.NOTA_GARWOOD_E0,
            "rotulo": "Primer contacto con los picos del sitio (2026-11-23). Descriptivo: no elige IET\\* ni u\\*.",
            "ajustes_hechos": False,
        }
        json_estricto.escribir(st / "ramas_picos.json", resumen)
        motivos = []
        if bajo:
            motivos.append(f"{R.REVISION_REQUERIDA}: {len(bajo)} picos de variante bajo el P95 de la F3 (G8)")
        gate = etapa.Gate(not motivos, {"iets": len(IETS_H), "ramas": len(R.RAMAS)}, motivos)
        return etapa.publicar(st, salida, gate, raices, ins, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-ramas-picos", description="Ramas y picos de la F7 en cada IET")
    ap.add_argument("--datos")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = ejecutar(a.datos)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
