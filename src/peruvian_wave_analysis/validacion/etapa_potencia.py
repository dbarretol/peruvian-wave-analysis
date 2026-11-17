"""Etapa 2 de la F5, ``validacion-potencia`` (contrato v6, §2, §3.7, §4.4, §5 y §19). No lee valores observados (G4).

Orden:
1. Colocalización geométrica (altimetría: hora, posición, bandera; boyas: tiempo y máscara de dato presente).
2. Escalón del modelo en las colocaciones frente al nodo: regla del ⅓ y traslado por equivalencia (§19).
3. **Antes de calcular ninguna potencia:** prueba de nulidad del generador (§8: b = 0, SI = 0.16, ρ = 0.8).
4. Réplicas con b = 0 en cada punto de la rejilla; error de tipo I (y recalibración si sale de rango).
5. Potencia por desplazamiento, efecto mínimo detectable al 80 % y su validación con inyección explícita.
6. Subida esperada de la regla de variantes con sesgo nulo (ADR-0024, punto 2).

Salidas en ``processed/F05-validacion-externa/potencia/<nombre>/``. ``--previo`` hace 1–3 y cronometra una réplica sin
publicar (en ``interim/``).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import manifiesto, rutas
from ..comun import etapa, json_estricto, orquestador, procesos, semillas
from ..comun.errores import GateFallido
from ..datos.run import ruta_serie
from . import config as C
from . import geometria as G
from . import potencia as P
from .modelo import ETAPA_MODELO, ruta_modelo

NOMBRE = "marcona-200m"
ETAPA_POTENCIA = orquestador.Etapa("validacion-potencia", ETAPA_MODELO.codigo, Path("."))
TROZO = 25  # réplicas por tarea
REPLICAS_NULIDAD = 200
REPLICAS_VALIDACION = 200
HALLAZGO_ESCENARIOS = (
    "Con esta muestra, la F5 no puede retirar escenarios (contrato §19): el traslado al nodo sale «dudoso» en todas "
    "las comparaciones. V1 y V2 informan a la F8, pero no deciden; E0, E1 y E2 siguen activos."
)


def ruta_potencia(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F05-validacion-externa" / "potencia" / nombre


def ruta_previo(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "interim" / "F05-validacion-externa" / "potencia-previo" / nombre


# --- Insumos ------------------------------------------------------------------------------------------


def insumos(raiz: Path, nombre: str) -> tuple[dict[str, Path], dict[str, str]]:
    raw = rutas.raw(raiz)
    ent = manifiesto.leer(raw)
    ins, esp = {}, {}
    for r, p in {**G.rutas_altimetria(raw), **G.rutas_ndbc(raw), G.GEBCO: raw / G.GEBCO}.items():
        ins[f"raw:{r}"], esp[f"raw:{r}"] = p, ent[r]["sha256"]
    for r in sorted(ent):
        if r.startswith("costa/natural_earth/"):
            ins[f"raw:{r}"], esp[f"raw:{r}"] = raw / r, ent[r]["sha256"]
    dm, ds = ruta_modelo(raiz, nombre), ruta_serie(raiz, nombre)
    dfi = Path(raiz) / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / nombre
    d4 = Path(raiz) / "processed" / "F04-homogeneidad" / "homogeneidad" / nombre
    for d in (dm, ds, dfi, d4):
        etapa.exigir_publicada(d)
    ins |= {
        "modelo_region": dm / "modelo_region.parquet",
        "marcas_region": dm / "marcas_region.parquet",
        "modelo_boyas": dm / "modelo_boyas.parquet",
        "serie_nodo": ds / "serie_nodo.parquet",
        "exposicion": dfi / "exposicion.csv",
        "pruebas_cambio": d4 / "pruebas_cambio.json",
    }
    return ins, esp


def geometria(ins: dict[str, Path], raiz: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict[str, float]]:
    raw = rutas.raw(raiz)
    region = pd.read_parquet(ins["modelo_region"])
    marcas = pd.read_parquet(ins["marcas_region"])
    nodo = pd.read_parquet(ins["serie_nodo"], columns=["VHM0"])["VHM0"]
    alt_r = [p for k, p in sorted(ins.items()) if k.startswith("raw:altimetria/")]
    boy_r = {k.removeprefix("raw:"): p for k, p in sorted(ins.items()) if k.startswith("raw:boyas/")}
    sect = pd.read_csv(ins["exposicion"])
    alt, qc = G.colocaciones_altimetria(alt_r, raw / G.GEBCO, G.tierra(raw), sect, region, marcas, nodo)
    boy = G.colocaciones_boyas(boy_r, pd.read_parquet(ins["modelo_boyas"]), nodo)
    efectos = G.efectos_nodo(json_estricto.leer(ins["pruebas_cambio"]))
    return alt, boy, qc, efectos


# --- Trabajo en paralelo --------------------------------------------------------------------------------


def _trabajo(args) -> list[dict]:
    """Una tarea: réplicas ``indices`` de un punto. Se ejecuta en un proceso hijo (``procesos.mapear``)."""
    ruta_alt, ruta_boy, punto, r, indices, inyeccion = args
    lados = P.preparar_lados(pd.read_parquet(ruta_alt), pd.read_parquet(ruta_boy))
    return [P.replica(lados, r, punto, i, inyeccion=inyeccion) for i in indices]


def correr(ruta_alt, ruta_boy, punto, r, n, procesos_, inyeccion=None, desde: int = 0) -> list[dict]:
    tareas = [
        (ruta_alt, ruta_boy, punto, r, list(range(a, min(a + TROZO, desde + n))), inyeccion)
        for a in range(desde, desde + n, TROZO)
    ]
    return [x for parte in procesos.mapear(_trabajo, tareas, procesos=procesos_) for x in parte]


def _fila_replica(punto: str, i: int, rp: dict) -> list[dict]:
    out = []
    for k in ("V1a", "V1b", "V1b_lado", "V2a", "V3a", "V3b"):
        pr = rp[k]
        fila = {"punto": punto, "replica": i, "prueba": k, "p": pr.p}
        fila |= {f"est_{j}": v for j, v in enumerate(pr.est)} | {f"ee_{j}": v for j, v in enumerate(pr.ee)}
        fila |= {f"critico_{a:g}": v for a, v in pr.critico.items()}
        out.append(fila)
    return out


# --- Etapa --------------------------------------------------------------------------------------------


def _efectos_fuente(esc: pd.DataFrame, efectos_nodo: dict[str, float]) -> dict[str, float]:
    p90 = esc[esc["estadistico"] == "P90"].set_index("comparacion")
    ef = {
        "V1a": float(p90.loc["V1a", "efecto_fuente"]),
        "V1b": float(p90.loc["V1b", "efecto_fuente"]),
        "V2a": float(p90.loc["V2a", "efecto_fuente"]),
        "V3a": C.EFECTO_V3,
        "V3b": C.EFECTO_V3,
    }
    ef["V1b_lado"] = ef["V1b"]
    return ef


def potencia_etapa(
    raiz_datos=None, nombre: str = NOMBRE, procesos_: int = 4, replicas: int = C.REPLICAS_POTENCIA, previo: bool = False
) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins, esp = insumos(raiz, nombre)
    etapa.exigir_insumos(ins, esp)  # G2
    salida = ruta_previo(raiz, nombre) if previo else ruta_potencia(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_POTENCIA)
    P.FACTORES.clear()
    with etapa.staging_protegido(salida, raices) as st:
        t0 = time.time()
        alt, boy, qc, efectos_nodo = geometria(ins, raiz)
        geo = pd.concat([alt.assign(fuente="altimetría"), boy.assign(fuente="boya")], ignore_index=True)
        geo.to_parquet(st / "colocaciones_geometria.parquet")
        ruta_alt, ruta_boy = st / "_alt.parquet", st / "_boy.parquet"
        alt.to_parquet(ruta_alt)
        boy.to_parquet(ruta_boy)
        pd.DataFrame(qc).to_csv(st / "qc_geometria.csv", index=False, lineterminator="\n")
        esc = G.escalones(alt, boy, efectos_nodo, C.B_BOOT, semillas.generador("f05", "escalon-modelo"))
        esc.to_csv(st / "escalon_modelo.csv", index=False, float_format="%.6g", lineterminator="\n")
        efectos = _efectos_fuente(esc, efectos_nodo)
        lados_res = {n: G.resumen_lado(G.colocaciones_de_lado(alt, boy, n)) for n in C.LADOS}
        print(f"Geometría y escalón del modelo: {time.time() - t0:.0f} s", flush=True)

        # 3. Nulidad del generador, antes de ninguna potencia (§8)
        lados = P.preparar_lados(alt, boy)
        p_nul = P.Punto("nulidad_SI0.16", 0.16, 0.16, 0.8, 1.0)
        r_nul = P.calibrar(lados, p_nul)
        t1 = time.time()
        reps_nul = correr(
            ruta_alt, ruta_boy, p_nul, r_nul, REPLICAS_NULIDAD if not previo else 2 * procesos_, procesos_
        )
        nul = P.nulidad_generador(reps_nul)
        seg_rep = (time.time() - t1) / len(reps_nul) * procesos_
        nul_ok = all(abs(v["media"]) <= 3 * v["ee_mc"] for k, v in nul.items() if k in ("B_P90", "B_P99"))
        print(f"Nulidad del generador: {nul} → {'cumple' if nul_ok else 'NO CUMPLE'}", flush=True)
        n_puntos = len(P.rejilla())
        est_h = seg_rep * replicas * n_puntos / procesos_ / 3600
        print(
            f"Una réplica ≈ {seg_rep:.1f} s por proceso; rejilla completa ≈ {est_h:.1f} h con {procesos_} procesos",
            flush=True,
        )
        resumen = {
            "valores_observados_leidos": False,
            "hallazgos": [HALLAZGO_ESCENARIOS],
            "efectos_nodo": efectos_nodo,
            "efectos_por_fuente": efectos,
            "atribuye": esc[esc["estadistico"] == "P90"].set_index("comparacion")["atribuye"].dropna().to_dict(),
            "traslado_p90": esc[esc["estadistico"] == "P90"].set_index("comparacion")["traslado"].dropna().to_dict(),
            "lados": lados_res,
            "pesos_hy2": G.pesos_hy2(alt),
            "nulidad_generador": {"replicas": len(reps_nul), "resultado": nul, "cumple": nul_ok},
            "segundos_por_replica": seg_rep,
            "estimacion_horas_rejilla": est_h,
            "replicas_por_punto": replicas,
            "B_bootstrap_potencia": C.B_POTENCIA,
            "aproximaciones_declaradas": [
                "Potencia por desplazamiento con réplicas de b = 0, validada con inyección explícita en el nominal.",
                "Calibración linealizada sobre el estadístico en el bootstrap (exacta en el generador).",
                "SIMEX como desplazamiento fijo de las réplicas bootstrap (no se repite dentro de cada réplica).",
                "Ruido de 1 Hz generado a nivel de superobservación (σ/√n), equivalente a la media de n puntos iid.",
                "V3b: B(P99) − B(P50) se desplaza en el efecto (sesgo creciente con el nivel).",
            ],
        }
        if previo:
            json_estricto.escribir(st / "potencia_previo.json", resumen)
            for f in (ruta_alt, ruta_boy):
                f.unlink()
            return etapa.publicar(
                st,
                salida,
                etapa.Gate(nul_ok, {"previo": True}, ["nulidad"] if not nul_ok else []),
                raices,
                ins,
                raiz,
                semillas.semilla("f05", "potencia"),
            )

        # 4–5. Rejilla
        tablas, filas_rep, mde, tipo1, subida = [], [], [], [], {}
        reps_nominal = None
        for punto in P.rejilla():
            r = P.calibrar(lados, punto)
            t2 = time.time()
            reps = correr(ruta_alt, ruta_boy, punto, r, replicas, procesos_)
            print(f"{punto.nombre}: {len(reps)} réplicas en {(time.time() - t2) / 60:.1f} min", flush=True)
            if punto.nombre == P.NOMINAL:
                reps_nominal = reps
                for prueba, alfa in (
                    ("V1a", C.ALFA_FUENTE),
                    ("V1b", C.ALFA_FUENTE),
                    ("V1b_lado", C.ALFA_FUENTE),
                    ("V2a", C.ALFA_FUENTE),
                    ("V3a", C.ALFA_FAMILIA),
                    ("V3b", C.ALFA_FAMILIA),
                ):
                    rc = P.recalibrar(reps, prueba, alfa)
                    tipo1.append(rc)
                    if rc["recalibrado"]:
                        P.FACTORES[prueba] = rc["factor"]
                tipo1.append(
                    {
                        "prueba": "V1 familia",
                        "alfa": C.ALFA_FAMILIA,
                        "tasa": P.tasa_familia_v1(reps),
                        "rango": [0.03, 0.07],
                    }
                )
            tablas.append(P.tabla_potencia(reps, punto, efectos))
            for prueba in ("V1a", "V1b", "V1b_lado", "V2a", "V3a", "V3b"):
                alfa = C.ALFA_FAMILIA if prueba.startswith("V3") else C.ALFA_FUENTE
                mde.append(
                    {
                        "punto": punto.nombre,
                        "prueba": prueba,
                        "efecto": efectos[prueba],
                        "efecto_minimo_80": P.efecto_minimo(reps, prueba, alfa, efectos[prueba]),
                    }
                )
            pv3a = P.potencia(reps, "V3a", efectos["V3a"], C.ALFA_FAMILIA)
            pv3b = P.potencia(reps, "V3b", efectos["V3b"], C.ALFA_FAMILIA)
            subida[punto.nombre] = P.subida_esperada(reps, pv3a, pv3b)
            filas_rep += [f for i, rp in enumerate(reps) for f in _fila_replica(punto.nombre, i, rp)]
        tabla = pd.concat(tablas, ignore_index=True)

        # 5b. Validación del desplazamiento con inyección explícita (nominal)
        nominal = next(p for p in P.rejilla() if p.nombre == P.NOMINAL)
        r_nom = P.calibrar(lados, nominal)
        valid = []
        for prueba, lado, alfa in (("V1b", "T1b", C.ALFA_FUENTE), ("V3a", "A", C.ALFA_FAMILIA)):
            for mu in (1.0, 3.0):
                b = mu * efectos[prueba]
                reps_i = correr(ruta_alt, ruta_boy, nominal, r_nom, REPLICAS_VALIDACION, procesos_, {lado: b})
                p_iny = P.potencia(reps_i, prueba, 0.0, alfa)
                p_des = P.potencia(reps_nominal[:REPLICAS_VALIDACION], prueba, b, alfa)
                ee = np.sqrt(max(p_des * (1 - p_des), 0.01) / REPLICAS_VALIDACION)
                valid.append(
                    {
                        "prueba": prueba,
                        "multiplo": mu,
                        "b": b,
                        "potencia_inyeccion": p_iny,
                        "potencia_desplazamiento": p_des,
                        "diferencia": p_iny - p_des,
                        "dentro_de_3_ee": bool(abs(p_iny - p_des) <= 3 * np.sqrt(2) * ee),
                    }
                )

        tabla.to_csv(st / "potencia.csv", index=False, float_format="%.6g", lineterminator="\n")
        pd.DataFrame(mde).to_csv(st / "efecto_minimo.csv", index=False, float_format="%.6g", lineterminator="\n")
        pd.DataFrame(filas_rep).to_parquet(st / "replicas_potencia.parquet")
        nom = subida[P.NOMINAL]
        if nom["supera_2_por_ciento"]:
            resumen["hallazgos"].append(
                f"La subida esperada de la regla de variantes con sesgo nulo es {100 * nom['subida_media']:.2f} %, "
                "mayor que el 2 % de ADR-0006 (ADR-0024, punto 2)."
            )
        resumen |= {
            "error_tipo_I": tipo1,
            "factores_recalibrados": dict(P.FACTORES),
            "validacion_desplazamiento": valid,
            "subida_esperada": subida,
            "rejilla": [asdict(p) for p in P.rejilla()],
            "nominal": P.NOMINAL,
        }
        json_estricto.escribir(st / "potencia.json", resumen)
        for f in (ruta_alt, ruta_boy):
            f.unlink()
        motivos = []
        if not nul_ok:
            motivos.append("§8: el generador no es nulo")
        if not all(v["dentro_de_3_ee"] for v in valid):
            motivos.append("la potencia por desplazamiento no coincide con la inyección explícita")
        criterios = {"replicas": replicas, "puntos": n_puntos, "nulidad": nul_ok}
        return etapa.publicar(
            st,
            salida,
            etapa.Gate(not motivos, criterios, motivos),
            raices,
            ins,
            raiz,
            semillas.semilla("f05", "potencia"),
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validacion-potencia", description="Etapa 2 de la F5: geometría y potencia.")
    ap.add_argument("--datos")
    ap.add_argument("--nombre", default=NOMBRE)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=C.REPLICAS_POTENCIA)
    ap.add_argument("--previo", action="store_true", help="Geometría, escalón, nulidad y cronometraje; sin rejilla.")
    a = ap.parse_args(argv)
    try:
        salida = potencia_etapa(a.datos, a.nombre, a.procesos, a.replicas, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
