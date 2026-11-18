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

import pandas as pd

from .. import manifiesto, rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..datos.run import ruta_serie
from . import config as C
from . import geometria as G
from . import potencia as P
from . import reanudable
from .modelo import ETAPA_MODELO, ruta_modelo

NOMBRE = "marcona-200m"
ETAPA_POTENCIA = orquestador.Etapa("validacion-potencia", ETAPA_MODELO.codigo, Path("."))
TROZO = 25  # réplicas por tarea
REPLICAS_NULIDAD = 200
REPLICAS_VALIDACION = 200
# Validación del desplazamiento (§20.2): el nominal y el punto más ruidoso de la rejilla (ρ = 0.8, SI 0.16/0.20,
# calibración y ruido × 2; S-22), con los dos signos. V3b con inyección que crece con el nivel (S-20).
PUNTO_RUIDOSO = "SI0.16-0.2_cal2_rho0.8"
VALIDACIONES = (
    ("V1a", "A", C.ALFA_FUENTE, "constante"),
    ("V1b", "T1b", C.ALFA_FUENTE, "constante"),
    ("V3a", "A", C.ALFA_FAMILIA, "constante"),
    ("V3b", "A", C.ALFA_FAMILIA, "nivel"),
)
PRUEBAS = (
    ("V1a", C.ALFA_FUENTE),
    ("V1b", C.ALFA_FUENTE),
    ("V1b_lado", C.ALFA_FUENTE),
    ("V2a", C.ALFA_FUENTE),
    ("V3a", C.ALFA_FAMILIA),
    ("V3b", C.ALFA_FAMILIA),
    ("V3a_sinmax", C.ALFA_FAMILIA),
    ("V3b_sinmax", C.ALFA_FAMILIA),
)
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


AVANCE: dict = {"dir": None}  # directorio de avance de la corrida en curso (lo fija ``potencia_etapa``)


def ruta_avance(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "interim" / "F05-validacion-externa" / "potencia-avance" / nombre


def _nombre_tarea(punto, indices, inyeccion) -> str:
    iny = "nula" if not inyeccion else reanudable.clave(sorted((k, v) for k, v in inyeccion.items()))
    return f"{punto.nombre}__{iny}__{indices[0]:05d}-{indices[-1]:05d}"


def correr(ruta_alt, ruta_boy, punto, r, n, procesos_, inyeccion=None, desde: int = 0) -> list[dict]:
    """Réplicas ``desde`` … ``desde + n − 1`` de un punto, en tareas de ``TROZO`` réplicas, **reanudables**: cada
    tarea se guarda al terminar en el directorio de avance y no se repite al relanzar la etapa."""
    tareas = []
    for a in range(desde, desde + n, TROZO):
        indices = list(range(a, min(a + TROZO, desde + n)))
        tareas.append((_nombre_tarea(punto, indices, inyeccion), (ruta_alt, ruta_boy, punto, r, indices, inyeccion)))
    aviso = lambda t: print(f"  {punto.nombre}{' (inyección)' if inyeccion else ''}: {t}", flush=True)  # noqa: E731
    partes = reanudable.ejecutar(_trabajo, tareas, AVANCE["dir"], procesos_, aviso)
    return [x for parte in partes for x in parte]


def _fila_replica(punto: str, i: int, rp: dict) -> list[dict]:
    out = []
    for k, _ in PRUEBAS:
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
    ef["V3a_sinmax"] = ef["V3b_sinmax"] = C.EFECTO_V3
    return ef


def potencia_etapa(
    raiz_datos=None, nombre: str = NOMBRE, procesos_: int = 4, replicas: int = C.REPLICAS_POTENCIA, previo: bool = False
) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins, esp = insumos(raiz, nombre)
    hashes = etapa.exigir_insumos(ins, esp)  # G2
    salida = ruta_previo(raiz, nombre) if previo else ruta_potencia(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_POTENCIA)
    P.FACTORES.clear()
    # Avance reanudable: la clave cambia si cambia el código de la etapa o cualquier insumo
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA_POTENCIA), hashes, C.B_POTENCIA, TROZO)
    AVANCE["dir"] = ruta_avance(raiz, nombre) / clave
    print(f"Avance reanudable en {AVANCE['dir']}", flush=True)
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

        # 3. Antes de ninguna potencia (§8 y §20.5): nulidad del generador solo (gate) y sesgo del estimador completo
        lados = P.preparar_lados(alt, boy)
        p_nul = P.Punto("nulidad_SI0.16", 0.16, 0.16, 0.8, 1.0)
        r_nul = P.calibrar(lados, p_nul)
        pura = P.nulidad_pura(lados, r_nul, p_nul.rho, REPLICAS_NULIDAD, p_nul.nombre)
        nul_ok = P.nulidad_cumple(pura)
        for lado, v in pura.items():
            txt = ", ".join(f"{e} {v[e]['media']:+.4f} (ee {v[e]['ee_mc']:.4f})" for e in ("P90", "P99", "max"))
            print(f"Nulidad del generador (sin ruido ni calibración), {lado}: {txt}", flush=True)
        print(f"Nulidad del generador: {'cumple' if nul_ok else 'NO CUMPLE'}", flush=True)
        t1 = time.time()
        reps_nul = correr(ruta_alt, ruta_boy, p_nul, r_nul, REPLICAS_NULIDAD, procesos_)
        seg_rep = (time.time() - t1) / len(reps_nul) * procesos_
        nul_original = P.nulidad_generador(reps_nul)
        sesgo = P.sesgo_estimador(reps_nul, efectos)
        for lado, v in sesgo["lados"].items():
            txt = ", ".join(
                f"{e} {v[e]['media']:+.4f}{'' if v[e]['estimable'] else ' (no estimable)'}"
                for e in ("P90", "P99", "max")
            )
            print(f"Sesgo del estimador (ruido, SIMEX y calibración), {lado}: {txt}", flush=True)
        for h in sesgo["hallazgos"]:
            print(f"Hallazgo: {h}", flush=True)
        n_puntos = len(P.rejilla())
        est_h = seg_rep * replicas * n_puntos / procesos_ / 3600
        print(
            f"Una réplica ≈ {seg_rep:.1f} s por proceso; rejilla completa ≈ {est_h:.1f} h con {procesos_} procesos",
            flush=True,
        )
        resumen = {
            "valores_observados_leidos": False,
            "hallazgos": [HALLAZGO_ESCENARIOS, *[f"{h} (§20.5; no se corrige)" for h in sesgo["hallazgos"]]],
            "efectos_nodo": efectos_nodo,
            "efectos_por_fuente": efectos,
            "atribuye": esc[esc["estadistico"] == "P90"].set_index("comparacion")["atribuye"].dropna().to_dict(),
            "traslado_p90": esc[esc["estadistico"] == "P90"].set_index("comparacion")["traslado"].dropna().to_dict(),
            "lados": lados_res,
            "pesos_hy2": G.pesos_hy2(alt),
            "nulidad_generador": {
                "criterio": "§20.5: generador solo, sin ruido ni calibración, sin centrar; |media| ≤ 3 EE de MC en "
                "P90, P99 y máx de los seis lados",
                "replicas": REPLICAS_NULIDAD,
                "resultado": pura,
                "cumple": nul_ok,
            },
            "prueba_original_s8": {
                "criterio": "§8 tal como se aceptó: con ruido, calibración y SIMEX, dentro del error de MC de 0",
                "replicas": len(reps_nul),
                "resultado": nul_original,
                "cumple": P.nulidad_cumple(nul_original),
                "nota": "No se usa como gate (§20.5): mide el sesgo del estimador, no el del generador.",
            },
            "sesgo_estimador": sesgo,
            "segundos_por_replica": seg_rep,
            "estimacion_horas_rejilla": est_h,
            "replicas_por_punto": replicas,
            "B_bootstrap_potencia": C.B_POTENCIA,
            "aproximaciones_declaradas": [
                "Potencia por desplazamiento con réplicas de b = 0 (§20), validada con inyección explícita en el "
                "nominal y en el punto más ruidoso.",
                "Generador simétrico (§20): m* y o* son perturbaciones independientes del modelo real.",
                "Calibración linealizada sobre el estadístico en el bootstrap (exacta en el generador).",
                "SIMEX como desplazamiento fijo de las réplicas bootstrap (no se repite dentro de cada réplica).",
                "Ruido de 1 Hz generado a nivel de superobservación (σ/√n), equivalente a la media de n puntos iid.",
                "V3b: B(P99) − B(P50) se desplaza en el efecto; validado con inyección que crece con el nivel.",
                "Término por lado de la boya en la verdad: desplazamiento exacto (el ruido de la boya es relativo).",
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
        potencias_v3, reps_por_punto = {}, {}
        for punto in P.rejilla():
            r = P.calibrar(lados, punto)
            t2 = time.time()
            reps = correr(ruta_alt, ruta_boy, punto, r, replicas, procesos_)
            print(f"{punto.nombre}: {len(reps)} réplicas en {(time.time() - t2) / 60:.1f} min", flush=True)
            if punto.nombre == P.NOMINAL:
                for prueba, alfa in PRUEBAS:
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
            for prueba, alfa in PRUEBAS:
                mde.append(
                    {
                        "punto": punto.nombre,
                        "prueba": prueba,
                        "efecto": efectos[prueba],
                        "efecto_minimo_80": P.efecto_minimo(reps, prueba, alfa, efectos[prueba]),
                    }
                )
            potencias_v3[punto.nombre] = {
                suf: (
                    P.potencia(reps, f"V3a{suf}", C.EFECTO_V3, C.ALFA_FAMILIA),
                    P.potencia(reps, f"V3b{suf}", C.EFECTO_V3, C.ALFA_FAMILIA),
                )
                for suf in ("", "_sinmax")
            }
            reps_por_punto[punto.nombre] = reps
            filas_rep += [f for i, rp in enumerate(reps) for f in _fila_replica(punto.nombre, i, rp)]
        tabla = pd.concat(tablas, ignore_index=True)

        # 5b. Validación del desplazamiento con inyección explícita (§20.2): dos puntos, los dos signos, error pareado
        valid = []
        for nombre_v in (P.NOMINAL, PUNTO_RUIDOSO):
            punto_v = next(p for p in P.rejilla() if p.nombre == nombre_v)
            r_v = P.calibrar(lados, punto_v)
            reps0 = reps_por_punto[nombre_v][:REPLICAS_VALIDACION]
            for prueba, lado, alfa, forma in VALIDACIONES:
                for mu in (1.0, -1.0, 3.0, -3.0):
                    b = mu * efectos[prueba]
                    iny = {lado: b if forma == "constante" else (b, "nivel")}
                    reps_i = correr(ruta_alt, ruta_boy, punto_v, r_v, REPLICAS_VALIDACION, procesos_, iny)
                    valid.append(P.comparar_desplazamiento(reps_i, reps0, prueba, b, alfa) | {"punto": nombre_v})
            print(f"Validación del desplazamiento en {nombre_v}: hecha", flush=True)
        ok_v = {prueba: all(v["coincide"] for v in valid if v["prueba"] == prueba) for prueba, *_ in VALIDACIONES}
        v3b_validado = ok_v["V3b"]
        for nombre_p, reps in reps_por_punto.items():
            subida[nombre_p] = {
                ("con_maximos" if suf == "" else "sin_maximos"): P.subida_esperada(
                    reps, *potencias_v3[nombre_p][suf], suf=suf, v3b_validado=v3b_validado
                )
                for suf in ("", "_sinmax")
            }

        tabla.to_csv(st / "potencia.csv", index=False, float_format="%.6g", lineterminator="\n")
        pd.DataFrame(mde).to_csv(st / "efecto_minimo.csv", index=False, float_format="%.6g", lineterminator="\n")
        pd.DataFrame(filas_rep).to_parquet(st / "replicas_potencia.parquet")
        for config, nom in subida[P.NOMINAL].items():
            if nom["supera_2_por_ciento"]:
                resumen["hallazgos"].append(
                    f"La subida esperada de la regla de variantes con sesgo nulo ({config.replace('_', ' ')}) es "
                    f"{100 * nom['subida_media']:.2f} %, mayor que el 2 % de ADR-0006 (ADR-0024, punto 2)."
                )
        if not v3b_validado:
            resumen["hallazgos"].append(
                "La potencia por desplazamiento de V3b no quedó validada: V3b no puede salir «descartada» (S-20)."
            )
        resumen |= {
            "error_tipo_I": tipo1,
            "factores_recalibrados": dict(P.FACTORES),
            "validacion_desplazamiento": valid,
            "validacion_por_prueba": ok_v,
            "subida_esperada": subida,
            "rejilla": [asdict(p) for p in P.rejilla()],
            "nominal": P.NOMINAL,
        }
        json_estricto.escribir(st / "potencia.json", resumen)
        for f in (ruta_alt, ruta_boy):
            f.unlink()
        motivos = []
        if not nul_ok:
            motivos.append("§20.5: el generador no es nulo")
        malas = [k for k, v in ok_v.items() if not v and k != "V3b"]
        if malas:
            motivos.append(f"§20.2: la potencia por desplazamiento no coincide con la inyección en {malas}")
        criterios = {"replicas": replicas, "puntos": n_puntos, "nulidad": nul_ok}
        publicada = etapa.publicar(
            st,
            salida,
            etapa.Gate(not motivos, criterios, motivos),
            raices,
            ins,
            raiz,
            semillas.semilla("f05", "potencia"),
        )
        reanudable.limpiar(AVANCE["dir"])  # solo si se publicó: un gate fallido lanza antes y conserva el avance
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validacion-potencia", description="Etapa 2 de la F5: geometría y potencia.")
    ap.add_argument("--datos")
    ap.add_argument("--nombre", default=NOMBRE)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=C.REPLICAS_POTENCIA)
    ap.add_argument("--previo", action="store_true", help="Geometría, escalón, nulidad y cronometraje; sin rejilla.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):  # consolas de Windows en cp1252: sin caída por un carácter
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = potencia_etapa(a.datos, a.nombre, a.procesos, a.replicas, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
