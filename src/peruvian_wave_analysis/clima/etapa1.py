"""Etapa 1 de la F6, ``clima-etapa1``: estratificación sin leer excesos (contrato §4, §6.1 punto 1, G4, G7, G7b).

Con datos del sitio, pero **sin leer ningún exceso ni ajustar nada a los picos**: de los picos provisionales usa solo
su fecha (mes, estrato B), su `VPED` (candidata C) y su número.

1. Insumos verificados (G1) y las cuatro ramas.
2. τ̂_e de E0 sin corregir y separación provisional max(τ̂_e, 48 h) (§3.4).
3. Picos provisionales de E0 sin corregir sobre u_tent; regla de bimodalidad de `VPED` (candidata C).
4. Conteos por estrato de B y de C en las cuatro ramas (G4: ≥ 30 por estrato; marca, no falla).
5. Para cada candidata, con la fracción f del estrato chico y λ = N / años de E0 sin corregir: frontera de σ y de ξ
   (hacia arriba y hacia abajo) con el procedimiento de §5.4 (8 semillas, 60 000).
6. Potencia de Anderson–Darling en la frontera de σ con los conteos reales, 1 000 réplicas, α por prueba
   = α_fam / k para α_fam ∈ {0.05, 0.075, 0.10}; α_fam elegido (G7: marca si alguna candidata no llega al 80 % ni
   con 0.10).
7. G7b: potencia de Anderson–Darling y de la razón de verosimilitudes en las fronteras de ξ, 1 000 réplicas. Si
   alguna llega al 80 % con α_fam, ADR-0028 no aplica: **el gate falla** y la fase se detiene (contrato §7).

Todo se publica en ``estratos_etapa1.json`` y queda congelado: la etapa principal (``clima``) verifica su manifiesto
y la huella de este código antes de leer ningún exceso. Reanudable, con avance. ``--previo`` cronometra.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .. import rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..extremos import dependencia as D
from ..validacion import reanudable
from . import descriptivos as DS
from . import estratificacion as E
from . import insumos as I
from . import picos as P
from . import simulacion_previa as SP

ETAPA1 = orquestador.Etapa("clima-etapa1", Path(__file__).resolve(), Path("."))
BASES = (0.0, -0.1)
R_POTENCIA = 1000
TROZO = 100
MIN_PICOS_ESTRATO = 30  # ADR-0006
TIPOS_FRONTERA = (E.SIGMA, E.XI_ARRIBA, E.XI_ABAJO)


def ruta_salida(raiz: Path, nombre: str = I.NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F06-clima-de-oleaje" / "etapa1" / nombre


def ruta_avance(raiz: Path, nombre: str = I.NOMBRE) -> Path:
    return Path(raiz) / "interim" / "F06-clima-de-oleaje" / "etapa1-avance" / nombre


def anios(indice) -> float:
    """Largo de la rejilla en años (365.25 días), con el último paso incluido."""
    return float((indice[-1] - indice[0] + np.timedelta64(3, "h")) / np.timedelta64(1, "D") / 365.25)


# --- Trabajos (importables por los procesos hijos) ------------------------------------------------------


def trabajo_pruebas(args) -> list[dict]:
    nombre, xi1, xi2, r, n1, n2, indices, con_lrt = args
    return [
        E.replica_pruebas(xi1, xi2, r, n1, n2, semillas.generador("f06", "etapa1", nombre, str(i)), con_lrt=con_lrt)
        for i in indices
    ]


def _trozos(n: int) -> list[list[int]]:
    return [list(range(a, min(a + TROZO, n))) for a in range(0, n, TROZO)]


# --- Conteos -------------------------------------------------------------------------------------------


def conteos_por_rama(datos: I.Datos, sep_h: float, arco) -> dict:
    """u_tent, número de picos y conteos de B y C en cada rama (solo fechas y direcciones)."""
    out = {}
    for rama, hs in datos.ramas.items():
        u = P.u_tentativo(hs)
        pk = P.picos(hs, datos.serie, u, sep_h)
        cB = P.conteos(pk["estrato_B"], (P.B_INVIERNO, P.B_VERANO))
        fila = {"u_tent": u, "n_picos": int(len(pk)), "anios": anios(hs.index), "B": cB}
        if arco is not None:
            fila["C"] = P.conteos(P.estrato_C(pk["VPED"], arco), (P.C_MAYOR, P.C_MENOR))
        out[rama] = fila
    return out


def candidatas(conteos: dict) -> list[dict]:
    """B siempre; C si es bimodal. Conteos y fracción del estrato chico en E0 sin corregir (la rama que decide)."""
    base = conteos[I.RAMA_DECISION]
    out = []
    for nombre in ("B", "C"):
        if nombre not in base:
            continue
        chico, f = P.fraccion_chica(base[nombre])
        n = sum(base[nombre].values())
        out.append({"candidata": nombre, "estrato_chico": chico, "f": f, "n1": n - base[nombre][chico],
                    "n2": base[nombre][chico], "N": n, "lambda": n / base["anios"]})  # fmt: skip
    return out


def g4(conteos: dict) -> list[str]:
    marcas = []
    for rama, fila in conteos.items():
        for cand in ("B", "C"):
            for estrato, n in fila.get(cand, {}).items():
                if n < MIN_PICOS_ESTRATO:
                    marcas.append(f"{rama}, {cand}, {estrato}: {n} picos (< {MIN_PICOS_ESTRATO})")
    return marcas


# --- Potencia ------------------------------------------------------------------------------------------


def potencia_sigma(reps: list[dict], k: int) -> dict[float, float]:
    p = [o["ad_p"] for o in reps]
    return {a: E.tasa_rechazo(p, a / k) for a in E.ALFAS_FAM}


def potencia_forma(reps: list[dict], alfa_fam: float, k: int) -> dict:
    p, lrt = [o["ad_p"] for o in reps], [o["lrt"] for o in reps]
    return {
        "ad_alfa_fam": E.tasa_rechazo(p, alfa_fam), "ad_alfa_por_prueba": E.tasa_rechazo(p, alfa_fam / k),
        "lrt_alfa_fam": E.rechazo_lrt(lrt, alfa_fam), "lrt_alfa_por_prueba": E.rechazo_lrt(lrt, alfa_fam / k),
        "lrt_no_calculable": int(np.sum(~np.isfinite(np.asarray(lrt, float)))),
    }  # fmt: skip


def detiene_g7b(forma: list[dict]) -> list[str]:
    """Casos con potencia ≥ 80 % frente a la forma con α_fam (el nivel más liberal: la regla para detenerse)."""
    return [
        f"{c['candidata']}, ξ₁ = {c['xi1']:g}, {c['tipo']} (ξ₂ = {c['xi2']:g}): {k} = {c[k]:.3f}"
        for c in forma
        for k in ("ad_alfa_fam", "lrt_alfa_fam")
        if c[k] >= E.POTENCIA_OBJETIVO
    ]


# --- Etapa ----------------------------------------------------------------------------------------------


def _preparar(raiz: Path, nombre: str, aviso):
    rutas_ins = I.rutas_insumos(raiz, nombre)
    etapa.exigir_insumos(rutas_ins)
    datos = I.cargar(raiz, rutas_ins, nombre)
    aviso("insumos verificados (G1); ramas: " + ", ".join(f"{k} ({len(v)} pasos)" for k, v in datos.ramas.items()))
    tau = D.tau_bloques(D.bloques_anuales(datos.ramas[I.RAMA_DECISION]))
    sep = P.separacion(tau.tau_e_h)
    aviso(f"tau_e = {tau.tau_e_h:g} h, tau_int = {tau.tau_int_h:.1f} h; separación provisional {sep:g} h")
    hs0 = datos.ramas[I.RAMA_DECISION]
    pk0 = P.picos(hs0, datos.serie, P.u_tentativo(hs0), sep)
    bim = DS.bimodalidad(pk0["VPED"].to_numpy(float))
    aviso(f"bimodalidad de VPED: {bim.n_modas} modas; C {'es' if bim.candidata else 'no es'} candidata")
    cont = conteos_por_rama(datos, sep, bim.cortes_grados if bim.candidata else None)
    return rutas_ins, datos, tau, sep, bim, cont


def etapa1(raiz_datos=None, procesos_: int = 4, previo: bool = False, nombre: str = I.NOMBRE) -> Path | None:
    raiz = rutas.directorio_datos(raiz_datos)
    t0 = time.time()

    def aviso(t: str) -> None:
        print(f"[{(time.time() - t0) / 60:5.1f} min] {t}", flush=True)

    if previo:
        *_, cont = _preparar(raiz, nombre, aviso)
        cands = candidatas(cont)
        t1 = time.time()
        SP.trabajo_frontera((E.SIGMA, 0.0, cands[0]["f"], 2.0, cands[0]["lambda"]))
        t_fr = time.time() - t1
        t1 = time.time()
        c = cands[0]
        trabajo_pruebas(("previo", 0.0, 0.2, 1.0, c["n1"], c["n2"], list(range(20)), True))
        t_rep = (time.time() - t1) / 20
        filas = len(cands) * len(BASES) * len(TIPOS_FRONTERA)
        total = (filas * 15 * t_fr + len(cands) * len(BASES) * 3 * R_POTENCIA * t_rep) / max(procesos_, 1) / 60
        aviso(f"previo: {len(cands)} candidata(s); punto de frontera {t_fr:.1f} s, réplica con LRT {t_rep:.3f} s; "
              f"total estimado unos {total:.0f} min con {procesos_} procesos (cota: 15 oleadas por fila)")  # fmt: skip
        return None

    salida = ruta_salida(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA1)
    avance = ruta_avance(raiz, nombre) / reanudable.clave(orquestador.huella_etapa(ETAPA1), TROZO, R_POTENCIA)
    with etapa.staging_protegido(salida, raices) as st:
        rutas_ins, datos, tau, sep, bim, cont = _preparar(raiz, nombre, aviso)
        aviso(f"avance reanudable en {avance}")
        cands = candidatas(cont)
        k = len(cands)
        for c in cands:
            aviso(f"candidata {c['candidata']}: N = {c['N']}, estrato chico {c['estrato_chico']} "
                  f"con {c['n2']} picos (f = {c['f']:.3f}), lambda = {c['lambda']:.2f}/año")  # fmt: skip
        marcas_g4 = g4(cont)

        # Fronteras para la fracción real
        filas = [(t, xi1, c["f"]) for c in cands for xi1 in BASES for t in TIPOS_FRONTERA]
        por_lam: dict = {}
        rejillas: dict = {}
        for c in cands:  # cada candidata con su λ (iguales salvo que C tenga otro N; se recorren por separado)
            fr, rej = SP.fronteras(avance / f"frontera_{c['candidata']}", procesos_,
                                   [f for f in filas if f[2] == c["f"]], c["lambda"], aviso)  # fmt: skip
            por_lam[c["candidata"]] = fr
            rejillas[c["candidata"]] = rej
        for nom, frs in por_lam.items():
            for x in frs:
                aviso(f"frontera {nom} {x.tipo} xi1={x.xi1:g}: r={x.r} xi2={x.xi2} ({x.media_pct:+.2f} %) {x.rotulo}")

        # Potencia frente a la escala
        tareas = []
        for c in cands:
            for x in por_lam[c["candidata"]]:
                if x.tipo != E.SIGMA:
                    continue
                n = f"sigma_{c['candidata']}_{x.xi1:g}"
                tareas += [(f"{n}_{tr[0]:04d}", (n, x.xi1, x.xi1, x.r, c["n1"], c["n2"], tr, False))
                           for tr in _trozos(R_POTENCIA)]  # fmt: skip
        aviso(f"potencia de Anderson-Darling en la frontera de sigma: {len(tareas)} tareas")
        res = reanudable.ejecutar(trabajo_pruebas, tareas, avance / "potencia", procesos_, lambda t: aviso(f"  {t}"))
        reps: dict = {}
        for (_, a), parte in zip(tareas, res, strict=True):
            reps.setdefault(a[0], []).extend(parte)
        pot_sigma = {}
        for c in cands:
            for xi1 in BASES:
                pot_sigma[(c["candidata"], xi1)] = potencia_sigma(reps[f"sigma_{c['candidata']}_{xi1:g}"], k)
        eleccion = E.elegir_alfa_fam(pot_sigma)
        aviso(f"alfa de familia = {eleccion.alfa_fam:g} (alcanza el 80 %: {eleccion.alcanza})")
        marca_g7 = [f"{c} con xi1 = {b:g}" for c, b in eleccion.bajo_objetivo_con_tope]

        # G7b: potencia frente a la forma
        tareas = []
        casos_forma = []
        for c in cands:
            for x in por_lam[c["candidata"]]:
                if x.tipo == E.SIGMA or x.xi2 is None:
                    continue
                n = f"forma_{c['candidata']}_{x.tipo}_{x.xi1:g}"
                casos_forma.append((n, c, x))
                tareas += [(f"{n}_{tr[0]:04d}", (n, x.xi1, x.xi2, 1.0, c["n1"], c["n2"], tr, True))
                           for tr in _trozos(R_POTENCIA)]  # fmt: skip
        aviso(f"G7b, potencia frente a la forma: {len(casos_forma)} casos, {len(tareas)} tareas")
        res = reanudable.ejecutar(trabajo_pruebas, tareas, avance / "forma", procesos_, lambda t: aviso(f"  {t}"))
        reps_f: dict = {}
        for (_, a), parte in zip(tareas, res, strict=True):
            reps_f.setdefault(a[0], []).extend(parte)
        forma = [
            {"candidata": c["candidata"], "xi1": x.xi1, "tipo": x.tipo, "xi2": x.xi2}
            | potencia_forma(reps_f[n], eleccion.alfa_fam, k)
            for n, c, x in casos_forma
        ]
        detiene = detiene_g7b(forma)

        resumen = {
            "excesos_leidos": False,
            "que_se_leyo": "Fechas, meses, VPED y número de los picos provisionales; u_tent y τ de la serie. "
            "Ningún exceso ni ajuste a los picos.",
            "insumos": {k_: etapa.ruta_relativa(v, raiz) for k_, v in rutas_ins.items()},
            "tau": {"tau_e_h": tau.tau_e_h, "tau_int_h": tau.tau_int_h, "separacion_provisional_h": sep},
            "bimodalidad_vped": {
                "candidata": bim.candidata, "n_modas": bim.n_modas, "modas_grados": list(bim.modas_grados),
                "masas": list(bim.masas), "arco_sector_menor": list(bim.cortes_grados) if bim.cortes_grados else None,
                "mas_de_dos_modas": bim.mas_de_dos_modas, "n_picos": bim.n,
            },
            "conteos": cont,
            "g4_estratos_con_menos_de_30": marcas_g4,
            "candidatas": cands,
            "fronteras": {nom: [x.como_dict() for x in frs] for nom, frs in por_lam.items()},
            "rejillas_frontera": rejillas,
            "potencia_sigma": [
                {"candidata": c, "xi1": b, "alfa_fam": {f"{a:g}": v for a, v in p.items()}, "k": k}
                for (c, b), p in pot_sigma.items()
            ],
            "alfa_fam": {
                "valor": eleccion.alfa_fam, "alcanza_80": eleccion.alcanza, "k": eleccion.k,
                "alfa_por_prueba": eleccion.alfa_fam / eleccion.k,
                "minima_potencia_por_nivel": {f"{a:g}": v for a, v in eleccion.por_nivel.items()},
                "g7_marca": marca_g7,
                "criterio": "Menor α_fam de {0.05, 0.075, 0.10} con potencia ≥ 80 % en todas las candidatas y "
                "ambas bases, con α por prueba = α_fam / k (§4); si ninguno, 0.10 y marca G7.",
            },
            "g7b": {
                "potencia_forma": forma, "detiene": detiene,
                "criterio": "Si Anderson–Darling o la razón de verosimilitudes llegan al 80 % con α_fam en alguna "
                "frontera de ξ, ADR-0028 no aplica y la fase se detiene (§7, G7b). Se informa también la potencia "
                "con α_fam / k.",
            },
            "repeticiones": {"replicas": R_POTENCIA, "semillas_frontera": E.SEMILLAS_FRONTERA, "n_grande": E.N_GRANDE},
            "segundos": round(time.time() - t0),
        }  # fmt: skip
        json_estricto.escribir(st / "estratos_etapa1.json", resumen)
        if marca_g7:
            aviso("AVISO G7: no llegan al 80 % ni con alfa_fam = 0.10: " + "; ".join(marca_g7))
        if marcas_g4:
            aviso("AVISO G4: " + "; ".join(marcas_g4))
        motivos = [f"G7b: {d}" for d in detiene]
        criterios = {"alfa_fam": eleccion.alfa_fam, "candidatas": [c["candidata"] for c in cands],
                     "g7_marca": marca_g7, "g4_marca": marcas_g4, "g7b_detiene": detiene}  # fmt: skip
        publicada = etapa.publicar(
            st, salida, etapa.Gate(not motivos, criterios, motivos), raices, rutas_ins, raiz,
            semillas.semilla("f06", "etapa1"),
        )  # fmt: skip
        reanudable.limpiar(avance)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clima-etapa1", description="Etapa 1 de la F6: estratificación sin excesos")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--previo", action="store_true", help="Cronometra; no publica.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = etapa1(a.datos, a.procesos, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    if salida:
        print(f"Publicado en {salida}")
    return 0
