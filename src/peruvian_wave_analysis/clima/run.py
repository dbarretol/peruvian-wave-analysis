"""Etapa principal de la F6, ``clima`` (contrato §2, §3, §4, §6 y §7).

Orden, para que los excesos no se lean antes de tiempo:

1. **G7 antes de leer excesos:** la etapa 1 está publicada, con su gate aprobado, sin cambios en sus archivos, con la
   huella de su código igual a la actual, ``excesos_leidos = False`` y los mismos insumos (hash) que esta corrida.
   Si algo falla, se detiene sin leer nada más. Lo mismo con la etapa 0 (factor de concordancia).
2. Picos provisionales de las cuatro ramas con la separación de la etapa 1; sus conteos deben ser los publicados.
3. Prueba de Anderson–Darling con el α congelado; desenlace y escenarios (§6.1; ADR-0028, punto 7; ADR-0029).
4. Dependencia con IC por bloques (B = 999, reanudable), descriptivos, GEV por rama, figuras y gate.

**Población mixta (§6.2):** si la señal se activa, la fase se detiene antes de publicar y avisa al usuario.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..extremos import dependencia as D
from ..validacion import reanudable
from . import clase_fisica as CF
from . import escalas as ES
from . import estratos_sitio as ET
from . import etapa0 as E0
from . import etapa1 as E1
from . import figuras as F
from . import gev_ramas as GV
from . import insumos as I
from . import picos as P
from . import tablas as TB

ETAPA = orquestador.Etapa("clima", Path(__file__).resolve(), Path("."))
ANIO_COMPLETO_FINAL = 2025
EXCLUIR_MENOS_DE_30 = True  # ADR-0029 (aceptada el 2026-11-19): lo que depende de un estrato con < 30 picos no entra


def ruta_salida(raiz: Path, nombre: str = I.NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F06-clima-de-oleaje" / nombre


def ruta_avance(raiz: Path, nombre: str = I.NOMBRE) -> Path:
    return Path(raiz) / "interim" / "F06-clima-de-oleaje" / "clima-avance" / nombre


# --- G7: la etapa 1 congelada ------------------------------------------------------------------------------


def verificar_etapa1(raiz: Path, rutas_ins: dict[str, Path], nombre: str = I.NOMBRE) -> tuple[dict, dict]:
    """(estratos_etapa1.json, comprobación). ``GateFallido`` si algo no cuadra; no se lee ningún exceso antes."""
    d = E1.ruta_salida(raiz, nombre)
    try:
        man = etapa.exigir_publicada(d)
    except Exception as exc:  # noqa: BLE001 — cualquier falla de la etapa 1 detiene aquí
        raise GateFallido(f"G7: la etapa 1 no está publicada o cambió: {exc}") from exc
    if not man["gate"]["aprobado"]:
        raise GateFallido("G7: la etapa 1 tiene el gate fallido.")
    actual = orquestador.huella_etapa(E1.ETAPA1)
    if man["huella_codigo"] != actual:
        raise GateFallido(
            "G7: el código de la etapa 1 cambió después de publicarla "
            f"(huella publicada {man['huella_codigo'][:12]}, actual {actual[:12]}); hay que repetirla."
        )
    hashes = etapa.exigir_insumos(rutas_ins)
    distintos = [k for k, v in man["insumos"].items() if k in hashes and v["sha256"] != hashes[k]]
    if distintos or set(man["insumos"]) != set(hashes):
        raise GateFallido(f"G7: los insumos cambiaron desde la etapa 1: {distintos or 'otro conjunto'}")
    r = json_estricto.leer(d / "estratos_etapa1.json")
    if r.get("excesos_leidos") is not False:
        raise GateFallido("G7: estratos_etapa1.json no declara excesos_leidos = False.")
    if r["g7b"]["detiene"]:
        raise GateFallido(f"G7b: {r['g7b']['detiene']}")
    return r, {
        "estratos_etapa1_sha256": man["salidas"]["estratos_etapa1.json"],
        "huella_codigo_etapa1": man["huella_codigo"],
        "comprobado_antes_de_leer_excesos": True,
    }


def verificar_etapa0(raiz: Path) -> dict:
    d = E0.ruta_salida(raiz)
    man = etapa.exigir_publicada(d)
    if not man["gate"]["aprobado"]:
        raise GateFallido("La etapa 0 sintética tiene el gate fallido.")
    actual = orquestador.huella_etapa(E0.ETAPA0)
    if man["huella_codigo"] != actual:
        raise GateFallido(
            "El código de la etapa 0 cambió después de publicarla "
            f"(huella publicada {man['huella_codigo'][:12]}, actual {actual[:12]}); hay que repetirla."
        )
    r = json_estricto.leer(d / "etapa0.json")
    if r.get("valores_del_sitio_leidos") is not False:
        raise GateFallido("etapa0.json no declara valores_del_sitio_leidos = False.")
    return r


# --- Piezas --------------------------------------------------------------------------------------------------


def picos_ramas(datos: I.Datos, sep_h: float) -> dict[str, tuple[float, pd.DataFrame]]:
    return {rama: (u := P.u_tentativo(hs), P.picos(hs, datos.serie, u, sep_h, datos.dudosos))
            for rama, hs in datos.ramas.items()}  # fmt: skip


def comprobar_conteos(pks: dict, r1: dict) -> list[str]:
    out = []
    for rama, (u, pk) in pks.items():
        c = r1["conteos"][rama]
        cb = P.conteos(pk["estrato_B"], (P.B_INVIERNO, P.B_VERANO))
        if len(pk) != c["n_picos"] or cb != c["B"] or not np.isclose(u, c["u_tent"]):
            out.append(f"{rama}: {len(pk)} picos y B = {cb}; la etapa 1 publicó {c['n_picos']} y {c['B']}")
        if "C" in c:
            cc = P.conteos(pk["estrato_C"], (P.C_MAYOR, P.C_MENOR)) if "estrato_C" in pk else None
            if cc != c["C"]:
                out.append(f"{rama}: C = {cc}; la etapa 1 publicó {c['C']}")
    return out


COLUMNA = {"B": ("estrato_B", (P.B_INVIERNO, P.B_VERANO)), "C": ("estrato_C", (P.C_MAYOR, P.C_MENOR))}


def marcar_C(pks: dict, arco) -> None:
    """Agrega ``estrato_C`` a los picos de cada rama si C es candidata (arco de la etapa 1)."""
    if arco is None:
        return
    for _, pk in pks.values():
        pk["estrato_C"] = P.estrato_C(pk["VPED"], tuple(arco))


def elegir_establecida(cands: dict) -> str | None:
    """Con dos establecidas, la de menor valor p; con empate en la cota de 0.001, la de mayor estadístico (§6.1)."""
    est = [n for n, c in cands.items() if c["desenlace"] == ET.ESTABLECIDO]
    if not est:
        return None
    return min(est, key=lambda n: (max(cands[n]["prueba"]["p"], ET.COTA_P[0]), -cands[n]["prueba"]["estadistico"]))


def escenarios_familia(cands: dict) -> list[dict]:
    """Sin estratos, más uno por la establecida elegida y uno por cada no concluyente (§6.1, punto 2)."""
    out = [{"nombre": "sin_estratos", "envolvente": True, "rotulo": ""}]
    elegida = elegir_establecida(cands)
    for n, c in cands.items():
        if n == elegida or c["desenlace"] == ET.NO_CONCLUYENTE:
            for e in c["escenarios"][1:]:
                out.append(e | {"nombre": f"{e['nombre']}_{n}", "candidata": n, "desenlace": c["desenlace"]})
    return out


def estratificacion(pks: dict, r1: dict, anios_por_rama: dict) -> dict:
    alfa = r1["alfa_fam"]
    k = alfa["k"]
    pk0 = pks[I.RAMA_DECISION][1]
    out = {"alfa_fam": alfa["valor"], "k": k, "candidatas": {}}
    menos_30 = r1["g4_estratos_con_menos_de_30"]
    for c in r1["candidatas"]:
        nom = c["candidata"]
        col, estratos = COLUMNA[nom]
        pr = ET.prueba(pk0, col, estratos, alfa["valor"] / k)
        pot = [p["alfa_fam"][f"{alfa['valor']:g}"] for p in r1["potencia_sigma"] if p["candidata"] == nom]
        des = ET.desenlace(pr["rechaza"], pot)
        chico = any(f", {nom}, " in m for m in menos_30)
        modelos = {rama: ET.modelos(pk, col, estratos, u, anios_por_rama[rama]) for rama, (u, pk) in pks.items()}
        alarmas = {rama: a for rama, m in modelos.items() if (a := ET.alarma_adr0029(m))}
        out["candidatas"][nom] = {
            "prueba": pr, "potencia_etapa1_con_alfa_fam": pot, "desenlace": des,
            "escenarios": ET.escenarios(des, chico, EXCLUIR_MENOS_DE_30),
            "estrato_con_menos_de_30": chico, "marcas_g4": [m for m in menos_30 if f", {nom}, " in m],
            "cota_alta_xi_por_estrato": (
                "fuera del valor alto: estrato con menos de 30 picos (ADR-0029); sensibilidad descriptiva"
                if chico and EXCLUIR_MENOS_DE_30 else "entra en el valor alto (ADR-0028, punto 3)"
            ) if nom == "B" else "no aplica (la cota alta usa B)",
            "alarma_adr0029": alarmas,
            "modelos_por_rama": modelos,
            "dentro_de_estratos": {rama: ET.dentro_de_estratos(pk, col, estratos) for rama, (u, pk) in pks.items()},
        }  # fmt: skip
    out["escenarios"] = escenarios_familia(out["candidatas"])
    out["establecida_elegida"] = elegir_establecida(out["candidatas"])
    return out


def separacion_cola(ic: dict) -> float:
    """La mayor de las escalas de la cola (T_(C) a P99 de invierno o el corte), al paso de 3 h hacia arriba."""
    v = [ic[f"T_C_h_q{ES.Q_PRINCIPAL:g}"]["punto"], ic["corte_h"]["punto"]]
    v = [x for x in v if np.isfinite(x)]
    return float(np.ceil(max(v) / P.PASO_H) * P.PASO_H) if v else float("nan")


# --- Etapa ---------------------------------------------------------------------------------------------------


def clima(raiz_datos=None, procesos_: int = 4, previo: bool = False, nombre: str = I.NOMBRE,
          b: int = ES.B) -> Path | None:  # fmt: skip
    raiz = rutas.directorio_datos(raiz_datos)
    t0 = time.time()

    def aviso(t: str) -> None:
        print(f"[{(time.time() - t0) / 60:5.1f} min] {t}", flush=True)

    def preparar():
        rutas_ins = I.rutas_insumos(raiz, nombre)  # G1
        r0 = verificar_etapa0(raiz)
        r1, g7 = verificar_etapa1(raiz, rutas_ins, nombre)
        aviso("G7: etapas 0 y 1 publicadas, sin cambios, con la misma huella de código y los mismos insumos")
        datos = I.cargar(raiz, rutas_ins, nombre)
        bl = D.bloques_anuales(datos.ramas[I.RAMA_DECISION])
        return rutas_ins, r0, r1, g7, datos, bl, ES.umbrales(datos.ramas[I.RAMA_DECISION])

    if previo:
        _, _, _, _, _, bl, u = preparar()
        t1 = time.time()
        ES.vector(bl, u)
        t_rep = time.time() - t1
        aviso(f"previo: una réplica de la dependencia {t_rep:.1f} s; B = {b} → unos "
              f"{t_rep * b / max(procesos_, 1) / 60:.0f} min con {procesos_} procesos, más unos minutos de "
              "tablas, GEV y figuras")  # fmt: skip
        return None

    salida = ruta_salida(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA)
    avance = ruta_avance(raiz, nombre) / reanudable.clave(orquestador.huella_etapa(ETAPA), b)
    F.PROBLEMAS.clear()
    with etapa.staging_protegido(salida, raices) as st:
        # un fallo de G1 o G7 aquí dentro borra la salida publicada anterior (§7)
        rutas_ins, r0, r1, g7, datos, bl, u = preparar()
        sep = float(r1["tau"]["separacion_provisional_h"])
        figs = st / "figuras"
        figs.mkdir()
        # --- picos y conteos -----------------------------------------------------------------------------
        pks = picos_ramas(datos, sep)
        marcar_C(pks, r1["bimodalidad_vped"]["arco_sector_menor"])
        malos = comprobar_conteos(pks, r1)
        if malos:
            raise GateFallido("Los picos no coinciden con la etapa 1: " + "; ".join(malos))
        aviso("picos provisionales de las cuatro ramas: conteos iguales a los de la etapa 1")
        anios = {rama: E1.anios(hs.index) for rama, hs in datos.ramas.items()}
        u0, pk0 = pks[I.RAMA_DECISION]

        # --- estratificación (lee excesos) -----------------------------------------------------------------
        est = estratificacion(pks, r1, anios)
        for nom, c in est["candidatas"].items():
            aviso(f"{nom}: Anderson-Darling p = {c['prueba']['p_publicado']} con alfa = "
                  f"{c['prueba']['alfa_por_prueba']:g}; desenlace: {c['desenlace']}")  # fmt: skip
            for rama, a in c["alarma_adr0029"].items():
                aviso(f"AVISO ADR-0029 ({rama}): " + "; ".join(a))

        # --- dependencia -----------------------------------------------------------------------------------
        aviso(f"dependencia: punto y {b} réplicas por años")
        punto = ES.vector(bl, u)
        reps = ES.replicas(bl, u, procesos_, avance / "dependencia", lambda t: aviso(f"  {t}"), b)
        ic = ES.intervalos(punto, reps)
        sin_ic, fuera_ic = ES.g6(ic)
        tau = D.tau_bloques(bl, kmax=ES.KMAX_TAU)
        conc = ES.concordancia(ic, float(r0["factor_concordancia"]), bool(r0["regla_no_discrimina"]), tau.tau_e_h)
        anual = ES.por_anio(bl, u[ES.Q_PRINCIPAL])
        anual = anual[anual["anio"] <= ANIO_COMPLETO_FINAL]  # 2026 tiene 8 meses: no entra en la autocorrelación
        anual_e1 = anual[anual["anio"] >= ES.INICIO_ASIMILACION.year + 1]
        rachas = ES.rachas_tabla(datos.ramas[I.RAMA_DECISION], u[ES.Q_PRINCIPAL])
        punto_corte = D.corte_natural(D.huecos_bloques(bl, u[ES.Q_PRINCIPAL], ES.INVIERNO) * D.PASO_H)
        if punto_corte.mas_de_dos_modas:
            aviso(f"AVISO §3.5: la densidad de los huecos tiene {punto_corte.n_modas} modas (más de dos)")
        dep = {
            "umbrales_m": {f"{q:g}": v for q, v in u.items()},
            "tau": {"tau_e_h": tau.tau_e_h, "tau_int_h": tau.tau_int_h, "kmax_h": ES.KMAX_TAU * D.PASO_H,
                    "tau_int_definido": bool(np.isfinite(tau.tau_int_h)),
                    "acf_en_dias": {str(dd): float(tau.acf[int(dd * 8)]) for dd in (1, 2, 5, 10, 30, 60, 90)}},
            "ic": ic, "B": b, "g6_sin_ic_finito": sin_ic, "g6_ic_sin_punto": fuera_ic,
            "sensibilidades": ES.sensibilidades(bl, datos.ramas[I.RAMA_DECISION], u),
            "por_tramo": ES.por_tramo(datos.ramas[I.RAMA_DECISION], u),
            "autocorrelacion_entre_anios": ES.autocorrelacion_entre_anios(anual),
            "autocorrelacion_entre_anios_desde_1994": ES.autocorrelacion_entre_anios(anual_e1),
            "nota_autocorrelacion": "Con toda la serie mezcla el escalón de 1993 (F4); dentro de E1 mide la "
            "dependencia entre años del régimen actual. 2026 (8 meses) no entra.",
            "concordancia": conc,
            "corte_natural": {"corte_h": punto_corte.corte_h, "n_modas": punto_corte.n_modas,
                              "frac_segunda": punto_corte.frac_segunda,
                              "mas_de_dos_modas": punto_corte.mas_de_dos_modas, "n_huecos": punto_corte.n_huecos},
            "nota_umbral": "Umbrales fijos en las réplicas (IC condicional al umbral).",
        }  # fmt: skip
        json_estricto.escribir(st / "dependencia.json", dep)
        rachas.to_csv(st / "rachas.csv", index=False)
        aviso(
            f"dependencia: theta(P99) = {ic['theta_q0.99']['punto']:.3f}, T_C = {ic['T_C_h_q0.99']['punto']:g} h, "
            f"corte = {ic['corte_h']['punto']:g} h, extremograma = {ic['escala_extremograma_h']['punto']:g} h"
        )

        # --- población mixta -------------------------------------------------------------------------------
        arco = r1["bimodalidad_vped"]["arco_sector_menor"]
        sep2 = separacion_cola(ic)
        por_sep = {f"{sep:g} h": pk0}
        if arco is not None and np.isfinite(sep2):
            por_sep[f"{sep2:g} h"] = P.picos(datos.ramas[I.RAMA_DECISION], datos.serie, u0, sep2, datos.dudosos)
        mixta = ET.poblacion_mixta(por_sep, arco) | {"separaciones_h": [sep, sep2]}
        if mixta["senal"]:
            raise GateFallido("Señal de población mixta (§6.2): decide el usuario. " + str(mixta))

        # --- descriptivos ----------------------------------------------------------------------------------
        serie = datos.serie
        cm = TB.clima_mensual(serie, datos.ramas)
        cm.to_csv(st / "clima_mensual.csv", index=False)
        ros = TB.rosas(serie, u0)
        cf = TB.clase_fisica(serie)
        cf.to_parquet(st / "clase_fisica.parquet")
        g2 = CF.comprobar_g2(cf)
        dom = TB.dominancia(serie, cf, pk0)
        dom.to_csv(st / "dominancia.csv", index=False)
        TB.picos_tabla(pk0).to_csv(st / "picos_provisionales.csv", index=False)
        om = TB.origen_mayores(pk0, serie)
        om.to_csv(st / "origen_mayores.csv", index=False)
        tpc = TB.tp_condicional(serie, pk0)
        tpc.to_csv(st / "tp_condicional.csv", index=False)
        per, per_res = TB.peralte(serie, u0)
        per.to_csv(st / "peralte.csv", index=False)
        pares = TB.pares_cercanos(pk0, rachas, sep)
        dudoso = pares.assign(motivo="candidato por cercanía al límite de separación; revisión visual pendiente")
        dudoso.to_csv(st / "declustering_dudoso.csv", index=False)
        sens_u = {f"{q:g}": {rama: int(len(P.picos(hs, serie, P.u_tentativo(hs, q), sep))) for rama, hs in
                             datos.ramas.items()} for q in P.Q_SENSIBILIDAD}  # fmt: skip
        aviso("descriptivos escritos")

        # --- GEV -------------------------------------------------------------------------------------------
        m0, meses_max = GV.mes_inicio(datos.ramas[I.RAMA_DECISION])
        gev, mxs = GV.por_rama(datos.ramas, m0)
        gev.to_csv(st / "gev_bloques.csv", index=False)
        g5 = GV.g5(gev)
        aviso(f"GEV: bloque desde el mes {m0}; " + "; ".join(f"{r.rama} xi = {r.xi:.3f} RL100 = {r.rl100:.2f} m"
                                                            for r in gev.itertuples() if r.valido))  # fmt: skip

        # --- figuras ---------------------------------------------------------------------------------------
        F.rosa(ros, figs / "rosa_direccion.png")
        F.ciclo_anual(cm, figs / "ciclo_anual.png")
        F.dominancia(dom, figs / "dominancia.png")
        F.tp_hs(serie, pk0, figs / "tp_hs.png")
        F.acf(tau.acf, tau.tau_e_h, figs / "acf.png")
        F.theta_u(ic, ES.CUANTILES_THETA, figs / "theta_u.png")
        F.huecos(D.huecos_bloques(bl, u[ES.Q_PRINCIPAL], ES.INVIERNO) * D.PASO_H, ic["corte_h"]["punto"],
                 figs / "huecos_entre_rachas.png")  # fmt: skip
        ex = D.extremograma_bloques(bl, u[ES.Q_PRINCIPAL], ES.INVIERNO, kmax=ES.KMAX_EXTREMOGRAMA)
        F.extremograma(ex.rho, ex.ref, ex.escala_h, figs / "extremograma.png")
        F.gev(gev, mxs, figs / "gev.png")
        F.picos_estrato(pk0, figs / "picos_por_estrato.png")
        for fila in om.itertuples():
            F.evento(serie, fila.t_pico, u0, sep, f"Mayor n.º {fila.orden}: Hs = {fila.Hs:.2f} m",
                     figs / "eventos" / f"mayor_{fila.orden:02d}.png")  # fmt: skip
        for i, fila in enumerate(pares.itertuples(), 1):
            otro = None if pd.isna(fila.t_pico_siguiente) else fila.t_pico_siguiente
            F.evento(serie, fila.t_pico, u0, sep, f"{fila.tipo}: hueco {fila.hueco_h:.0f} h",
                     figs / "eventos" / f"{fila.tipo.replace(' ', '_')}_{i:02d}.png", otro)  # fmt: skip
        aviso(f"figuras: {len(list(figs.rglob('*.png')))}")

        # --- resumen y gate --------------------------------------------------------------------------------
        top_viento = om[om["clase"].isin(["mar de viento", "mixto"])]
        anio_pico = pk0.index.year
        a0, a1 = int(datos.ramas[I.RAMA_DECISION].index[0].year), ES.INICIO_ASIMILACION.year
        n_antes, n_desp = int((anio_pico < a1).sum()), int(((anio_pico >= a1) & (anio_pico <= 2025)).sum())
        picos_por_tramo = {
            "antes_de_1993": {"picos": n_antes, "anios": a1 - a0, "por_anio": n_antes / (a1 - a0)},
            "1993_2025": {"picos": n_desp, "anios": 2026 - a1, "por_anio": n_desp / (2026 - a1)},
        }
        cascada = []
        for nom, c in est["candidatas"].items():
            fuera = [e for e in c["escenarios"][1:] if not e["envolvente"]]
            if c["desenlace"] == ET.ESTABLECIDO and fuera:
                m = c["modelos_por_rama"][I.RAMA_DECISION]
                sig = m["xi_comun"]["sigma"]
                cascada.append(
                    f"{nom}: heterogeneidad de escala establecida (σ del estrato chico / σ del grande = "
                    f"{sig[1] / sig[0]:.2f}) y no aplicada: estrato con menos de 30 picos (ADR-0029, punto 5). "
                    f"ΔRL100 con ξ común en E0: {m['xi_comun']['delta_pct']:+.1f} %."
                )
                if sig[1] < sig[0]:
                    cascada.append(
                        f"{nom}: el estrato chico es más liviano (razón de σ < 1); la frontera y la potencia de la "
                        "etapa 1 solo simularon un estrato chico más pesado (r de 1.1 a 4)."
                    )
        swell_sur = bool((om["fenomeno"].str.startswith("swell del sur")).mean() > 0.5)
        estratos_json = {
            "conteos": r1["conteos"], "etapa1": g7, "estratificacion": est, "poblacion_mixta": mixta,
            "sensibilidad_umbral_n_picos": sens_u,
        }  # fmt: skip
        json_estricto.escribir(st / "estratos.json", estratos_json)
        g3 = list(F.PROBLEMAS)
        candB = est["candidatas"]["B"]
        resumen = {
            "decisiones": {
                "estratificacion": {"B": candB["desenlace"],
                                    "C": est["candidatas"].get("C", {}).get("desenlace", "no candidata"),
                                    "escenarios": est["escenarios"],
                                    "cota_alta_xi_por_estrato": candB["cota_alta_xi_por_estrato"]},
                "poblacion": mixta["veredicto"],
                "covariables_f8": ET.covariables(pk0, swell_sur),
            },
            "separacion_provisional_h": sep, "tau_e_h": tau.tau_e_h, "tau_int_h": tau.tau_int_h,
            "escalas_cola": conc, "u_tent_m": {rama: u_ for rama, (u_, _) in pks.items()},
            "picos": {rama: int(len(pk)) for rama, (_, pk) in pks.items()},
            "mayores_mar_de_viento_o_mixto": int(len(top_viento)),
            "senal_f10": (
                "La cola mezcla Tp de swell y de mar de viento: entre los 20 mayores, "
                + "; ".join(f"{r.t_pico} ({r.clase}, Tp {r.VTPK:.1f} s)" for r in top_viento.itertuples())
                + ". Para el modelo de Tp dado Hs de la F10."
            )
            if len(top_viento)
            else None,
            "cascada_f12": cascada,
            "picos_por_tramo": picos_por_tramo,
            "aviso_alcance": bool(len(top_viento)),
            "limitacion_viento": "«Poco mar de viento en los extremos» es un resultado del modelo forzado por "
            "ERA5, no del sitio (§3.8).",
            "clase_fisica": TB.comprobacion_por_clase(serie, cf)
            | {"g2": g2, "swell_secundario": TB.swell_secundario(serie)},
            "peralte": per_res, "tp_discretos_hs_alto": TB.tp_discretos(serie, u0),
            "gev": {"mes_inicio_bloque": m0, "meses_de_maximos_e0": meses_max},
            "declustering_dudoso": "candidatos automáticos; la revisión visual de las figuras de eventos es "
            "posterior y se anota en la bóveda",
            "gate": {"G1": True, "G2": g2["cumple"], "G3": g3, "G4_marca": r1["g4_estratos_con_menos_de_30"],
                     "G5": g5, "G6_sin_ic_finito": sin_ic, "G6_ic_sin_punto": fuera_ic, "G7": g7,
                     "G7_marca": r1["alfa_fam"]["g7_marca"], "G7b": "no detiene (etapa 1)",
                     "G8": "pendiente de aceptación del usuario"},
            "segundos": round(time.time() - t0),
        }  # fmt: skip
        json_estricto.escribir(st / "resumen_f06.json", resumen)
        motivos = []
        if not g2["cumple"]:
            motivos.append(f"G2: {g2}")
        motivos += [f"G3: {p}" for p in g3]
        motivos += [f"G5: {p}" for p in g5]
        exentos = [m for m in sin_ic if m == "tau_int_h" or (m == "corte_h" and punto_corte.corte_h is None)]
        motivos += [f"G6: {m} sin valor o sin IC finito" for m in sin_ic if m not in exentos]
        for m in exentos:
            aviso(f"MARCA G6: {m} sin valor o sin IC finito (exento)")
        for m in fuera_ic:
            aviso(f"MARCA G6: el IC de {m} no contiene su punto")
        if len(top_viento):
            aviso(f"AVISO §3.8: {len(top_viento)} de los 20 mayores son mar de viento o mixto")
        criterios = {"G2": g2["cumple"], "G3": len(g3), "G5": g5, "G6_marcas": exentos + fuera_ic,
                     "G7": g7["comprobado_antes_de_leer_excesos"]}  # fmt: skip
        ins_hash = rutas_ins | {"estratos_etapa1": E1.ruta_salida(raiz, nombre) / "estratos_etapa1.json",
                                "etapa0": E0.ruta_salida(raiz) / "etapa0.json"}  # fmt: skip
        publicada = etapa.publicar(st, salida, etapa.Gate(not motivos, criterios, motivos), raices, ins_hash, raiz,
                                   semillas.semilla("f06", "clima", "dependencia"))  # fmt: skip
        reanudable.limpiar(avance)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clima", description="Etapa principal de la F6: clima de oleaje")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--previo", action="store_true", help="Verifica la etapa 1 y cronometra; no lee excesos.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = clima(a.datos, a.procesos, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    if salida:
        print(f"Publicado en {salida}")
    return 0
