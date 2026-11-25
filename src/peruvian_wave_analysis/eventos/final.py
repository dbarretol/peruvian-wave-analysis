"""Salidas finales de la F7, ``eventos-final`` (contrato de la F7, §2, §3.2 punto 4, §3.6, §4 y §7; ADR-0038).

No elige nada: reúne lo publicado por las etapas anteriores y escribe lo que la F7 entrega.

1. **``eventos.csv``:** todos los eventos identificados sobre u_id en toda la serie (1980-01 a 2026-08, §3.1), con
   el pico de E0 sin corregir y sus etiquetas (Tp, dirección, partición dominante), la asignación a cada era y si
   el evento da un pico sobre u\\* en las ramas que pasan a la F8 (§3.6, último párrafo: «para la F8 y la F10»).
2. **``picos_<rama>.csv``** de las ramas que pasan a la F8 (ADR-0038): E1 sin corregir (principal) y E0 sin
   corregir (sensibilidad). Se rehacen con las funciones de ``eventos-umbral`` y se comprueba que coinciden con los
   publicados en ``umbral/``.
3. **``resumen_f07.json``:** decisiones, estados, marcas, gates G1–G8 comprobables sobre las salidas, entregas por
   fase y rótulos. El factor de Hs de la corrección por cuantiles va con su intervalo y su fuente (F5); solo
   corrige Hs, no Tp. La nota de Garwood va en las dos eras.

**Alerta de rachas largas (§3.2, punto 4; E-15; G3):** si más del 5 % de los eventos (después de las divisiones)
duran más de 10 días, se marca y se avisa. No cambia nada.

Salida en ``processed/F07-eventos/marcona-200m/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..clima import insumos as CI
from ..comun import etapa, json_estricto, orquestador
from ..comun.errores import GateFallido
from ..homogeneidad import cuantiles as Q
from ..validacion_obs import config as C5
from ..validacion_obs import decision as D5
from . import diagnosticos_datos as DD
from . import identificacion as ID
from . import iet as IE
from . import ramas as R
from . import ramas_picos as RP
from . import sensibilidades as S
from . import umbral as U
from . import umbral_datos as UD

ETAPA = orquestador.Etapa("eventos-final", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
NOMBRE = CI.NOMBRE
ADR_RAMAS = "ADR-0038"
PRINCIPAL = "E1_sin_corregir"
SENSIBILIDAD = "E0_sin_corregir"
RAMAS_F8 = (PRINCIPAL, SENSIBILIDAD)
PAPEL = {PRINCIPAL: "principal", SENSIBILIDAD: "sensibilidad"}
DURACION_LARGA_H = 240.0  # 10 días (§3.2, punto 4)
FRACCION_LARGAS = 0.05
TOL_FACTOR = 1e-6
FORMATO = {"index": False, "float_format": "%.6g", "lineterminator": "\n", "date_format": "%Y-%m-%d %H:%M"}

NOTA_FACTOR = (
    "ADR-0038: la corrección por cuantiles no pasa como rama. Pasa como factor de Hs, aplicado al final. Solo corrige "
    "Hs; Tp no se corrige. En los contornos Hs–Tp, el modelo conjunto se ajusta con Hs sin corregir, el factor "
    "multiplica solo la coordenada Hs y Tp | Hs se evalúa en la Hs sin corregir. Vale en la cola (sobre el P99 de la "
    "F5), donde están todos los picos sobre u*. Su intervalo incluye 1: en el P99 la corrección no es significativa. "
    "La F9 decide cómo propaga el intervalo junto con el IC de muestreo."
)
NOTA_XI = (
    "ξ se deja libre y su incertidumbre pasa tal cual (IC de perfil). Fijar ξ = 0 puede ir solo como sensibilidad de "
    "la F9 (decisión del usuario, 2026-11-24)."
)
# pendientes de contrato, anotados donde se resuelven (pedido del usuario, 2026-11-24; no resueltos)
PENDIENTE_F9 = "La F9 produce el IC de ξ y lo entrega a la F11 para que lo propague a los contornos."
PENDIENTE_F10 = (
    "Con qué datos se ajusta Tp condicionado a Hs: solo los picos sobre u* o todos los estados de mar. Si se usan "
    "todos, el factor de Hs no se aplica bajo el P99."
)
PENDIENTE_F11 = "Cómo pasa a los contornos la incertidumbre de ξ que entrega la F9 (no solo el valor central)."
ROTULOS = [
    "ADR-0038 (ramas que pasan a la F8): decidido después de ver u*, los RL100, las sensibilidades y el factor.",
    "Nota de Garwood en E1: escrita después de ver los diagnósticos de §3.5.",
    "Regla de Tp (§3.7, punto 4): posterior a ver las figuras de la F6; no es criterio.",
]


NOTA_REGRESION = (
    "Prueba de regresión: sobre el P99 la corrección de la F5 es exactamente el factor, así que la desviación es 0 "
    "por construcción. No es una verificación independiente del factor (nota del usuario, 2026-11-24)."
)


def _o_none(x):
    return None if x is None else float(x)


def verificar_insumos(ins: dict[str, Path], hashes: dict[str, str], raiz: Path) -> dict:
    """G1: cada insumo, contra el manifiesto de la salida publicada que lo contiene (el más cercano hacia arriba)."""
    raiz = Path(raiz).resolve()
    ok, sin, distintos = [], [], []
    for k, p in ins.items():
        p = Path(p).resolve()
        d = next((a for a in p.parents if (a / etapa.MANIFIESTO).is_file() and raiz in a.parents), None)
        if d is None:
            sin.append(k)
            continue
        h = json_estricto.leer(d / etapa.MANIFIESTO)["salidas"].get(p.relative_to(d).as_posix())
        (ok if h == hashes[k] else distintos).append(k)
    return {
        "cumple": not sin and not distintos,
        "insumos": len(ins),
        "verificados_contra_manifiesto": len(ok),
        "sin_manifiesto": sin,
        "hash_distinto": distintos,
    }


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / NOMBRE


def factor_hs(resumen_f05: dict) -> dict:
    """Factor de la cola de la corrección por cuantiles (exp(−B_P99)) y su intervalo, de la F5."""
    v = next(x for x in resumen_f05["correccion"]["variantes"] if x["nombre"] == "cuantiles")
    lo, hi = resumen_f05["fila_fuente"]["ic95_B_q"]["P99"]
    f = float(v["factor_cola"])
    if abs(f - float(np.exp(-v["B_q"]["P99"]))) > TOL_FACTOR:
        raise ValueError("factor_cola de la F5 no es exp(−B_P99)")
    intervalo = [float(np.exp(-hi)), float(np.exp(-lo))]
    return {
        "valor": f,
        "intervalo_95": intervalo,
        "B_P99": float(v["B_q"]["P99"]),
        "ic95_B_P99": [float(lo), float(hi)],
        "significativo": not (intervalo[0] <= 1.0 <= intervalo[1]),
        "corrige": "Hs",
        "tp": "no se corrige",
        "fuente": "resumen_f05.json: correccion.variantes[cuantiles].factor_cola y fila_fuente.ic95_B_q.P99",
        "nota": NOTA_FACTOR,
    }


def comprobar_factor(picos: pd.DataFrame, f: float, u: dict[str, float]) -> dict:
    """Desviación relativa máxima de Hs_cuantiles / Hs_sin_corregir frente al factor, en los eventos con pico sobre
    u\\* en la rama sin corregir. Por debajo del P99 la corrección no es constante: allí no se comprueba."""
    out = {}
    for e in ("E0", "E1"):
        a = picos[picos["rama"] == f"{e}_sin_corregir"].set_index("id_evento")["hs"]
        b = picos[picos["rama"] == f"{e}_cuantiles"].set_index("id_evento")["hs"]
        a, b = a.align(b, join="inner")
        sobre = a > u[f"{e}_sin_corregir"]
        d = np.abs(b[sobre] / a[sobre] / f - 1.0)
        out[e] = {"picos": int(sobre.sum()), "desviacion_relativa_maxima": float(d.max()) if len(d) else None}
    return out


def cola_del_factor(t, hs: np.ndarray, b_q: dict, u: dict[str, float], f: float) -> dict:
    """Premisa del factor de Hs (ADR-0038, punto 3): u\\* está en la cola, donde la corrección de la F5 es constante.

    Repite la selección y los pesos de la F5 (``validacion_obs.etapa``, paso 6, y ``decision.corregir``): E0 corrige
    toda la serie válida; E1, el tramo asimilado (desde ``INICIO_ASIMILADO``). Con peso mensual, B queda constante
    desde el cuantil estimable más alto (``q_max``, el P99), así que el umbral de la cola es el menor Hs sin corregir
    con F_w ≥ q_max. En la rama por cuantiles ese umbral se multiplica por el factor. Se exige u\\* ≥ umbral.
    """
    t = pd.DatetimeIndex(t)
    hs = np.asarray(hs, float)
    q, _ = D5.funcion_cuantiles(b_q)
    q_max = float(np.max(q))
    validos = np.isfinite(hs)
    sel = {"E0": validos, "E1": validos & (t >= pd.Timestamp(C5.INICIO_ASIMILADO))}
    out = {"q_max": q_max, "seleccion": {"E0": "toda la serie válida", "E1": f"desde {C5.INICIO_ASIMILADO}"}}
    for e, m in sel.items():
        x = hs[m]
        h = float(Q.cuantil_ponderado(x, Q.pesos_mensuales(t[m], np.isfinite(x)), q_max)[0])
        for v, escala in (("sin_corregir", 1.0), ("cuantiles", f)):
            rama = f"{e}_{v}"
            out[rama] = {"umbral_cola_m": h * escala, "u_m": float(u[rama]), "cumple": bool(u[rama] >= h * escala)}
    return out


def tabla_eventos(eventos: pd.DataFrame, serie: pd.DataFrame, inicio_e1, picos_f8: dict[str, pd.DataFrame],
                  fin=R.FIN_MODELO) -> pd.DataFrame:  # fmt: skip
    """``eventos.csv``: eventos de toda la serie con el pico y las etiquetas de E0 sin corregir."""
    p = R.picos_de_rama(eventos, serie["VHM0"].astype(float), serie, tol_empate=None)
    if (p["t_pico"].to_numpy() != pd.DatetimeIndex(eventos["t_pico"]).to_numpy()).any():
        raise RuntimeError("El pico de E0 sin corregir no coincide con el de la identificación")
    t = pd.DatetimeIndex(eventos["t_pico"])
    out = eventos[["id_evento", "id_racha", "t_inicio", "t_fin", "t_pico", "n_excedencias", "partes_racha",
                   "dividido_por_bajada", "toca_hueco", "duracion_h"]].copy()  # fmt: skip
    out["hs_pico"] = p["hs"].to_numpy()
    for c in ("tp", "vmdr", "vped", "particion_dominante"):
        out[c] = p[c].to_numpy()
    out["en_periodo_modelo"] = t <= pd.Timestamp(fin)
    out["en_E0"] = out["en_periodo_modelo"]
    out["en_E1"] = out["en_periodo_modelo"] & (t >= pd.Timestamp(inicio_e1))
    for rama, pr in picos_f8.items():
        out[f"sobre_u_{rama}"] = out["id_evento"].isin(pr["id_evento"])
    out["duracion_larga"] = out["duracion_h"] > DURACION_LARGA_H
    return out


def alerta_rachas_largas(ev: pd.DataFrame) -> dict:
    out = {"duracion_h": DURACION_LARGA_H, "fraccion_maxima": FRACCION_LARGAS}
    for nombre, sub in (("toda_la_serie", ev), ("E0", ev[ev["en_E0"]]), ("E1", ev[ev["en_E1"]])):
        n = int(sub["duracion_larga"].sum())
        out[nombre] = {"eventos": int(len(sub)), "largos": n, "fraccion": n / len(sub) if len(sub) else None}
    out["marca"] = any(
        out[k]["fraccion"] is not None and out[k]["fraccion"] > FRACCION_LARGAS for k in ("toda_la_serie", "E0", "E1")
    )
    return out


def anidados(barrido: pd.DataFrame) -> dict[str, bool]:
    """G3: en cada rama, el número de picos no crece con u."""
    return {r: bool(np.all(np.diff(g.sort_values("u_m")["n"].to_numpy()) <= 0)) for r, g in barrido.groupby("rama")}


def comprobar_eventos(ev: pd.DataFrame, hs: pd.Series, u_id: float, iet: float) -> dict:
    """G2 sobre la salida: eventos sin solaparse, pico dentro de su evento y sobre u_id, picos consecutivos a ≥ IET\\*.
    ``identificar`` ya exige la rejilla completa (``comprobar_rejilla``): si no la hay, no se llega aquí."""
    ini, fin, pico = (pd.DatetimeIndex(ev[c]) for c in ("t_inicio", "t_fin", "t_pico"))
    sep = (pico[1:] - pico[:-1]).total_seconds().to_numpy() / 3600.0 if len(ev) > 1 else np.array([np.inf])
    hs_pico = hs.reindex(pico).to_numpy(float)
    out = {
        "rejilla_completa": True,
        "eventos": int(len(ev)),
        "sin_solape": bool(len(ev) < 2 or (ini[1:] > fin[:-1]).all()),
        "pico_dentro_del_evento": bool(((pico >= ini) & (pico <= fin)).all()),
        "picos_sobre_u_id": bool(np.all(hs_pico > u_id)),
        "separacion_minima_entre_picos_h": float(sep.min()),
        "picos_a_iet_o_mas": bool(sep.min() >= iet),
    }
    out["cumple"] = bool(len(ev) > 0) and all(v for v in out.values() if isinstance(v, bool))
    return out


def evaluar_gates(g1: dict, g2: dict, g3: dict, largas: dict, umb: dict, n_picos: dict[str, int], j_iet: dict,
                  bajo: list, cola: dict) -> tuple[dict, list[str]]:  # fmt: skip
    """Gates G1–G10 (contrato §7), la premisa del factor de Hs y los motivos de fallo. G9 y G10 se comprueban fuera
    de la etapa: quedan «pendiente», con su referencia."""
    sens = bool(j_iet["iet_sensible"])
    eras_largas = ("toda_la_serie", "E0", "E1")
    premisa = [v["cumple"] for v in cola.values() if isinstance(v, dict) and "cumple" in v]
    gates = {
        "G1": g1,
        "G2": g2 | {"pruebas": "tests/test_eventos_identificacion.py (huecos, rachas y bajada; metodologia.md §3.4)"},
        "G3": {
            "cumple": all(g3.values()) and all(largas[k]["fraccion"] is not None for k in eras_largas),
            "anidados_por_rama": g3,
            "alerta_rachas_largas_calculada_en": list(eras_largas),
            "marca_rachas_largas": bool(largas["marca"]),
        },
        "G4": {"cumple": all(bool(umb["ramas"][r]["g4"]) for r in R.RAMAS), "como": "de umbral.json (eventos-umbral)"},
        "G5": {
            "cumple": all(n >= U.N_MIN_ARRIBA for n in n_picos.values())
            and all(n == int(umb["ramas"][r][UD.REGLA]["n"]) for r, n in n_picos.items()),
            "picos_en_u": n_picos,
            "minimo": U.N_MIN_ARRIBA,
            "como": "picos rehechos sobre u*, contados aquí y comparados con el n de umbral.json",
        },
        "G6": {"cumple": all(umb["ramas"][r][UD.REGLA]["estado"] == "con_resultado" for r in R.RAMAS)},
        # contrato §7: «IET* cumple §3.3, o lleva la marca "IET sensible" con el aviso al usuario» (F07-R-01)
        "G7": {"cumple": True, "iet_sensible": sens, "aviso": "IET sensible: se avisa al usuario" if sens else None},
        "G8": {"cumple": not bajo and not umb["g8"]["picos_bajo_p95_f3"], "picos_bajo_p95_f3": bajo},
        "G9": {
            "cumple": "pendiente",
            "como": "pruebas de 0 frente a 2 procesos con comparación exacta "
            "(test_determinismo_con_uno_y_dos_procesos, "
            "test_simulacion_igual_con_0_y_2_procesos, test_piloto_r3_igual_con_0_y_2_procesos, "
            "test_ruido_iet_igual_con_0_y_2_procesos); resultado en metodologia.md §4",
        },
        "G10": {
            "cumple": "pendiente",
            "como": "revisión de código de cierre (revision.md §8) y aceptación del usuario",
        },
        "premisa_factor_hs": {
            "cumple": bool(premisa) and all(premisa),
            "como": "u* ≥ umbral de la cola de la F5 en cada rama (ADR-0038, punto 3; decisiones.factor_hs_cuantiles)",
        },
    }
    motivos = [f"{g} no se cumple" for g, v in gates.items() if v["cumple"] is False]
    return gates, motivos


def resumen_rama(rama: str, umb: dict, conteo: pd.Series, diag: pd.Series, sens: pd.DataFrame, n_dud: int,
                 verano: int, nota_garwood: str) -> dict:  # fmt: skip
    u = umb["ramas"][rama]
    r1b = u[UD.REGLA]
    d = sens[sens["rama"] == rama].set_index("sensibilidad")["d_rl100"]
    return {
        "papel": PAPEL.get(rama, "trazabilidad"),
        "pasa_a_la_f8": rama in RAMAS_F8,
        "periodo": [str(pd.Timestamp(conteo["inicio"]).date()), str(pd.Timestamp(R.FIN_MODELO).date())],
        "anios": float(u["anios"]),
        "eventos": int(conteo["eventos"]),
        "N0": int(u["N0"]),
        "regla": UD.REGLA,
        "estado": r1b["estado"],
        "u_m": float(r1b["u_m"]),
        "n_sobre_u": int(r1b["n"]),
        "lambda_sobre_u_por_anio": int(r1b["n"]) / float(u["anios"]),
        "xi": r1b["xi"],
        "xi_ic95_perfil": [r1b["xi_lo"], r1b["xi_hi"]],
        "rl100_m": r1b["rl100"],
        "rl100_ic95_perfil_m": [r1b["rl100_lo"], r1b["rl100_hi"]],
        "tasa_sobre_u_id": {
            "lambda_por_anio": float(conteo["lambda_por_anio"]),
            "ic_garwood_95": [float(conteo["ic_garwood_95_inf"]), float(conteo["ic_garwood_95_sup"])],
            "nota_garwood": nota_garwood,
        },  # fmt: skip
        "dispersion_conteos_anuales": {"indice": float(diag["dispersion_indice"]), "p": float(diag["dispersion_p"])},
        "marcas": {
            "punto_sensible": bool(u["relevancia"]["punto_sensible"]),
            "deriva_xi": bool(u["deriva_xi"]["marca"]),
            "nota_deriva_xi": u["deriva_xi"].get("nota"),
        },  # fmt: skip
        "sensibilidad_r2_r3": u["sensibilidad"],
        "sensibilidades_37_d_rl100": {k: float(v) for k, v in d.items() if k != "principal"},
        "picos_dudoso_f3_sobre_u": n_dud,
        "picos_nov_abr_sobre_u": verano,
    }


def ejecutar(raiz_datos=None) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = RP.rutas_insumos(raiz)
    dirs = {"iet": IE.ruta_salida(raiz), "umbral": UD.ruta_salida(raiz), "diagnosticos": DD.ruta_salida(raiz),
            "sensibilidades": S.ruta_salida(raiz), "ramas_picos": RP.ruta_salida(raiz)}  # fmt: skip
    manifiestos = {k: etapa.exigir_publicada(d) for k, d in dirs.items()}
    ins |= {"iet": dirs["iet"] / "iet.json", "umbral": dirs["umbral"] / "umbral.json",
            "barrido_umbral": dirs["umbral"] / "barrido_umbral.csv",
            "diagnosticos": dirs["diagnosticos"] / "diagnosticos.csv",
            "diagnosticos_json": dirs["diagnosticos"] / "diagnosticos.json",
            "sensibilidades": dirs["sensibilidades"] / "sensibilidades.csv",
            "sensibilidades_json": dirs["sensibilidades"] / "sensibilidades.json",
            "conteos": dirs["ramas_picos"] / "conteos.csv"}  # fmt: skip
    ins |= {f"picos_{r}": dirs["umbral"] / f"picos_{r}.csv" for r in R.RAMAS}
    hashes = etapa.exigir_insumos(ins)
    g1 = verificar_insumos(ins, hashes, raiz) | {"salidas_previas_verificadas": sorted(manifiestos)}
    datos = CI.cargar(raiz, ins, NOMBRE)
    p95 = float(json_estricto.leer(ins["resumen_candidatos"])["p95_hs_m"])
    j_iet = json_estricto.leer(ins["iet"])
    iet = float(j_iet["par"]["iet_estrella_h"])
    umb = json_estricto.leer(ins["umbral"])
    if float(umb["iet_estrella_h"]) != iet:
        raise RuntimeError("umbral.json y iet.json no usan el mismo IET*")
    dj = json_estricto.leer(ins["diagnosticos_json"])
    sj = json_estricto.leer(ins["sensibilidades_json"])
    r5 = json_estricto.leer(ins["resumen_f05"])
    cand, rev = pd.read_csv(ins["candidatos"]), pd.read_csv(ins["declustering_dudoso"])
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        conteos, picos, _, bajo = RP.ramas_picos(datos, p95, p95, cand, rev, iets=(iet,))
        # 1. picos de las ramas que pasan a la F8, comprobados contra los publicados
        picos_f8, motivos = {}, []
        for rama in R.RAMAS:
            u = float(umb["ramas"][rama][UD.REGLA]["u_m"])
            t = UD.tabla_picos(picos[picos["rama"] == rama], u)
            pub = pd.read_csv(ins[f"picos_{rama}"])
            if list(pub["id_evento"]) != list(t["id_evento"]) or not np.allclose(pub["hs"], t["hs"], rtol=1e-5):
                motivos.append(f"Los picos de {rama} no coinciden con umbral/picos_{rama}.csv")
            if rama in RAMAS_F8:
                picos_f8[rama] = t
                t.to_csv(st / f"picos_{rama}.csv", **FORMATO)
        # 2. eventos de toda la serie
        ev = ID.identificar(datos.serie["VHM0"].astype(float), p95, iet)
        tabla = tabla_eventos(ev, datos.serie, datos.inicio_e1, picos_f8)
        tabla.to_csv(st / "eventos.csv", **FORMATO)
        largas = alerta_rachas_largas(tabla)
        # 3. resumen
        fac = factor_hs(r5)
        u_estrella = {r: float(umb["ramas"][r][UD.REGLA]["u_m"]) for r in R.RAMAS}
        comp = comprobar_factor(picos, fac["valor"], u_estrella)
        fac["comprobacion_en_los_picos_sobre_u"] = comp | {"tolerancia": TOL_FACTOR, "tipo": NOTA_REGRESION}
        desv = [v["desviacion_relativa_maxima"] for v in comp.values()]
        if None in desv:
            motivos.append("No hay picos sobre u* para comprobar el factor")
        elif max(desv) > TOL_FACTOR:
            motivos.append(f"Los picos por cuantiles no son los sin corregir × {fac['valor']:.4f} ({max(desv):.2e})")
        sn = pd.read_parquet(ins["serie_nodo"], columns=["VHM0", "valido"])
        b_q = next(x for x in r5["correccion"]["variantes"] if x["nombre"] == "cuantiles")["B_q"]
        cola = cola_del_factor(sn.index, sn["VHM0"].where(sn["valido"]).to_numpy(float), b_q, u_estrella, fac["valor"])
        fac["premisa_u_en_la_cola"] = cola
        barrido = pd.read_csv(ins["barrido_umbral"])
        diag = pd.read_csv(ins["diagnosticos"])
        diag = diag[diag["iet_h"] == iet].set_index("rama")
        sens = pd.read_csv(ins["sensibilidades"])
        cont = conteos[conteos["iet_h"] == iet].set_index("rama")
        nota_g = dj["notas"]["garwood"]
        inicio_serie = pd.Timestamp(datos.serie.index[0])
        ramas = {}
        for r in R.RAMAS:
            c = cont.loc[r].copy()
            c["inicio"] = R.inicio_rama(r, inicio_serie, datos.inicio_e1)
            ramas[r] = resumen_rama(r, umb, c, diag.loc[r], sens, int(sj["dudosos_f3"][r]),
                                    int(dj["picos_de_verano"]["picos_nov_abr_sobre_u"][r]), nota_g)  # fmt: skip
        g2 = comprobar_eventos(ev, datos.serie["VHM0"].astype(float), p95, iet)
        n_picos = {r: int((picos[picos["rama"] == r]["hs"] > u_estrella[r]).sum()) for r in R.RAMAS}
        gates, m = evaluar_gates(g1, g2, anidados(barrido), largas, umb, n_picos, j_iet, bajo, cola)
        motivos += m
        resumen = {
            "fase": "F7",
            "nodo": NOMBRE,
            "decisiones": {
                "u_id_m": p95,
                "iet_estrella_h": iet,
                "iet_siguiente_h": _o_none(j_iet["par"].get("siguiente_h")),
                "regla_umbral": f"{UD.REGLA} (ADR-0037)",
                "ramas_a_la_f8": {
                    "adr": ADR_RAMAS,
                    "principal": PRINCIPAL,
                    "sensibilidad": SENSIBILIDAD,
                    "no_pasan": [r for r in R.RAMAS if r not in RAMAS_F8],
                    "e2": "E2 sigue siendo E0 con el escalón de 2016-03 como covariable; toma el u* de E0 sin "
                    "corregir (P5). Como E0, es sensibilidad.",
                    "nota_e1": "E1 excluye el escalón de 1993-02, pero contiene el de 2016-03 (C2 de la F4).",
                },  # fmt: skip
                "factor_hs_cuantiles": fac,
                "xi": NOTA_XI,
            },
            "ramas": ramas,
            "independencia": {
                "dependencia_no_resuelta_por_el_iet": {
                    r: bool(v["dependencia_no_resuelta_por_el_iet"])
                    for r, v in dj["marcas"].items()
                    if isinstance(v, dict)
                },
                "nota_garwood": nota_g,
            },  # fmt: skip
            "alerta_rachas_largas": largas,
            "eventos": {
                "toda_la_serie": int(len(tabla)),
                "periodo_modelo_E0": int(tabla["en_E0"].sum()),
                "periodo_modelo_E1": int(tabla["en_E1"].sum()),
                "con_2026": int((~tabla["en_periodo_modelo"]).sum()),
            },
            "sensibilidades_37": {"nota": sj["nota"], "regla_tp": sj["regla_tp"]},
            "gates": gates,
            "entregas": {
                "F8": {
                    "recibe": [
                        "eventos.csv",
                        "IET*",
                        "u* por rama",
                        "picos_E1_sin_corregir.csv (principal)",
                        "picos_E0_sin_corregir.csv (sensibilidad)",
                        "dispersión de los conteos anuales",
                        "factor de Hs con su intervalo (solo Hs)",
                    ],
                    "condicion_e2": "Un escalón en la GPD va también en λ (E-07).",
                },
                "F9": {
                    "recibe": [
                        "u*, IET* y su par (sensibilidad obligatoria)",
                        "tablas de picos",
                        "regla aplicada y estados",
                        "sensibilidades de §3.7",
                        "banderas dudoso_f3",
                        "marcas de deriva de ξ y de punto sensible",
                        "dependencia entre años",
                        "intervalo del factor de Hs",
                    ],
                    "xi": NOTA_XI,
                    "pendiente_del_contrato_de_la_f9": PENDIENTE_F9,
                },
                "F10": {
                    "recibe": [
                        "eventos.csv y picos con Tp, dirección y partición dominante",
                        "factor de Hs con su intervalo: solo la coordenada Hs; Tp | Hs en la Hs sin corregir",
                    ],
                    "pendiente_del_contrato_de_la_f10": PENDIENTE_F10,
                },
                "F11": {
                    "recibe": ["factor de Hs con su intervalo: solo la coordenada Hs del contorno"],
                    "pendiente_del_contrato_de_la_f11": PENDIENTE_F11,
                },
                "F12": {
                    "recibe": [
                        "regla aplicada, estado y argumento de elección (ADR-0025)",
                        "limitación de la resolución de 3 h",
                    ]
                },
            },  # fmt: skip
            "rotulos": ROTULOS,
        }
        json_estricto.escribir(st / "resumen_f07.json", resumen)
        if largas["marca"]:
            print("AVISO: rachas largas (§3.2, punto 4)", flush=True)
        if gates["G7"]["iet_sensible"]:
            print("AVISO: IET sensible (§3.3; G7)", flush=True)
        gate = etapa.Gate(not motivos, {"iet_estrella_h": iet, "ramas_f8": list(RAMAS_F8)}, motivos)
        return etapa.publicar(st, salida, gate, raices, ins, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-final", description="Salidas finales de la F7 (§2, §4; ADR-0038)")
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
