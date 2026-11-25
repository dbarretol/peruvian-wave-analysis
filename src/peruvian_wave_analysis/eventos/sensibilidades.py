"""Sensibilidades de §3.7, ``eventos-sensibilidades`` (contrato de la F7, §3.7; ADR-0026 §5). **Descriptivas: no
mueven u\\* ni el IET\\*.** Solo las seis que fija el contrato; lo raro que salga va como nota.

**Lectura común (precisión n.º 28, aceptada por el usuario el 2026-11-24):** en cada sensibilidad, u\\* e IET\\* de cada rama quedan fijos (los
publicados por ``eventos-umbral`` y ``eventos-iet``). Se rehacen los eventos, los picos de cada rama y el ajuste de la
GPD en u\\* (``iet.ajuste_descriptivo``: ξ̂ y RL100 con su IC de perfil). No se vuelve a correr R1b. En las ramas por
cuantiles, u_id llevado y el borde de E-29 son los del período principal.

1. **IET del par:** el siguiente de la rejilla (``ramas_picos.ramas_picos``, el mismo código de la etapa publicada).
2. **Sin división por bajada intermedia** (``identificacion.identificar(bajada=None)``).
3. **Con enero a agosto de 2026:** los mismos eventos, asignados hasta el último estado de la serie; exposición en años
   decimales con la misma fórmula (P1).
4. **Regla de Tp (E-06, E-28), rotulada posterior a ver las figuras de la F6.** Sobre cada evento de §3.2 (después de la
   bajada), **el mismo procedimiento que la bajada** (``identificacion.dividir(orden="mas_honda")``, con la
   comprobación frente a las vecinas); solo cambia la condición de validez del corte. Un corte en m con crestas tl
   (máximo de Hs a la izquierda, primera aparición) y tr (a la derecha) vale si (umbrales del contrato, §3.7 punto 4):
   - tr − tl ≥ 24 h; Hs(m) es el mínimo entre las crestas; **cada** cresta supera Hs(m) en ≥ 0.3 m (la prominencia,
     30 veces la discretización de 0.01 m: una oscilación de Hs no parte);
   - VTPK(tr) − VTPK(m) ≥ 3 s;
   - en tr domina una partición de swell (SW1 o SW2, ``ramas.particion_dominante``), y su VTM01 sube ≥ 2 s de m a tr.

   Un NaN en cualquiera de esas variables hace fallar el corte. Se publican los eventos partidos y el RL100 en u\\*.
5. **Picos con ``dudoso_f3``** sobre u\\* de cada rama y su rango en la cola (1 = el mayor).
6. **Intensidad de la prueba de Poisson con 2 y 4 estaciones:** ya calculada en ``eventos-diagnosticos``; aquí se
   copia en (u\\*, IET\\*) de cada rama, con su tamaño.

Salida en ``processed/F07-eventos/sensibilidades/marcona-200m/``: ``sensibilidades.csv`` (1 a 4),
``dudosos_f3.csv`` (5) y ``sensibilidades.json`` (todo, con el 6).
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
from . import diagnosticos_datos as DD
from . import identificacion as ID
from . import iet as IE
from . import ramas as R
from . import ramas_picos as RP
from . import umbral_datos as UD

ETAPA = orquestador.Etapa("eventos-sensibilidades", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
NOMBRE = CI.NOMBRE
PASO_H = ID.PASO_H
SEP_TP_H = 24.0
BAJADA_TP_M = 0.3
SUBIDA_TP_S = 3.0
SUBIDA_TM01_S = 2.0
SWELL = ("SW1", "SW2")
SENSIBILIDADES = ("principal", "iet_siguiente", "sin_bajada", "con_2026", "regla_tp")
NOTA = (
    "Descriptivas (§3.7; ADR-0026 §5): u* e IET* fijos, no se vuelve a correr R1b (precisión n.º 28). No cambian "
    "ninguna decisión de la F7. Pasan a la F9."
)
NOTA_TP = "Regla de Tp (E-06, E-28): rotulada posterior a ver las figuras de la F6; no es criterio."


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "sensibilidades" / NOMBRE


# --- Regla de Tp -----------------------------------------------------------------------------------


def cumple_tp(hs, tp, tm01, dominante, cresta1: int, m: int, cresta2: int, bajada: float) -> bool:
    """Las condiciones de §3.7, punto 4, en posiciones absolutas: crestas a ≥ 24 h; ``bajada`` ≥ 0.3 m sobre Hs(m);
    VTPK sube ≥ 3 s de m a la segunda cresta; en ella domina un swell cuyo VTM01 sube ≥ 2 s. Un NaN no cumple."""
    if cresta2 - cresta1 < SEP_TP_H / PASO_H - ID.TOL or not bajada >= BAJADA_TP_M - ID.TOL:
        return False
    if not tp[cresta2] - tp[m] >= SUBIDA_TP_S - ID.TOL:
        return False
    p = dominante[cresta2]
    return p in SWELL and bool(tm01[p][cresta2] - tm01[p][m] >= SUBIDA_TM01_S - ID.TOL)


def corte_tp(hs, tp, tm01, dominante, s: int, e: int, izquierda: int | None = None,
             derecha: int | None = None) -> tuple[int, float] | None:  # fmt: skip
    """(posición relativa a ``s``, profundidad) del corte válido más hondo de la regla de Tp en la parte [s, e], o None.

    Mismo procedimiento que la bajada (``identificacion.corte``) con otra condición de validez: crestas = máximo de Hs
    a cada lado (primera aparición), m = el mínimo entre ellas. ``izquierda``/``derecha``: posición absoluta de la
    cresta de la parte vecina ya cortada (el corte compartido es s − 1 o e). **Comprobación frente a las vecinas:** la
    subparte junto a una vecina cumple las condiciones frente a ella (su cresta sobre el corte compartido, como en la
    bajada). El corte deja [s, s + m] y [s + m + 1, e].
    """
    x = np.asarray(hs[s : e + 1], dtype=float)
    if x.size < 3:
        return None
    xm = np.where(np.isnan(x), -np.inf, x)
    izq, der = ID._arg_max_prefijo(xm), ID._arg_max_sufijo(xm)
    mejor, prof_mejor = None, -np.inf
    for m in range(1, x.size - 1):
        if np.isnan(x[m]):
            continue
        tl, tr = izq[m], der[m + 1]
        if x[m] > np.nanmin(x[tl : tr + 1]):
            continue
        prof = min(x[tl], x[tr]) - x[m]
        if not cumple_tp(hs, tp, tm01, dominante, s + tl, s + m, s + tr, prof):
            continue
        if izquierda is not None and not cumple_tp(hs, tp, tm01, dominante, izquierda, s - 1, s + tl,
                                                   x[tl] - hs[s - 1]):  # fmt: skip
            continue
        if derecha is not None and not cumple_tp(hs, tp, tm01, dominante, s + tr, e, derecha, x[tr] - hs[e]):
            continue
        if prof > prof_mejor + ID.TOL:
            mejor, prof_mejor = m, float(prof)
    return None if mejor is None else (mejor, prof_mejor)


def dividir_tp(hs, tp, tm01, dominante, a: int, b: int) -> list[tuple[int, int]]:
    """Partes de [a, b] con la regla de Tp y el procedimiento de ``identificacion.dividir(orden="mas_honda")``: en cada
    paso, el corte válido más hondo entre todas las partes (empate: el de más a la izquierda), con la comprobación
    frente a las vecinas; se repite hasta que ninguno cumpla."""
    partes = [(a, b)]
    while True:
        elegido = None
        for i, (s, e) in enumerate(partes):
            izq = ID._pico(hs, *partes[i - 1]) if i > 0 else None
            der = ID._pico(hs, *partes[i + 1]) if i + 1 < len(partes) else None
            c = corte_tp(hs, tp, tm01, dominante, s, e, izq, der)
            if c is not None and (elegido is None or c[1] > elegido[2] + ID.TOL):
                elegido = (i, c[0], c[1])
        if elegido is None:
            return partes
        i, m, _ = elegido
        s, e = partes[i]
        partes[i : i + 1] = [(s, s + m), (s + m + 1, e)]


def aplicar_regla_tp(eventos: pd.DataFrame, serie: pd.DataFrame, u_id: float) -> pd.DataFrame:
    """Los eventos de §3.2 con la regla de Tp. Conserva ``id_racha`` y ``dividido_por_bajada``; ``partido_por_tp``."""
    hs = serie["VHM0"].to_numpy(dtype=float)
    tp = serie["VTPK"].to_numpy(dtype=float)
    tm01 = {p: serie[f"VTM01_{p}"].to_numpy(dtype=float) for p in SWELL}
    dom = R.particion_dominante(serie)
    filas = []
    for ev in eventos.itertuples(index=False):
        partes = dividir_tp(hs, tp, tm01, dom, int(ev.i_inicio), int(ev.i_fin))
        for s, e in partes:
            seg = np.where(np.isnan(hs[s : e + 1]), -np.inf, hs[s : e + 1])
            filas.append({"id_evento_original": int(ev.id_evento), "id_racha": int(ev.id_racha), "i_inicio": s,
                          "i_fin": e, "i_pico": s + int(np.argmax(seg)),
                          "n_excedencias": int(np.sum(seg > u_id)), "partes_racha": int(ev.partes_racha),
                          "dividido_por_bajada": bool(ev.dividido_por_bajada), "toca_hueco": bool(ev.toca_hueco),
                          "duracion_h": (e - s + 1) * PASO_H, "partido_por_tp": len(partes) > 1})  # fmt: skip
    df = pd.DataFrame(filas)
    df.insert(0, "id_evento", np.arange(len(df)))
    idx = serie.index
    df["t_inicio"], df["t_fin"], df["t_pico"] = idx[df["i_inicio"]], idx[df["i_fin"]], idx[df["i_pico"]]
    return df


# --- Picos de cada rama sobre un conjunto de eventos -----------------------------------------------


def picos_por_rama(datos: CI.Datos, eventos: pd.DataFrame, variantes: dict, candidatos, revision, p95_f3: float,
                   fin=R.FIN_MODELO) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:  # fmt: skip
    """Picos de cada rama (con el borde de E-29 de ``variantes``) y exposición, como ``ramas_picos.ramas_picos``."""
    serie = datos.serie
    inicio_serie = pd.Timestamp(serie.index[0])
    picos, info = {}, {}
    for rama in R.RAMAS:
        ini = R.inicio_rama(rama, inicio_serie, datos.inicio_e1)
        ev_r = R.eventos_de_rama(eventos, ini, fin)
        p = R.picos_de_rama(ev_r, datos.ramas[rama], serie)
        if R.es_variante(rama):
            p, _ = R.borde_inferior(p, variantes[rama])
        g8 = R.proteccion_cola(p, p95_f3)
        p = p.assign(dudoso_f3=R.marcar_dudosos_f3(p, candidatos), revision_visual_f6=R.marcar_revision_f6(p, revision))
        expo = R.exposicion(datos.ramas[rama], ini, fin=fin, n_picos=int(len(p)), rama=rama)
        picos[rama] = p
        info[rama] = {"eventos": int(len(ev_r)), "picos": int(len(p)), "anios": float(expo["anios"]),
                      "picos_bajo_p95_f3": int(g8["picos_bajo_p95"])}  # fmt: skip
    return picos, info


def fila(sens: str, rama: str, picos: pd.DataFrame, info: dict, u: float, rl_principal: float | None) -> dict:
    """Una fila de ``sensibilidades.csv``: ajuste en u* fijo, con IC de perfil."""
    hs = picos["hs"].to_numpy(dtype=float) if len(picos) else np.array([], dtype=float)
    f = {"sensibilidad": sens, "rama": rama} | info | IE.ajuste_descriptivo(hs, u, info["anios"])
    rl = f["rl100"]
    f["d_rl100"] = rl / rl_principal - 1.0 if (rl_principal and np.isfinite(rl)) else np.nan
    return f


def dudosos(picos: pd.DataFrame, rama: str, u: float) -> pd.DataFrame:
    """§3.7, punto 5: picos con ``dudoso_f3`` sobre u* y su rango en la cola (1 = el mayor)."""
    sobre = picos[picos["hs"] > u].sort_values("hs", ascending=False, kind="stable").reset_index(drop=True)
    sobre["rango"] = np.arange(1, len(sobre) + 1)
    d = sobre[sobre["dudoso_f3"].astype(bool)]
    return pd.DataFrame({"rama": rama, "t_pico": d["t_pico"], "hs": d["hs"], "rango": d["rango"],
                         "picos_sobre_u": len(sobre)})  # fmt: skip


def calcular(datos: CI.Datos, u_id: float, p95_f3: float, u_estrella: dict, iet: float, iet_siguiente: float,
             candidatos, revision) -> tuple[pd.DataFrame, pd.DataFrame, dict]:  # fmt: skip
    """(tabla de §3.7, puntos 1 a 4; dudosos del punto 5; conteos de la regla de Tp)."""
    _, pic, variantes, _ = RP.ramas_picos(datos, u_id, p95_f3, candidatos, revision, iets=(iet, iet_siguiente))
    hs0 = datos.serie["VHM0"].astype(float)
    ev = ID.identificar(hs0, u_id, iet)
    ev_tp = aplicar_regla_tp(ev, datos.serie, u_id)
    pr = lambda e, fin=R.FIN_MODELO: picos_por_rama(datos, e, variantes, candidatos, revision, p95_f3, fin)  # noqa: E731
    conjuntos = {
        "principal": pr(ev),
        "iet_siguiente": pr(ID.identificar(hs0, u_id, iet_siguiente)),
        "sin_bajada": pr(ID.identificar(hs0, u_id, iet, bajada=None)),
        "con_2026": pr(ev, pd.Timestamp(datos.serie.index[-1])),
        "regla_tp": pr(ev_tp),
    }
    # comprobación: con IET* y su par, los picos son los de la función de la etapa publicada
    for sens, i in (("principal", iet), ("iet_siguiente", iet_siguiente)):
        for r in R.RAMAS:
            a = pic[(pic["iet_h"] == i) & (pic["rama"] == r)]["hs"].to_numpy()
            if not np.array_equal(a, conjuntos[sens][0][r]["hs"].to_numpy()):
                raise RuntimeError(f"Los picos de {sens} en {r} no coinciden con ramas_picos")
    filas, rl0 = [], {}
    inicio = pd.Timestamp(hs0.index[0])
    for sens in SENSIBILIDADES:
        picos, info = conjuntos[sens]
        for r in R.RAMAS:
            f = fila(sens, r, picos[r], info[r], u_estrella[r], rl0.get(r))
            if sens == "principal":
                rl0[r] = f["rl100"]
                f["d_rl100"] = 0.0
            if sens == "regla_tp":
                e = R.eventos_de_rama(ev_tp, R.inicio_rama(r, inicio, datos.inicio_e1))
                f["eventos_partidos_por_tp"] = int(e.loc[e["partido_por_tp"], "id_evento_original"].nunique())
            filas.append(f)
    tabla = pd.DataFrame(filas)
    dud = pd.concat([dudosos(conjuntos["principal"][0][r], r, u_estrella[r]) for r in R.RAMAS], ignore_index=True)
    partidos = int(ev_tp.loc[ev_tp["partido_por_tp"], "id_evento_original"].nunique())
    tp = {"eventos_antes": int(len(ev)), "eventos_despues": int(len(ev_tp)), "eventos_partidos": partidos}
    return tabla, dud, tp


def intensidad_estaciones(diag: pd.DataFrame, tam: pd.DataFrame, iet: float, u_estrella: dict) -> dict:
    """§3.7, punto 6, copiado de ``eventos-diagnosticos`` en (u*, IET*) de cada rama."""
    t = diag[diag["iet_h"] == iet].set_index("rama")
    out = {"fuente": "diagnosticos/marcona-200m/ (diagnosticos.csv y tamano_poisson.csv)", "por_rama": {}}
    for r in R.RAMAS:
        out["por_rama"][r] = {"u_m": u_estrella[r]} | {
            f"{m}_p_ad_texto": str(t.loc[r, f"poisson_{m}_p_ad_texto"]) for m in DD.MODELOS
        }  # fmt: skip
    out["tamano"] = tam.to_dict(orient="records")
    return out


def ejecutar(raiz_datos=None) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = RP.rutas_insumos(raiz)
    d_iet, d_umb, d_dia = IE.ruta_salida(raiz), UD.ruta_salida(raiz), DD.ruta_salida(raiz)
    for d in (d_iet, d_umb, d_dia):
        etapa.exigir_publicada(d)
    ins |= {"iet": d_iet / "iet.json", "umbral": d_umb / "umbral.json", "diagnosticos": d_dia / "diagnosticos.csv",
            "tamano_poisson": d_dia / "tamano_poisson.csv"}  # fmt: skip
    etapa.exigir_insumos(ins)
    datos = CI.cargar(raiz, ins, NOMBRE)
    p95 = float(json_estricto.leer(ins["resumen_candidatos"])["p95_hs_m"])
    j_iet = json_estricto.leer(ins["iet"])
    iet, iet_sig = float(j_iet["par"]["iet_estrella_h"]), float(j_iet["par"]["siguiente_h"])
    umb = json_estricto.leer(ins["umbral"])
    u_estrella = {r: float(umb["ramas"][r][UD.REGLA]["u_m"]) for r in R.RAMAS}
    cand, rev = pd.read_csv(ins["candidatos"]), pd.read_csv(ins["declustering_dudoso"])
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        tabla, dud, tp = calcular(datos, p95, p95, u_estrella, iet, iet_sig, cand, rev)
        op = {"index": False, "float_format": "%.6g", "lineterminator": "\n"}
        tabla.to_csv(st / "sensibilidades.csv", **op)
        dud.to_csv(st / "dudosos_f3.csv", **op, date_format="%Y-%m-%d %H:%M")
        diag = pd.read_csv(ins["diagnosticos"])
        tam = pd.read_csv(ins["tamano_poisson"])
        resumen = {
            "nota": NOTA,
            "iet_estrella_h": iet,
            "iet_siguiente_h": iet_sig,
            "u_estrella_m": u_estrella,
            "fin_con_2026": str(datos.serie.index[-1]),
            "regla_tp": tp | {"nota": NOTA_TP},
            "dudosos_f3": {r: int((dud["rama"] == r).sum()) for r in R.RAMAS},
            "intensidad_estaciones": intensidad_estaciones(diag, tam, iet, u_estrella),
        }
        json_estricto.escribir(st / "sensibilidades.json", resumen)
        motivos = [] if len(tabla) == len(SENSIBILIDADES) * len(R.RAMAS) else ["Faltan filas de sensibilidad"]
        gate = etapa.Gate(not motivos, {"iet_estrella_h": iet}, motivos)
        return etapa.publicar(st, salida, gate, raices, ins, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-sensibilidades", description="Sensibilidades de §3.7 de la F7")
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
