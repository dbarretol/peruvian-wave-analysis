"""Diagnósticos de §3.5 sobre el sitio, ``eventos-diagnosticos`` (contrato de la F7, §3.3 «Qué se calcula en cada
IET» y P11, §3.5, §3.7 punto 6). **Descriptivos: no mueven u\\* ni el IET\\*.**

Qué calcula (funciones en ``diagnosticos.py``):

- **en cada IET de la rejilla, en E0 y en E1 sin corregir** (P11): prueba de Poisson sobre las rachas con intensidad
  armónica, dispersión de los conteos anuales, τ de Kendall de picos consecutivos, separaciones y A² en u\\*;
- **en (u\\*, IET\\*) de las ramas por cuantiles:** lo mismo;
- **sensibilidad de la intensidad (§3.7, punto 6):** la prueba de Poisson con 2 y con 4 estaciones fijas, en
  (u\\*, IET\\*) de cada rama;
- **Süveges y Davison** en E0 sin corregir: 11 celdas, todo el año y mayo–octubre;
- **picos de verano** sobre u\\* por rama y la condición de ADR-0029;
- **tamaño de la prueba completa** con el proceso sintético de §8 (500 réplicas, B = 99): armónica, 2 y 4 estaciones.
  Una intensidad por estaciones con tamaño > 8 % queda rotulada «frágil».

**Umbral en cada IET (precisión n.º 25, propuesta):** el u\\* de la rama, elegido con IET\\* (§3.4). El contrato
no fija el umbral de los diagnósticos por IET.

**Rachas de una era:** las que empiezan dentro de [inicio de la rama, 2025-12-31]. Una racha que empieza antes del
inicio y sigue dentro de él saca de la era su resto y su tiempo muerto: su inicio no es una llegada observable.

Reanudable (``validacion.reanudable``): cada tarea tiene su semilla por nombre, así que el resultado no depende del
número de procesos (G9).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..clima import insumos as CI
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..homogeneidad.cuantiles import cuantiles_mensuales
from ..validacion import reanudable
from . import diagnosticos as D
from . import identificacion as ID
from . import iet as IE
from . import ramas as R
from . import ramas_picos as RP
from . import umbral_datos as UD

ETAPA = orquestador.Etapa("eventos-diagnosticos", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
NOMBRE = CI.NOMBRE
IETS_H = RP.IETS_H
ERAS = ("E0_sin_corregir", "E1_sin_corregir")
MODELOS = ("armonico", "2_estaciones", "4_estaciones")
REPLICAS_TAMANO = 500
B_TAMANO = 99
TROZO_TAMANO = 25
ANIOS_TAMANO = 10.0
UMBRAL_FRAGIL = 0.08
SEMILLA = ("f07", "diagnosticos")

NOTA_E0 = (
    "Lectura rotulada de §3.5, punto 1 (escrita después del conteo de §5.0 y antes de ver ningún pico): la tasa sobre "
    "u_id se multiplica por ≈ 2.1 en 1993 (6.9 → 14.6 eventos por año). La prueba de E0 supone una intensidad que "
    "solo depende de la estación, así que se espera que rechace por ese escalón, y lo mismo la dispersión. Ese "
    "rechazo no dice nada del declustering ni del IET. La marca «dependencia no resuelta por el IET» se evalúa en E1 "
    "(P11). Las ramas por cuantiles comparten los eventos con E0, y también esta lectura."
)
NOTA_UMBRAL_POR_IET = (
    "Precisión n.º 25 (propuesta, pendiente de aceptación): en cada IET de la rejilla, los diagnósticos usan el u* de "
    "la rama, elegido con IET* (§3.4). El contrato no fija el umbral de los diagnósticos por IET."
)
NOTA_P_POISSON = (
    "El p de la prueba de Poisson se reporta como «≤ 1/(B + 1)» cuando ninguna réplica alcanza el A² observado: es el "
    "piso del Monte Carlo (con B = 199, ≤ 1/200), no un valor estimado."
)
NOTA_GARWOOD = (
    "El IC de Garwood de λ supone llegadas de Poisson. La prueba de §3.5 rechaza en E0 y también en E1 (en los 8 IET, "
    "por variabilidad entre años), así que en las dos eras ese intervalo es demasiado estrecho. Pasa a la F9: el "
    "período de retorno depende de λ (rotulado: nota escrita después de ver los diagnósticos)."
)
NOTA_TAMANO = (
    "Observación rotulada, escrita después de ver los diagnósticos: con 2 estaciones fijas el tamaño es del 22.6 %, y "
    "esa sensibilidad queda «frágil». No cambia la regla de la marca (precisión n.º 26): la prueba armónica, con "
    "tamaño correcto (5.0 %), rechaza igual."
)
NOTA_TAU = (
    "τ de picos consecutivos: descriptivo. Los p de los 8 IET no son pruebas independientes, porque los conjuntos de "
    "picos casi coinciden; un p bajo en algunos IET no es un rechazo."
)
NOTA_SD = (
    "Süveges y Davison (2010), descriptivo (P8). F̄(u) = 1 − q nominal y u = cuantil q con peso mensual de los "
    "estados de 3 h del período (en mayo–octubre, de los estados de esos meses; los huecos entre dos excedencias "
    "solo cuentan dentro de un mismo mayo–octubre). Un NaN cuenta como estado sin excedencia. El 0.2 de K·F̄(u) es un "
    "criterio del revisor, no del artículo."
)


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "diagnosticos" / NOMBRE


def ruta_avance(raiz: Path) -> Path:
    return Path(raiz) / "interim" / FASE / "diagnosticos-avance"


# --- Tareas (una semilla por nombre) ------------------------------------------------------------------------------


def _rng(*nombre) -> np.random.Generator:
    return semillas.generador(*SEMILLA, *(str(x) for x in nombre))


def _poisson(a: dict) -> dict:
    t = pd.DatetimeIndex(a["tiempos"])
    r = D.prueba_poisson_rachas(
        t,
        a["validos"],
        a["ini"],
        a["fin"],
        a["marcadas"],
        a["iet_h"],
        D.modelo_por_nombre(a["modelo"], t),
        _rng(*a["nombre"]),
        B=a["B"],
    )
    return r


def _picos(a: dict) -> dict:
    nombre = a["nombre"]
    sobre = a["sobre"]
    exc = sobre["hs"].to_numpy(float) - a["u"]
    return {
        "dispersion": D.dispersion_anual(
            pd.DatetimeIndex(sobre["t_pico"]).year, a["exposicion"], _rng(*nombre, "dispersion"), B=a["B_disp"]
        ),
        "tau": D.tau_consecutivos(sobre["t_pico"], exc, sobre["id_racha"], _rng(*nombre, "tau"), B=a["B_perm"]),
        "a2": D.bondad_a2(sobre["hs"].to_numpy(float), a["u"], _rng(*nombre, "a2"), B=a["B_a2"]),
    }


def _tamano(a: dict) -> list[bool]:
    t = D.rejilla_sintetica(ANIOS_TAMANO)
    mod = D.modelo_por_nombre(a["modelo"], t)
    validos = np.ones(t.size, bool)
    out = []
    for i in a["indices"]:
        rng = _rng("tamano", a["modelo"], i)
        ini, fin, marc = D.proceso_sintetico(t, rng)
        out.append(bool(D.prueba_poisson_rachas(t, validos, ini, fin, marc, 48.0, mod, rng, B=a["B"])["rechaza"]))
    return out


def _trabajo(args) -> object:
    tipo, a = args
    return {"poisson": _poisson, "picos": _picos, "tamano": _tamano}[tipo](a)


# --- Preparación --------------------------------------------------------------------------------------------------


def era(datos: CI.Datos, rama: str, eventos: pd.DataFrame, iet: float):
    """(ini, fin, id_racha, validos, rachas que cruzan el inicio, exposición por año) de la rama en ese IET."""
    t = pd.DatetimeIndex(datos.serie.index)
    inicio = R.inicio_rama(rama, pd.Timestamp(t[0]), datos.inicio_e1)
    a = int(t.searchsorted(inicio))
    b = int(t.searchsorted(R.FIN_MODELO, side="right")) - 1
    serie = datos.ramas[rama].reindex(t)
    validos_serie = serie.notna().to_numpy()
    muertas = int(round(iet / D.PASO_H))
    ini, fin, ids, validos, cruzan = D.rachas_de_era(eventos, t.size, a, b, muertas, validos_serie)
    expo_valida = np.zeros(t.size, bool)
    expo_valida[a : b + 1] = validos_serie[a : b + 1]
    return ini, fin, ids, validos, cruzan, D.exposicion_por_anio(t, expo_valida)


def tareas_y_contexto(datos, u_id, picos, u_estrella, iet_estrella, iets, B):
    """Lista de tareas (nombre, args) y el contexto para armar las tablas."""
    t = pd.DatetimeIndex(datos.serie.index)
    hs0 = datos.serie["VHM0"].astype(float)
    tareas, ctx = [], {}
    combos = [(iet, r) for iet in iets for r in ERAS] + [(iet_estrella, r) for r in R.RAMAS if R.es_variante(r)]
    eventos = {iet: ID.identificar(hs0, u_id, iet) for iet in sorted({c[0] for c in combos})}
    for iet, rama in combos:
        u = float(u_estrella[rama])
        p = picos[(picos["rama"] == rama) & (picos["iet_h"] == iet)]
        sobre = p[p["hs"] > u].sort_values("t_pico")
        ini, fin, ids, validos, cruzan, expo = era(datos, rama, eventos[iet], iet)
        marcadas = np.isin(ids, sobre["id_racha"].unique())
        clave = f"{rama}__{iet:g}"
        inicio = R.inicio_rama(rama, pd.Timestamp(t[0]), datos.inicio_e1)
        ev = eventos[iet]
        tp = pd.DatetimeIndex(ev["t_pico"])
        ctx[(rama, iet)] = {
            "sobre": sobre,
            "t_eventos": tp[(tp >= inicio) & (tp <= R.FIN_MODELO)],
            "cruzan": cruzan,
            "marcadas": int(marcadas.sum()),
            "picos_en_rachas_fuera": int((~sobre["id_racha"].isin(ids)).sum()),
        }
        base = {"tiempos": t, "validos": validos, "ini": ini, "fin": fin, "marcadas": marcadas, "iet_h": iet}
        modelos = MODELOS if iet == iet_estrella else ("armonico",)
        for m in modelos:
            tareas.append(
                (
                    f"poisson__{clave}__{m}",
                    ("poisson", base | {"modelo": m, "B": B["poisson"], "nombre": ("poisson", rama, iet, m)}),
                )
            )
        tareas.append(
            (
                f"picos__{clave}",
                (
                    "picos",
                    {
                        "sobre": sobre,
                        "u": u,
                        "exposicion": expo,
                        "B_disp": B["disp"],
                        "B_perm": B["perm"],
                        "B_a2": B["a2"],
                        "nombre": ("picos", rama, iet),
                    },
                ),
            )
        )
    for m in MODELOS:
        for i0 in range(0, B["replicas_tamano"], TROZO_TAMANO):
            idx = list(range(i0, min(i0 + TROZO_TAMANO, B["replicas_tamano"])))
            tareas.append((f"tamano__{m}__{i0:04d}", ("tamano", {"modelo": m, "indices": idx, "B": B["tamano"]})))
    return tareas, ctx


def suveges_davison(hs0: pd.Series, iets) -> pd.DataFrame:
    """§3.5, punto 6, en E0 sin corregir (hasta 2025-12-31): las 11 celdas, todo el año y mayo–octubre."""
    s = hs0[hs0.index <= R.FIN_MODELO]
    v = s.to_numpy(float)
    meses = s.index.month.to_numpy()
    filas = []
    for periodo in ("todo_el_anio", "mayo_octubre"):
        en = np.ones(v.size, bool) if periodo == "todo_el_anio" else np.isin(meses, D.MESES_MAYO_OCTUBRE)
        for c in D.celdas_sd(iets):
            u = float(cuantiles_mensuales(s[en], [c["q"]])[0])
            pos = np.flatnonzero(en & (np.nan_to_num(v, nan=-np.inf) > u))
            bloques = None if periodo == "todo_el_anio" else s.index.year.to_numpy()[pos]
            r = D.kgaps(pos, c["K"], 1.0 - c["q"], bloques)
            filas.append(
                {
                    "periodo": periodo,
                    "q": c["q"],
                    "iet_h": c["iet_h"],
                    "K": c["K"],
                    "KF": c["KF"],
                    "u_m": u,
                    "excedencias": int(pos.size),
                }
                | r
            )
    return pd.DataFrame(filas)


def tabla_sd(sd: pd.DataFrame) -> list[dict]:
    """Resumen completo de Süveges y Davison: cada período (todo el año, mayo–octubre) con cada cuantil (P99, P99.5),
    para no comparar cuantiles y temporadas mezclados."""
    out = []
    for (periodo, q), g in sd.groupby(["periodo", "q"], sort=True):
        out.append(
            {
                "periodo": periodo,
                "q": float(q),
                "u_m": float(g["u_m"].iloc[0]),
                "celdas": int(len(g)),
                "excedencias": int(g["excedencias"].iloc[0]),
                "theta_min": float(g["theta"].min()),
                "theta_max": float(g["theta"].max()),
                "p_min": float(g["p"].min()),
                "p_max": float(g["p"].max()),
                "rechazos": int((g["p"] <= 0.05).sum()),
            }
        )
    return out


# --- Armado ---------------------------------------------------------------------------------------------------


def armar(tareas, resultados, ctx, iet_estrella, u_estrella):
    res = {n: r for (n, _), r in zip(tareas, resultados, strict=True)}
    filas, seps = [], []
    for (rama, iet), c in ctx.items():
        clave = f"{rama}__{iet:g}"
        pk = res[f"picos__{clave}"]
        fila = {
            "rama": rama,
            "iet_h": iet,
            "u_m": float(u_estrella[rama]),
            "picos": int(len(c["sobre"])),
            "rachas_marcadas": c["marcadas"],
            "rachas_que_cruzan_el_inicio": c["cruzan"],
            "picos_en_rachas_fuera_de_la_era": c["picos_en_rachas_fuera"],
        }
        for m in MODELOS if iet == iet_estrella else ("armonico",):
            pp = res[f"poisson__{clave}__{m}"]
            for k in (
                "ad",
                "p_ad",
                "p_ad_en_el_piso",
                "p_ad_texto",
                "rechaza",
                "ks",
                "p_ks",
                "n_rachas",
                "n_llegadas",
                "n_intervalos",
            ):
                fila[f"poisson_{m}_{k}"] = pp[k]
        d = pk["dispersion"]
        fila |= {"dispersion_indice": d["indice"], "dispersion_p": d["p"], "anios": d["anios"]}
        for sub in ("todos", "menos_de_7_dias", "misma_racha"):
            for k in ("pares", "tau", "p", "rechaza"):
                fila[f"tau_{sub}_{k}"] = pk["tau"][sub][k]
        a = pk["a2"]
        fila |= {"a2": a["a2"], "a2_p": a["p"], "a2_rechaza": a.get("rechaza")}
        sp = D.separaciones(c["sobre"]["t_pico"], iet)
        se = D.separaciones(c["t_eventos"], iet)
        fila |= {
            "sep_picos_fraccion_franja": sp["fraccion_en_franja"],
            "sep_picos_en_franja": sp["en_franja"],
            "sep_eventos_fraccion_franja": se["fraccion_en_franja"],
            "sep_eventos_en_franja": se["en_franja"],
            "sep_eventos_n": se["n"],
        }
        filas.append(fila)
        for conj, s in (("picos_sobre_u", sp), ("eventos_sobre_u_id", se)):
            seps.append(pd.DataFrame({"rama": rama, "iet_h": iet, "conjunto": conj, "horas": s["horas"]}))
    tam = []
    for m in MODELOS:
        r = [x for n, x in res.items() if n.startswith(f"tamano__{m}__") for x in x]
        tam.append(
            {
                "modelo": m,
                "replicas": len(r),
                "rechazos": int(sum(r)),
                "tamano": float(np.mean(r)),
                "fragil": bool(m != "armonico" and np.mean(r) > UMBRAL_FRAGIL),
            }
        )
    return pd.DataFrame(filas), pd.concat(seps, ignore_index=True), pd.DataFrame(tam)


def marcas(tabla: pd.DataFrame, iets) -> dict:
    """«Dependencia no resuelta por el IET»: la prueba de Poisson (armónica) o el τ de todos los pares rechazan en
    toda la rejilla. Se evalúa en E1 sin corregir; la de E0 se publica con la lectura de §3.5."""
    out = {}
    for rama in ERAS:
        g = tabla[(tabla["rama"] == rama) & tabla["iet_h"].isin(iets)]
        pois = bool(g["poisson_armonico_rechaza"].all())
        tau = bool(g["tau_todos_rechaza"].fillna(False).astype(bool).all())
        out[rama] = {
            "poisson_rechaza_en_toda_la_rejilla": pois,
            "tau_rechaza_en_toda_la_rejilla": tau,
            "dependencia_no_resuelta_por_el_iet": pois or tau,
            "iets_con_rechazo_poisson": g.loc[g["poisson_armonico_rechaza"].astype(bool), "iet_h"].tolist(),
            "iets_con_rechazo_tau": g.loc[g["tau_todos_rechaza"].fillna(False).astype(bool), "iet_h"].tolist(),
        }
    out["E0_sin_corregir"]["lectura"] = NOTA_E0
    out["se_evalua_en"] = "E1_sin_corregir"
    return out


def verano(picos, u_estrella, iet_estrella) -> dict:
    por = {}
    for rama in R.RAMAS:
        p = picos[(picos["rama"] == rama) & (picos["iet_h"] == iet_estrella)]
        por[rama] = D.picos_de_verano(p["t_pico"], p["hs"], float(u_estrella[rama]))
    return {
        "picos_nov_abr_sobre_u": por,
        "condicion_adr_0029": bool(min(por.values()) < 30),
        "lectura": "ADR-0029 supone algún estrato con < 30 picos en alguna rama; si deja de valer, se avisa.",
    }


def diagnosticar(
    datos,
    u_id,
    p95,
    u_estrella,
    iet_estrella,
    candidatos,
    revision,
    avance: Path,
    procesos_: int,
    B: dict,
    iets=IETS_H,
    avisar=None,
):
    _, picos, _, bajo = RP.ramas_picos(datos, u_id, p95, candidatos, revision, iets=tuple(iets))
    if bajo:
        raise GateFallido(f"G8: {len(bajo)} picos de variante bajo el P95 de la F3 ({R.REVISION_REQUERIDA})")
    tareas, ctx = tareas_y_contexto(datos, u_id, picos, u_estrella, iet_estrella, iets, B)
    resultados = reanudable.ejecutar(_trabajo, tareas, avance, procesos_, avisar)
    tabla, seps, tam = armar(tareas, resultados, ctx, iet_estrella, u_estrella)
    sd = suveges_davison(datos.serie["VHM0"].astype(float), iets)
    return tabla, seps, tam, sd, marcas(tabla, iets), verano(picos, u_estrella, iet_estrella)


# --- Figuras ---------------------------------------------------------------------------------------------------


def figuras(tabla, seps, sd, iet_estrella, carpeta: Path) -> list[str]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    carpeta.mkdir(parents=True, exist_ok=True)
    hechas = []
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (col, titulo) in zip(
        axs,
        (
            ("poisson_armonico_p_ad", "Poisson sobre rachas (A², armónica)"),
            ("tau_todos_p", "τ de Kendall, picos consecutivos"),
            ("dispersion_p", "Dispersión de los conteos anuales"),
        ),
        strict=True,
    ):
        for rama in ERAS:
            g = tabla[tabla["rama"] == rama].sort_values("iet_h")
            ax.plot(g["iet_h"], g[col], marker="o", label=rama.replace("_", " "))
        ax.axhline(0.05, color="k", lw=0.8, ls=":")
        ax.axvline(iet_estrella, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("IET (h)")
        ax.set_title(titulo)
    axs[0].set_ylabel("p")
    axs[0].legend()
    fig.suptitle("Diagnósticos por IET en u* de cada era (descriptivos; línea de puntos: p = 0.05)")
    fig.tight_layout()
    fig.savefig(carpeta / "p_frente_al_iet.png", dpi=130)
    plt.close(fig)
    hechas.append("p_frente_al_iet.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, periodo in zip(axs, ("todo_el_anio", "mayo_octubre"), strict=True):
        g = sd[sd["periodo"] == periodo]
        for q, gq in g.groupby("q"):
            ax.plot(gq["iet_h"], gq["p"], marker="o", label=f"P{100 * q:g}")
        ax.axhline(0.05, color="k", lw=0.8, ls=":")
        ax.axvline(iet_estrella, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("IET (h)")
        ax.set_title("Todo el año" if periodo == "todo_el_anio" else "Mayo–octubre")
    axs[0].set_ylabel("p de T (matriz de información)")
    axs[0].legend(title="Umbral")
    fig.suptitle("Süveges y Davison en E0 sin corregir: celdas con K·F̄(u) ≤ 0.2 (descriptivo)")
    fig.tight_layout()
    fig.savefig(carpeta / "suveges_davison.png", dpi=130)
    plt.close(fig)
    hechas.append("suveges_davison.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, rama in zip(axs, ERAS, strict=True):
        h = seps[(seps["rama"] == rama) & (seps["iet_h"] == iet_estrella) & (seps["conjunto"] == "eventos_sobre_u_id")]
        ax.hist(h["horas"] / 24.0, bins=np.arange(0, 31, 0.5), color="0.5")
        ax.axvspan(
            iet_estrella / 24,
            (iet_estrella + D.ANCHO_FRANJA_H) / 24,
            color="C1",
            alpha=0.3,
            label="[IET*, IET* + 24 h]",
        )
        ax.set_xlabel("Separación entre picos consecutivos (días; hasta 30)")
        ax.set_title(rama.replace("_", " ") + ": todos los eventos sobre u_id")
    axs[0].set_ylabel("Pares")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(carpeta / "separaciones.png", dpi=130)
    plt.close(fig)
    hechas.append("separaciones.png")
    return hechas


# --- Etapa ----------------------------------------------------------------------------------------------------


B_PUBLICACION = {
    "poisson": D.B_POISSON,
    "a2": D.B_A2,
    "perm": D.B_PERMUTACION,
    "disp": D.B_DISPERSION,
    "tamano": B_TAMANO,
    "replicas_tamano": REPLICAS_TAMANO,
}


def ejecutar(raiz_datos=None, procesos_: int = 4, previo: bool = False) -> Path | None:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = RP.rutas_insumos(raiz)
    d_iet, d_umb = IE.ruta_salida(raiz), UD.ruta_salida(raiz)
    for d in (d_iet, d_umb):
        etapa.exigir_publicada(d)
    ins |= {"iet": d_iet / "iet.json", "umbral": d_umb / "umbral.json"}
    hashes = etapa.exigir_insumos(ins)
    datos = CI.cargar(raiz, ins, NOMBRE)
    p95 = float(json_estricto.leer(ins["resumen_candidatos"])["p95_hs_m"])
    iet_estrella = float(json_estricto.leer(ins["iet"])["iet_estrella_h"])
    umb = json_estricto.leer(ins["umbral"])
    u_estrella = {r: float(umb["ramas"][r][UD.REGLA]["u_m"]) for r in R.RAMAS}
    cand, rev = pd.read_csv(ins["candidatos"]), pd.read_csv(ins["declustering_dudoso"])
    if previo:
        return _previo(datos, p95, u_estrella, iet_estrella, cand, rev, procesos_)
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA), hashes, B_PUBLICACION)
    avance = ruta_avance(raiz) / clave
    print(f"Avance reanudable en {avance}", flush=True)
    with etapa.staging_protegido(salida, raices) as st:
        aviso = lambda t: print(f"  diagnósticos: {t}", flush=True)  # noqa: E731
        tabla, seps, tam, sd, mk, ver = diagnosticar(
            datos, p95, p95, u_estrella, iet_estrella, cand, rev, avance, procesos_, B_PUBLICACION, avisar=aviso
        )
        op = {"index": False, "float_format": "%.6g", "lineterminator": "\n"}
        tabla.to_csv(st / "diagnosticos.csv", **op)
        seps.to_csv(st / "separaciones.csv", **op)
        tam.to_csv(st / "tamano_poisson.csv", **op)
        sd.to_csv(st / "suveges_davison.csv", **op)
        figs = figuras(tabla, seps, sd, iet_estrella, st / "figuras")
        sd_vacia = bool(sd["p"].notna().sum() == 0)
        resumen = {
            "iet_estrella_h": iet_estrella,
            "u_estrella_m": u_estrella,
            "iets_h": list(IETS_H),
            "B": B_PUBLICACION,
            "umbral_por_iet": NOTA_UMBRAL_POR_IET,
            "marcas": mk,
            "picos_de_verano": ver,
            "tamano_poisson": tam.to_dict(orient="records"),
            "notas": {"p_poisson": NOTA_P_POISSON, "garwood": NOTA_GARWOOD, "tamano": NOTA_TAMANO, "tau": NOTA_TAU},
            "suveges_davison": {
                "celdas": int(len(sd) // 2),
                "nota": NOTA_SD,
                "retirado": sd_vacia,
                "rechazos_todo_el_anio": int((sd.query("periodo == 'todo_el_anio'")["p"] <= 0.05).sum()),
                "rechazos_mayo_octubre": int((sd.query("periodo == 'mayo_octubre'")["p"] <= 0.05).sum()),
                "por_periodo_y_cuantil": tabla_sd(sd),
            },
            "figuras": figs,
        }
        json_estricto.escribir(st / "diagnosticos.json", resumen)
        motivos = [] if len(tam) == len(MODELOS) else ["Faltan modelos en el tamaño"]
        gate = etapa.Gate(not motivos, {"iet_estrella_h": iet_estrella}, motivos)
        publicada = etapa.publicar(st, salida, gate, raices, ins, raiz, semillas.semilla(*SEMILLA))
        reanudable.limpiar(avance)
        return publicada


def _previo(datos, p95, u_estrella, iet_estrella, cand, rev, procesos_) -> None:
    """Mide una tarea de cada tipo y estima la corrida completa."""
    import tempfile  # noqa: PLC0415

    B = B_PUBLICACION | {"replicas_tamano": TROZO_TAMANO}
    t0 = time.time()
    _, picos, _, _ = RP.ramas_picos(datos, p95, p95, cand, rev, iets=(iet_estrella,))
    tareas, _ = tareas_y_contexto(datos, p95, picos, u_estrella, iet_estrella, (iet_estrella,), B)
    print(f"previo: preparación {time.time() - t0:.0f} s", flush=True)
    tiempos = {}
    with tempfile.TemporaryDirectory():
        for tipo in ("poisson", "picos", "tamano"):
            n, args = next(x for x in tareas if x[1][0] == tipo)
            t0 = time.time()
            _trabajo(args)
            tiempos[tipo] = time.time() - t0
            print(f"previo: una tarea {tipo} ({n}) {tiempos[tipo]:.1f} s", flush=True)
    n_pois = len(IETS_H) * 2 + 2 + 2 * 4
    n_pic = len(IETS_H) * 2 + 2
    n_tam = len(MODELOS) * REPLICAS_TAMANO // TROZO_TAMANO
    total = (n_pois * tiempos["poisson"] + n_pic * tiempos["picos"] + n_tam * tiempos["tamano"]) / max(procesos_, 1)
    print(f"previo: corrida completa ≈ {total / 60:.0f} min con {procesos_} procesos (más la preparación)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-diagnosticos", description="Diagnósticos de §3.5 de la F7")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--previo", action="store_true", help="mide una tarea de cada tipo y estima el total")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = ejecutar(a.datos, a.procesos, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    if salida is not None:
        print(f"Publicado en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
