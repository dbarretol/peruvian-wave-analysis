"""Reproducción en el repositorio de la simulación previa de la F6, ``clima-simulacion-previa`` (§5 y §5.4; ADR-0019).

Sin datos del sitio. Rehace con la biblioteca (``estratificacion``) lo que el contrato tomó de los scripts exploratorios
de ``interim/F06-clima-de-oleaje/simulacion_previa_contrato/``, con el procedimiento de 8 semillas de §5.4 para todas
las fronteras:

1. **Fronteras (§5.1, §5.4):** ξ₁ ∈ {0, −0.1}, λ = 230/46. De σ con f ∈ {0.10, 0.15, 0.20, 0.30, 0.50}; de ξ hacia
   arriba y hacia abajo con f ∈ {0.15, 0.30, 0.50}. Se recorre la rejilla por oleadas y una fila se cierra al cruzar
   el 2 %, que da la misma frontera que recorrerla entera.
2. **Potencia, sesgo y ancho (§5.2, §5.3):** casos nulo y frontera (σ, ξ+ y ξ−) con f ∈ {0.15, 0.3, 0.5},
   N ∈ {170, 230, 320}, 400 réplicas. Anderson–Darling con α ∈ {0.025, 0.05, 0.10}; razón de verosimilitudes;
   ΔRL100 con nulo simulado; sesgo de la mediana y ancho P10–P90 de los tres modelos.
3. **Barrido de umbral de la F7 como disparador (§5.2):** nulo con f = 0.3 y frontera de ξ+, 300 réplicas.
4. **Mínimo de picos del modelo con ξ común (§5.4):** frontera de σ con f ∈ {0.10, …, 0.30}, N ∈ {100, 140, 170,
   230, 320}, 1 000 réplicas; el criterio fijado antes en ``revision.md`` §7.

Reanudable por tareas (``reanudable``), con avance en pantalla. ``--previo`` cronometra y no publica. Salida en
``processed/F06-clima-de-oleaje/simulacion-previa/``.
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
from ..extremos.gpd import ajustar_gpd
from ..validacion import reanudable
from . import estratificacion as E

ETAPA = orquestador.Etapa("clima-simulacion-previa", Path(__file__).resolve(), Path("."))
BASES = (0.0, -0.1)
LAM = 230 / E.ANIOS
FR_SIGMA = (0.10, 0.15, 0.20, 0.30, 0.50)
FR_XI = (0.15, 0.30, 0.50)
FR_PARTES23 = (0.15, 0.30, 0.50)
NS_PARTES23 = (170, 230, 320)
R_PARTES23 = 400
ALFAS_AD = (0.025, 0.05, 0.10)
QS_BARRIDO = tuple(float(q) for q in np.round(np.arange(0.0, 0.61, 0.1), 2))
R_BARRIDO = 300
FR_MINIMO = (0.10, 0.15, 0.20, 0.30)
NS_MINIMO = (100, 140, 170, 230, 320)
R_MINIMO = 1000
TROZO = 100  # réplicas por tarea
FALLOS_MAX = 0.01


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / "F06-clima-de-oleaje" / "simulacion-previa"


def ruta_avance(raiz: Path) -> Path:
    return Path(raiz) / "interim" / "F06-clima-de-oleaje" / "simulacion-previa-avance"


# --- Trabajos (importables por los procesos hijos) ------------------------------------------------------


def trabajo_frontera(args) -> list[float]:
    """ΔRL100 (%) de las 8 semillas en un punto de la rejilla."""
    tipo, xi1, f, valor, lam = args
    r, xi2 = E.parametros(tipo, xi1, valor)
    return [
        E.delta_grande(
            xi1,
            f,
            r,
            xi2,
            lam,
            semillas.generador("f06", "frontera", tipo, f"{xi1:g}", f"{f:g}", f"{valor:g}", f"{lam:.6g}", str(s)),
        )
        for s in range(E.SEMILLAS_FRONTERA)
    ]


def trabajo_modelos(args) -> list[dict]:
    parte, caso, xi1, xi2, r, N, indices, por_estrato = args
    n1, n2 = E.partir(N, caso_f(caso))
    return [
        E.replica_modelos(
            xi1, xi2, r, n1, n2, semillas.generador("f06", parte, caso, str(N), str(i)), por_estrato=por_estrato
        )
        for i in indices
    ]


def caso_f(caso: str) -> float:
    """La fracción va en el nombre del caso: ``<tipo>_xi1=<ξ₁>_f=<f>``."""
    return float(caso.split("_f=")[1])


def nivel_retorno_un_estrato(lam: float, xi: float, sigma: float, u: float) -> float:
    return E.rl_mezcla([(lam, xi, sigma)], u=u)


def replica_barrido(xi1, xi2, f, N, rng) -> dict:
    """Barrido de umbral al estilo de la F7 sobre los picos de una mezcla (sin estratos)."""
    n1, n2 = E.partir(N, f)
    a, b = E.muestra(xi1, xi2, 1.0, n1, n2, rng)
    x = np.concatenate([a, b])
    rls, xis = [], []
    for q in QS_BARRIDO:
        v = float(np.quantile(x, q)) if q > 0 else 0.0
        e = x[x > v] - v
        aj = ajustar_gpd(e)
        if not aj.valido:
            return {"fallo": True}
        rls.append(nivel_retorno_un_estrato(e.size / E.ANIOS, aj.xi, aj.sigma, E.U_SIM + v))
        xis.append(aj.xi)
    rls = np.array(rls)
    ventana = any(
        (rls[i : i + 3].max() - rls[i : i + 3].min()) / np.median(rls[i : i + 3]) < 0.02 for i in range(len(rls) - 2)
    )
    return {"fallo": False, "sin_ventana": not ventana, "d": xis[5] - xis[0]}


def trabajo_barrido(args) -> list[dict]:
    caso, xi1, xi2, N, indices = args
    f = caso_f(caso)
    return [
        replica_barrido(xi1, xi2, f, N, semillas.generador("f06", "barrido", caso, str(N), str(i))) for i in indices
    ]


# --- Fronteras por oleadas --------------------------------------------------------------------------------


def filas_frontera() -> list[tuple[str, float, float]]:
    filas = [(E.SIGMA, xi1, f) for xi1 in BASES for f in FR_SIGMA]
    filas += [(t, xi1, f) for t in (E.XI_ARRIBA, E.XI_ABAJO) for xi1 in BASES for f in FR_XI]
    return filas


def fronteras(avance: Path, procesos_: int, filas, lam: float = LAM, aviso=print) -> tuple[list, dict]:
    """Recorre las filas por oleadas (un valor de la rejilla por fila abierta en cada oleada)."""
    hechos: dict[tuple, list] = {fila: [] for fila in filas}
    abiertas = list(filas)
    ola = 0
    while abiertas:
        tareas = []
        for tipo, xi1, f in abiertas:
            valor = E.rejilla(tipo, xi1)[ola]
            tareas.append((f"fr_{tipo}_{xi1:g}_{f:g}_{valor:g}", (tipo, xi1, f, valor, lam)))
        aviso(f"frontera, oleada {ola + 1}: {len(tareas)} filas abiertas")
        res = reanudable.ejecutar(trabajo_frontera, tareas, avance, procesos_, lambda t: aviso(f"  {t}"))
        siguen = []
        for (_, (tipo, xi1, f, valor, _l)), d in zip(tareas, res, strict=True):
            hechos[(tipo, xi1, f)].append((valor, d))
            cruzo = abs(float(np.nanmean(d))) >= E.EFECTO_PCT
            if not cruzo and ola + 1 < len(E.rejilla(tipo, xi1)):
                siguen.append((tipo, xi1, f))
        abiertas = siguen
        ola += 1
    out = [E.elegir_frontera(t, x, f, hechos[(t, x, f)]) for t, x, f in filas]
    return out, {f"{t}_{x:g}_{f:g}": [(v, list(map(float, d))) for v, d in hechos[(t, x, f)]] for t, x, f in filas}


# --- Resúmenes -------------------------------------------------------------------------------------------


def _sesgo_ancho(x: np.ndarray, verdadero: float) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan"), float("nan")
    sesgo = 100.0 * (float(np.median(x)) / verdadero - 1.0)
    ancho = 100.0 * float(np.quantile(x, 0.9) - np.quantile(x, 0.1)) / verdadero
    return sesgo, ancho


def resumen_modelos(xi1, xi2, r, f, N, reps: list[dict], modelos=("unico", "comun", "sep")) -> dict:
    n1, n2 = E.partir(N, f)
    verdadero = E.rl_mezcla([(n1 / E.ANIOS, xi1, E.SIGMA1), (n2 / E.ANIOS, xi2, E.SIGMA1 * r)])
    fila = {"xi1": xi1, "xi2": xi2, "r": r, "f": f, "N": N, "n2": n2, "replicas": len(reps), "rl_verdadero": verdadero}
    for m in modelos:
        x = np.array([o[m] for o in reps], dtype=float)
        fila[f"{m}_fallos"] = float(np.mean(~np.isfinite(x)))
        fila[f"{m}_sesgo_pct"], fila[f"{m}_ancho_pct"] = _sesgo_ancho(x, verdadero)
    p = np.array([o["ad_p"] for o in reps], dtype=float)
    fila["ad_rechazo"] = {f"{a:g}": E.tasa_rechazo(p, a) for a in ALFAS_AD}
    if "lrt" in reps[0]:
        fila["lrt_rechazo_0.05"] = E.rechazo_lrt([o["lrt"] for o in reps], 0.05)
    return fila


def cumple_minimo(fila: dict) -> bool:
    """Criterio de ``revision.md`` §7, fijado antes de correr la simulación exploratoria."""
    return bool(
        abs(fila["comun_sesgo_pct"]) < abs(fila["unico_sesgo_pct"])
        and fila["comun_ancho_pct"] <= 1.1 * fila["unico_ancho_pct"]
        and fila["comun_fallos"] < FALLOS_MAX
    )


def minimo_n2(casos: list[dict]) -> int | None:
    for m in sorted({c["n2"] for c in casos}):
        if all(c["cumple"] for c in casos if c["n2"] >= m):
            return m
    return None


def _trozos(n: int) -> list[list[int]]:
    return [list(range(a, min(a + TROZO, n))) for a in range(0, n, TROZO)]


def _nombre(tipo: str, xi1: float, f: float) -> str:
    return f"{tipo}_xi1={xi1:g}_f={f:g}"


def casos_partes23(fr: list[E.Frontera]) -> list[tuple[str, float, float, float]]:
    """(caso, ξ₁, ξ₂, r): nulo y fronteras con frontera definida, en las fracciones de §5.1."""
    out = [(_nombre("nulo", xi1, f), xi1, xi1, 1.0) for xi1 in BASES for f in FR_PARTES23]
    for x in fr:
        if x.f in FR_PARTES23 and x.r is not None and not (x.tipo == E.SIGMA and x.rotulo):
            out.append((_nombre(x.tipo, x.xi1, x.f), x.xi1, x.xi2, x.r))
    return out


def potencia_drl(filas: dict, crudos: dict) -> None:
    """Potencia del ΔRL100 (estratificado con ξ por estrato frente a sin estratos) con el P95 del nulo (misma f, N)."""
    for (caso, N), fila in filas.items():
        nulo = crudos[(_nombre("nulo", fila["xi1"], fila["f"]), N)]
        d0 = np.abs([o["sep"] / o["unico"] - 1 for o in nulo if np.isfinite(o["sep"]) and np.isfinite(o["unico"])])
        c95 = float(np.quantile(d0, 0.95))
        d = np.abs([o["sep"] / o["unico"] - 1 for o in crudos[(caso, N)]])
        fila["drl_c95_pct"] = 100 * c95
        fila["drl_rechazo"] = float(np.mean(np.nan_to_num(d, nan=0.0) > c95))


# --- Etapa -------------------------------------------------------------------------------------------------


def _previo(procesos_: int) -> None:
    t0 = time.time()
    trabajo_frontera((E.SIGMA, 0.0, 0.15, 2.0, LAM))
    t_fr = time.time() - t0
    t0 = time.time()
    trabajo_modelos(("previo", "nulo_xi1=0_f=0.3", 0.0, 0.0, 1.0, 230, list(range(20)), True))
    t_mod = (time.time() - t0) / 20
    t0 = time.time()
    trabajo_barrido(("nulo_xi1=0_f=0.3", 0.0, 0.0, 230, list(range(10))))
    t_bar = (time.time() - t0) / 10
    n_fr = sum(12 for _ in filas_frontera())  # unas 12 oleadas por fila en promedio (cota gruesa)
    n_mod = (12 + 14) * len(NS_PARTES23) * R_PARTES23 + 8 * len(NS_MINIMO) * R_MINIMO
    n_bar = 8 * len(NS_PARTES23) * R_BARRIDO
    total = (n_fr * t_fr + n_mod * t_mod + n_bar * t_bar) / max(procesos_, 1) / 60
    print(
        f"previo: punto de frontera {t_fr:.1f} s, réplica de modelos {t_mod:.2f} s, réplica de barrido {t_bar:.2f} s; "
        f"total estimado unos {total:.0f} min con {procesos_} procesos",
        flush=True,
    )


def simulacion_previa(raiz_datos=None, procesos_: int = 4, previo: bool = False) -> Path | None:
    if previo:
        _previo(procesos_)
        return None
    raiz = rutas.directorio_datos(raiz_datos)
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    avance = ruta_avance(raiz) / reanudable.clave(orquestador.huella_etapa(ETAPA), TROZO)
    print(f"Avance reanudable en {avance}", flush=True)
    t0 = time.time()

    def aviso(t: str) -> None:
        print(f"[{(time.time() - t0) / 60:5.1f} min] {t}", flush=True)

    with etapa.staging_protegido(salida, raices) as st:
        # 1. Fronteras
        fr, rejillas = fronteras(avance, procesos_, filas_frontera(), LAM, aviso)
        for x in fr:
            aviso(f"frontera {x.tipo} xi1={x.xi1:g} f={x.f:g}: r={x.r} xi2={x.xi2} ({x.media_pct:+.2f} %) {x.rotulo}")

        # 2. Partes 2 y 3
        casos = casos_partes23(fr)
        tareas = [
            (f"p23_{c}_{N}_{tr[0]:04d}", ("p23", c, xi1, xi2, r, N, tr, True))
            for c, xi1, xi2, r in casos
            for N in NS_PARTES23
            for tr in _trozos(R_PARTES23)
        ]
        aviso(f"partes 2 y 3: {len(casos)} casos x {len(NS_PARTES23)} N x {R_PARTES23} réplicas")
        res = reanudable.ejecutar(trabajo_modelos, tareas, avance, procesos_, lambda t: aviso(f"  partes 2-3: {t}"))
        crudos: dict = {}
        for (_, (_, c, *_resto)), parte in zip(tareas, res, strict=True):
            crudos.setdefault((c, _resto[3]), []).extend(parte)
        filas23 = {}
        for c, xi1, xi2, r in casos:
            for N in NS_PARTES23:
                filas23[(c, N)] = {"caso": c, **resumen_modelos(xi1, xi2, r, caso_f(c), N, crudos[(c, N)])}
        potencia_drl(filas23, crudos)

        # 3. Barrido de umbral
        casos_b = [(_nombre("nulo", xi1, 0.3), xi1, xi1) for xi1 in BASES]
        casos_b += [
            (_nombre(x.tipo, x.xi1, x.f), x.xi1, x.xi2) for x in fr if x.tipo == E.XI_ARRIBA and x.r is not None
        ]
        tareas_b = [
            (f"bar_{c}_{N}_{tr[0]:04d}", (c, xi1, xi2, N, tr))
            for c, xi1, xi2 in casos_b
            for N in NS_PARTES23
            for tr in _trozos(R_BARRIDO)
        ]
        aviso(f"barrido de umbral: {len(casos_b)} casos")
        res_b = reanudable.ejecutar(trabajo_barrido, tareas_b, avance, procesos_, lambda t: aviso(f"  barrido: {t}"))
        crudos_b: dict = {}
        for (_, (c, _x1, _x2, N, _tr)), parte in zip(tareas_b, res_b, strict=True):
            crudos_b.setdefault((c, N), []).extend(parte)
        filas_b = []
        for c, xi1, xi2 in casos_b:
            for N in NS_PARTES23:
                ok = [o for o in crudos_b[(c, N)] if not o["fallo"]]
                nulo = [o for o in crudos_b[(_nombre("nulo", xi1, 0.3), N)] if not o["fallo"]]
                c95 = float(np.quantile([o["d"] for o in nulo], 0.95))
                filas_b.append({
                    "caso": c, "xi1": xi1, "xi2": xi2, "N": N, "fallos": len(crudos_b[(c, N)]) - len(ok),
                    "f7_sin_ventana": float(np.mean([o["sin_ventana"] for o in ok])),
                    "d_mediana": float(np.median([o["d"] for o in ok])), "d_c95_nulo": c95,
                    "rechazo_d": float(np.mean([o["d"] > c95 for o in ok])),
                })  # fmt: skip

        # 4. Mínimo del modelo con ξ común
        fr_min = [x for x in fr if x.tipo == E.SIGMA and x.f in FR_MINIMO]
        tareas_m = [
            (f"min_{_nombre('sigma', x.xi1, x.f)}_{N}_{tr[0]:04d}",
             ("minimo", _nombre("sigma", x.xi1, x.f), x.xi1, x.xi1, x.r, N, tr, False))
            for x in fr_min
            for N in NS_MINIMO
            for tr in _trozos(R_MINIMO)
        ]  # fmt: skip
        aviso(f"mínimo con xi común: {len(fr_min)} fronteras x {len(NS_MINIMO)} N x {R_MINIMO} réplicas")
        res_m = reanudable.ejecutar(trabajo_modelos, tareas_m, avance, procesos_, lambda t: aviso(f"  mínimo: {t}"))
        crudos_m: dict = {}
        for (_, (_, c, _x1, _x2, _r, N, _tr, _p)), parte in zip(tareas_m, res_m, strict=True):
            crudos_m.setdefault((c, N), []).extend(parte)
        casos_m = []
        for x in fr_min:
            for N in NS_MINIMO:
                fila = resumen_modelos(x.xi1, x.xi1, x.r, x.f, N, crudos_m[(_nombre("sigma", x.xi1, x.f), N)],
                                       ("unico", "comun"))  # fmt: skip
                fila["cumple"] = cumple_minimo(fila)
                casos_m.append(fila)
        casos_m.sort(key=lambda c: (c["n2"], c["xi1"], c["f"]))
        minimo = minimo_n2(casos_m)

        fallos = [
            f"{k}: {v:.1%} de ajustes fallidos"
            for fila in [*filas23.values(), *casos_m]
            for k, v in fila.items()
            if k.endswith("_fallos") and v >= FALLOS_MAX
        ]
        resumen = {
            "valores_del_sitio_leidos": False,
            "supuestos": {
                "u": E.U_SIM,
                "anios": E.ANIOS,
                "sigma1": E.SIGMA1,
                "lambda_frontera": LAM,
                "n_grande": E.N_GRANDE,
                "semillas_frontera": E.SEMILLAS_FRONTERA,
            },  # fmt: skip
            "fronteras": [x.como_dict() for x in fr],
            "rejillas_frontera": rejillas,
            "partes23": list(filas23.values()),
            "barrido_umbral": filas_b,
            "minimo_xi_comun": {
                "criterio": "revision.md §7: |sesgo común| < |sesgo sin estratos|, ancho común ≤ 1.1 × ancho sin "
                "estratos y < 1 % de fallos; mínimo = menor n₂ desde el que todos los casos cumplen",
                "casos": casos_m,
                "minimo_n2": minimo,
            },
            "ajustes_fallidos_sobre_1pct": fallos,
            "segundos": round(time.time() - t0),
        }
        json_estricto.escribir(st / "simulacion_previa.json", resumen)
        aviso(f"mínimo de n2 con xi común: {minimo}")
        criterios = {"casos_partes23": len(filas23), "casos_minimo": len(casos_m), "fronteras": len(fr)}
        publicada = etapa.publicar(
            st, salida, etapa.Gate(not fallos, criterios, fallos), raices, {}, raiz,
            semillas.semilla("f06", "simulacion-previa"),
        )  # fmt: skip
        reanudable.limpiar(avance)
        return publicada


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clima-simulacion-previa", description="Reproducción de §5 y §5.4 de la F6")
    ap.add_argument("--datos")
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--previo", action="store_true", help="Cronometra unas unidades; no publica.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = simulacion_previa(a.datos, a.procesos, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    if salida:
        print(f"Publicado en {salida}")
    return 0
