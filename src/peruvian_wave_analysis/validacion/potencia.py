"""Potencia, error de tipo I y subida esperada de la F5 (contrato v6, §5; ADR-0024, punto 2). Sin valores observados.

Una **réplica** simula, con un punto de la rejilla de supuestos y sesgo nulo, las observaciones de todos los lados
principales (T0, A, A1, A2 de la altimetría; T0b y T1b de la boya 32302) y guarda, para cada prueba, lo necesario
para la potencia por desplazamiento (``simulacion``): V1a, V1b (con y sin el término por lado), V2a, V3a y V3b, y
B_cola, B_alta y sus pruebas para la subida esperada.

V2b no se simula: la boya 32012 no atribuye C2 (regla del ⅓, contrato §19).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..comun import semillas
from . import config as C
from . import simulacion as S

LADOS_SIM = ("T0", "A", "A1", "A2", "T0b", "T1b")
CLASE_SI = {"T0": "T0", "T0b": "T0", "A": "asimilado", "A1": "asimilado", "A2": "asimilado", "T1b": "asimilado"}
ESTS_ATRIB = ("P50", "P90", "P99", "max")  # §4.1
ESTS_V3A = ("P90", "P99", "max")  # §4.3


@dataclass(frozen=True)
class Punto:
    """Un punto de la rejilla de supuestos (§5)."""

    nombre: str
    si_asimilado: float
    si_t0: float
    rho: float
    factor_cal: float


def rejilla() -> list[Punto]:
    """SI (3, emparejados) × calibración y ruido (3) con ρ nominal, más ρ = 0.5 en el nominal: 10 puntos."""
    out = []
    for sa, s0 in zip(C.SI_REJILLA["asimilado"], C.SI_REJILLA["T0"], strict=True):
        for f in C.FACTOR_CAL_REJILLA:
            out.append(Punto(f"SI{sa:g}-{s0:g}_cal{f:g}_rho{C.RHO_NOMINAL:g}", sa, s0, C.RHO_NOMINAL, f))
    sa, s0 = C.SI_NOMINAL["asimilado"], C.SI_NOMINAL["T0"]
    out.append(Punto(f"SI{sa:g}-{s0:g}_cal1_rho0.5", sa, s0, 0.5, 1.0))
    return out


NOMINAL = f"SI{C.SI_NOMINAL['asimilado']:g}-{C.SI_NOMINAL['T0']:g}_cal1_rho{C.RHO_NOMINAL:g}"


def preparar_lados(alt: pd.DataFrame, boy: pd.DataFrame) -> dict[str, S.LadoSim]:
    from .geometria import colocaciones_de_lado  # noqa: PLC0415

    return {n: S.LadoSim.desde(n, colocaciones_de_lado(alt, boy, n)) for n in LADOS_SIM}


def calibrar(lados: dict[str, S.LadoSim], punto: Punto) -> dict[str, float]:
    """r de cada lado para el SI del punto (números aleatorios fijos por lado)."""
    out = {}
    for n, L in lados.items():
        si = punto.si_t0 if CLASE_SI[n] == "T0" else punto.si_asimilado
        out[n] = S.calibrar_r(L, si, punto.rho, semillas.semilla("f05", "calibrar-r", n, f"{si:g}", f"{punto.rho:g}"))
    return out


def _idx(ests):
    return np.array([S.IDX[e] for e in ests])


def _diferencia(m1: S.Medida, r1, m2: S.Medida, r2, ests) -> S.Prueba:
    i = _idx(ests)
    usar = m1.estimable[i] & m2.estimable[i]
    return S.max_t(m2.B[i] - m1.B[i], r2[:, i] - r1[:, i], usar)


def replica(
    lados: dict[str, S.LadoSim],
    r: dict[str, float],
    punto: Punto,
    i: int,
    Bboot: int = C.B_POTENCIA,
    inyeccion: dict[str, float] | None = None,
) -> dict:
    """Una réplica con sesgo nulo (o con ``inyeccion``: b por lado, para validar el desplazamiento)."""
    rng = semillas.generador("f05", "potencia", punto.nombre, str(i))
    claves = set()
    for L in lados.values():
        claves |= set(S._claves(L, lado_boya=True))
    verdad = S.sortear_errores(claves, punto.factor_cal, rng)
    boot = S.sortear_errores(claves, punto.factor_cal, rng, size=Bboot)
    med, reps, reps_lado = {}, {}, {}
    for n, L in lados.items():
        m, o = S.par(L, r[n], punto.rho, rng)
        if inyeccion and n in inyeccion:
            o = o * np.exp(-inyeccion[n])
        o = S.aplicar_calibracion(L, o, {k: v for k, v in verdad.items() if not k.startswith("lado_")})
        sig = S.sigma_ruido(L, o, punto.factor_cal)
        o_obs = o + sig * rng.standard_normal(L.n)
        med[n] = S.medir(L, o_obs, sig, Bboot, rng, m=m)
        reps[n] = S.con_calibracion(L, med[n], boot, lado_boya=False)
        reps_lado[n] = S.con_calibracion(L, med[n], boot, lado_boya=True)
    out = {
        "V1a": _diferencia(med["T0"], reps["T0"], med["A"], reps["A"], ESTS_ATRIB),
        "V1b": _diferencia(med["T0b"], reps["T0b"], med["T1b"], reps["T1b"], ESTS_ATRIB),
        "V1b_lado": _diferencia(med["T0b"], reps_lado["T0b"], med["T1b"], reps_lado["T1b"], ESTS_ATRIB),
        "V2a": _diferencia(med["A1"], reps["A1"], med["A2"], reps["A2"], ESTS_ATRIB),
    }
    mA, rA = med["A"], reps["A"]
    i3 = _idx(ESTS_V3A)
    out["V3a"] = S.max_t(mA.B[i3], rA[:, i3], mA.estimable[i3])
    alto = "P99" if mA.estimable[S.IDX["P99"]] else "max"
    d = mA.B[S.IDX[alto]] - mA.B[S.IDX["P50"]]
    dr = rA[:, S.IDX[alto]] - rA[:, S.IDX["P50"]]
    out["V3b"] = S.max_t(np.array([d]), dr[:, None], np.array([bool(mA.estimable[S.IDX[alto]])]))
    # B_cola (inversa de la varianza bootstrap) y B_alta (§3.4)
    ok = mA.estimable[i3]
    var = np.var(rA[:, i3], axis=0, ddof=1)
    w = np.where(ok & (var > 0), 1.0 / var, 0.0)
    b_cola = float(np.sum(w * mA.B[i3]) / w.sum()) if w.sum() > 0 else np.nan
    b_alta = next((float(mA.B[S.IDX[e]]) for e in ("max", "P99", "P90") if mA.estimable[S.IDX[e]]), np.nan)
    out["subida"] = {
        "B_cola": b_cola,
        "B_alta": b_alta,
        "B_P90": float(mA.B[S.IDX["P90"]]),
        "B_P99": float(mA.B[S.IDX["P99"]]),
    }
    out["estimable"] = {n: med[n].estimable.tolist() for n in lados}
    return out


# --- Agregación -----------------------------------------------------------------------------------------


FACTORES: dict[str, float] = {}  # factor del valor crítico por prueba si el error de tipo I se recalibró (§5)


def potencia(reps: list[dict], prueba: str, desplazamiento: float, alfa: float) -> float:
    f = FACTORES.get(prueba, 1.0)
    return float(np.mean([S.rechaza(rp[prueba], desplazamiento, alfa, f) for rp in reps]))


def potencia_conjunta_v1(reps: list[dict], ef_a: float, ef_b: float, mult: float, alfa: float = C.ALFA_FUENTE) -> float:
    fa, fb = FACTORES.get("V1a", 1.0), FACTORES.get("V1b", 1.0)
    return float(
        np.mean(
            [S.rechaza(rp["V1a"], mult * ef_a, alfa, fa) or S.rechaza(rp["V1b"], mult * ef_b, alfa, fb) for rp in reps]
        )
    )


def efecto_minimo(reps: list[dict], prueba: str, alfa: float, efecto: float, objetivo: float = 0.8) -> float:
    """Menor |b| (en una rejilla fina, los dos signos) con potencia ≥ ``objetivo``; NaN si no se alcanza hasta 6×."""
    for b in np.arange(0.0, 6.0 * efecto + 1e-12, efecto / 50):
        if min(potencia(reps, prueba, b, alfa), potencia(reps, prueba, -b, alfa)) >= objetivo:
            return float(b)
    return float("nan")


def tabla_potencia(reps: list[dict], punto: Punto, efectos: dict[str, float]) -> pd.DataFrame:
    """Una fila por prueba y múltiplo del efecto (con signo), incluido 0 (error de tipo I)."""
    filas = []
    alfas = {
        "V1a": C.ALFA_FUENTE,
        "V1b": C.ALFA_FUENTE,
        "V1b_lado": C.ALFA_FUENTE,
        "V2a": C.ALFA_FUENTE,
        "V3a": C.ALFA_FAMILIA,
        "V3b": C.ALFA_FAMILIA,
    }
    mults = [0.0, *[s * m for m in C.MULTIPLOS_EFECTO for s in (1, -1)]]
    for prueba, alfa in alfas.items():
        ef = efectos[prueba]
        for mu in mults:
            filas.append(
                {
                    "punto": punto.nombre,
                    "prueba": prueba,
                    "alfa": alfa,
                    "multiplo": mu,
                    "efecto": ef,
                    "b": mu * ef,
                    "potencia": potencia(reps, prueba, mu * ef, alfa),
                    "replicas": len(reps),
                }
            )
    for mu in mults:
        filas.append(
            {
                "punto": punto.nombre,
                "prueba": "V1 conjunta",
                "alfa": C.ALFA_FAMILIA,
                "multiplo": mu,
                "efecto": np.nan,
                "b": np.nan,
                "potencia": potencia_conjunta_v1(reps, efectos["V1a"], efectos["V1b"], mu),
                "replicas": len(reps),
            }
        )
    return pd.DataFrame(filas)


def subida_esperada(reps: list[dict], pot_v3a: float, pot_v3b: float) -> dict:
    """Subida media del factor de diseño con sesgo nulo, con la regla de §6.2 y la ADR-0024 (punto 2)."""
    factores = []
    for rp in reps:
        bc, ba = rp["subida"]["B_cola"], rp["subida"]["B_alta"]
        a = (
            "est"
            if S.rechaza(rp["V3a"], 0.0, C.ALFA_FAMILIA, FACTORES.get("V3a", 1.0))
            else ("desc" if pot_v3a >= 0.8 else "nc")
        )
        f = (
            "est"
            if S.rechaza(rp["V3b"], 0.0, C.ALFA_FAMILIA, FACTORES.get("V3b", 1.0))
            else ("desc" if pot_v3b >= 0.8 else "nc")
        )
        corr_forma = (
            ba if f == "est" else bc
        )  # con forma establecida, la cola alta aproxima la corrección por cuantiles
        variantes = {
            ("est", "est"): [ba],
            ("est", "desc"): [bc],
            ("est", "nc"): [bc, ba],
            ("nc", "est"): [0.0, corr_forma],
            ("nc", "desc"): [0.0, bc],
            ("nc", "nc"): [0.0, bc, ba],
        }.get((a, f), [0.0])  # sesgo descartado: sin corrección
        v = [x for x in variantes if np.isfinite(x)]
        factores.append(float(np.exp(-min(v))) if v else 1.0)
    f = np.asarray(factores)
    return {
        "subida_media": float(f.mean() - 1),
        "ee_mc": float(f.std(ddof=1) / np.sqrt(f.size)),
        "replicas": int(f.size),
        "supera_2_por_ciento": bool(f.mean() - 1 > 0.02),
    }


def nulidad_generador(reps: list[dict]) -> dict:
    """§8: con b = 0, la media de B(P90) y B(P99) descontados está dentro del error de Monte Carlo de 0."""
    out = {}
    for k in ("B_P90", "B_P99", "B_cola", "B_alta"):
        x = np.array([rp["subida"][k] for rp in reps], float)
        x = x[np.isfinite(x)]
        out[k] = {"media": float(x.mean()), "ee_mc": float(x.std(ddof=1) / np.sqrt(x.size)), "n": int(x.size)}
    return out


def recalibrar(reps: list[dict], prueba: str, alfa: float) -> dict:
    """Error de tipo I con b = 0. Fuera de [0.6·α, 1.4·α] ([3 %, 7 %] con α = 0.05): factor del crítico que da α."""
    tasa = float(np.mean([S.rechaza(rp[prueba], 0.0, alfa) for rp in reps]))
    lo, hi = 0.6 * alfa, 1.4 * alfa
    out = {"prueba": prueba, "alfa": alfa, "tasa": tasa, "rango": [lo, hi], "factor": 1.0, "recalibrado": False}
    if lo <= tasa <= hi:
        return out
    a, b = 0.3, 3.0
    for _ in range(40):
        c = (a + b) / 2
        t = np.mean([S.rechaza(rp[prueba], 0.0, alfa, c) for rp in reps])
        a, b = (c, b) if t > alfa else (a, c)
    out |= {"factor": (a + b) / 2, "recalibrado": True}
    return out


def tasa_familia_v1(reps: list[dict]) -> float:
    """Error de tipo I de la familia V1: alguna fuente rechaza con α = 0.025 (Bonferroni, P12)."""
    return potencia_conjunta_v1(reps, 0.0, 0.0, 0.0)
