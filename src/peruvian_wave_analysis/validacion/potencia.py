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
from scipy import stats as st

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


def _diferencia(m1: S.Medida, r1, m2: S.Medida, r2, ests, desplazamiento: float = 0.0) -> S.Prueba:
    i = _idx(ests)
    usar = m1.estimable[i] & m2.estimable[i]
    return S.max_t(m2.B[i] - m1.B[i] + desplazamiento, r2[:, i] - r1[:, i] + desplazamiento, usar)


def _inyectar(L: S.LadoSim, o: np.ndarray, iny) -> np.ndarray:
    """Sesgo inyectado en o*: ``b`` (constante) o ``(b, "nivel")``, que crece con el nivel: exp(−b·g(u)), con
    g = Φ⁻¹(u)/Φ⁻¹(0.99) y u el nivel de o* en la marginal del lado; así B(P99) − B(P50) ≈ b (V3b, §20.2)."""
    b, forma = (iny, "constante") if np.isscalar(iny) else iny
    if forma == "constante":
        return o * np.exp(-b)
    xs = L.m[L._orden_m]
    u = np.clip(np.interp(o, xs, L._F_m), 1e-6, 1 - 1e-6)
    g = st.norm.ppf(u) / st.norm.ppf(0.99)
    return o * np.exp(-b * g)


def _v3(mA: S.Medida, rA: np.ndarray, con_max: bool) -> tuple[S.Prueba, S.Prueba, dict]:
    """V3a, V3b, B_cola y B_alta del lado A, con los máximos o sin ellos (regla de representatividad, §20.3)."""
    ests = ESTS_V3A if con_max else ("P90", "P99")
    i3 = _idx(ests)
    v3a = S.max_t(mA.B[i3], rA[:, i3], mA.estimable[i3])
    reserva = "max" if con_max else "P90"
    alto = "P99" if mA.estimable[S.IDX["P99"]] else reserva
    d = mA.B[S.IDX[alto]] - mA.B[S.IDX["P50"]]
    dr = rA[:, S.IDX[alto]] - rA[:, S.IDX["P50"]]
    usar = bool(mA.estimable[S.IDX[alto]])
    v3b = S.max_t(np.array([d]), dr[:, None], np.array([usar]))
    ok = mA.estimable[i3]
    var = np.var(rA[:, i3], axis=0, ddof=1)
    w = np.where(ok & (var > 0), 1.0 / var, 0.0)
    b_cola = float(np.sum(w * mA.B[i3]) / w.sum()) if w.sum() > 0 else np.nan
    orden = ("max", "P99", "P90") if con_max else ("P99", "P90")
    b_alta = next((float(mA.B[S.IDX[e]]) for e in orden if mA.estimable[S.IDX[e]]), np.nan)
    return v3a, v3b, {"B_cola": b_cola, "B_alta": b_alta}


def replica(
    lados: dict[str, S.LadoSim],
    r: dict[str, float],
    punto: Punto,
    i: int,
    Bboot: int = C.B_POTENCIA,
    inyeccion: dict | None = None,
) -> dict:
    """Una réplica con sesgo nulo (o con ``inyeccion``: por lado, b o (b, "nivel"), para validar el desplazamiento)."""
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
            o = _inyectar(L, o, inyeccion[n])
        o = S.aplicar_calibracion(L, o, {k: v for k, v in verdad.items() if not k.startswith("lado_")})
        sig = S.sigma_ruido(L, o, punto.factor_cal)
        o_obs = o + sig * rng.standard_normal(L.n)
        med[n] = S.medir(L, o_obs, sig, Bboot, rng, m=m)
        reps[n] = S.con_calibracion(L, med[n], boot, lado_boya=False)
        reps_lado[n] = S.con_calibracion(L, med[n], boot, lado_boya=True)
    # Término por lado de la boya también en la «verdad» (S-25). El ruido de la boya es relativo, así que un factor
    # (1 + e) en un lado desplaza exactamente todos sus B en −log(1 + e).
    lado_verdad = float(np.log1p(verdad["lado_T0b"][2]) - np.log1p(verdad["lado_T1b"][2]))
    out = {
        "V1a": _diferencia(med["T0"], reps["T0"], med["A"], reps["A"], ESTS_ATRIB),
        "V1b": _diferencia(med["T0b"], reps["T0b"], med["T1b"], reps["T1b"], ESTS_ATRIB),
        "V1b_lado": _diferencia(med["T0b"], reps_lado["T0b"], med["T1b"], reps_lado["T1b"], ESTS_ATRIB, lado_verdad),
        "V2a": _diferencia(med["A1"], reps["A1"], med["A2"], reps["A2"], ESTS_ATRIB),
    }
    for con_max, suf in ((True, ""), (False, "_sinmax")):
        v3a, v3b, sub = _v3(med["A"], reps["A"], con_max)
        out[f"V3a{suf}"], out[f"V3b{suf}"], out[f"subida{suf}"] = v3a, v3b, sub
    out["B_lados"] = {n: med[n].B.tolist() for n in lados}
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
        "V3a_sinmax": C.ALFA_FAMILIA,
        "V3b_sinmax": C.ALFA_FAMILIA,
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


def subida_esperada(reps: list[dict], pot_v3a: float, pot_v3b: float, suf: str = "", v3b_validado: bool = True) -> dict:
    """Subida media del factor de diseño con sesgo nulo, con la regla de §6.2 y la ADR-0024 (punto 2).

    ``suf``: "" con los máximos, "_sinmax" sin ellos (§20.3). Si la potencia por desplazamiento de V3b no quedó
    validada, V3b no puede salir «descartada» (S-20).
    """
    fa, fb = FACTORES.get(f"V3a{suf}", 1.0), FACTORES.get(f"V3b{suf}", 1.0)
    factores = []
    for rp in reps:
        bc, ba = rp[f"subida{suf}"]["B_cola"], rp[f"subida{suf}"]["B_alta"]
        a = "est" if S.rechaza(rp[f"V3a{suf}"], 0.0, C.ALFA_FAMILIA, fa) else ("desc" if pot_v3a >= 0.8 else "nc")
        desc_b = pot_v3b >= 0.8 and v3b_validado
        f = "est" if S.rechaza(rp[f"V3b{suf}"], 0.0, C.ALFA_FAMILIA, fb) else ("desc" if desc_b else "nc")
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
    """§8 y §20.1 (S-24): con b = 0, media de B descontado, sin centrar, en todos los lados y estadísticos."""
    out = {}
    for lado in reps[0]["B_lados"]:
        B = np.array([rp["B_lados"][lado] for rp in reps], float)
        out[lado] = {}
        for e in ("P50", "P90", "P99", "P99.9", "max"):
            x = B[:, S.IDX[e]]
            x = x[np.isfinite(x)]
            ee = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
            out[lado][e] = {"media": float(x.mean()) if x.size else float("nan"), "ee_mc": ee, "n": int(x.size)}
    return out


def nulidad_cumple(nul: dict, ests=("P90", "P99", "max")) -> bool:
    """La media de cada estadístico de decisión está a ≤ 3 errores de Monte Carlo de 0 en todos los lados."""
    return all(abs(v[e]["media"]) <= 3 * v[e]["ee_mc"] for v in nul.values() for e in ests if v[e]["n"] > 1)


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


def comparar_desplazamiento(reps_iny: list[dict], reps0: list[dict], prueba: str, b: float, alfa: float) -> dict:
    """Potencia con inyección explícita frente a la de desplazamiento, con las mismas semillas (pares; S-21).

    Coincide si |diferencia| ≤ máx(3·EE pareado, 0.02) y además ≤ 0.05. EE pareado = √((d − δ²)/n), con d la fracción
    de réplicas discordantes y δ la diferencia (McNemar).
    """
    f = FACTORES.get(prueba, 1.0)
    ri = np.array([S.rechaza(rp[prueba], 0.0, alfa, f) for rp in reps_iny])
    rd = np.array([S.rechaza(rp[prueba], b, alfa, f) for rp in reps0[: len(reps_iny)]])
    n = ri.size
    delta = float(ri.mean() - rd.mean())
    d = float(np.mean(ri != rd))
    ee = float(np.sqrt(max(d - delta**2, 0.0) / n))
    return {
        "prueba": prueba,
        "b": b,
        "potencia_inyeccion": float(ri.mean()),
        "potencia_desplazamiento": float(rd.mean()),
        "diferencia": delta,
        "discordantes": d,
        "ee_pareado": ee,
        "coincide": bool(abs(delta) <= max(3 * ee, 0.02) and abs(delta) <= 0.05),
    }


def nulidad_pura(lados: dict[str, S.LadoSim], r: dict[str, float], rho: float, n: int, nombre: str) -> dict:
    """§20.5: el **generador solo** (m* y o* sin ruido ni calibración), sin centrar, en todos los lados.

    Es lo que §8 quería comprobar: que el generador no fabrica sesgo. Media de B por lado y estadístico.
    """
    out = {}
    for lado, L in lados.items():
        Bs = []
        for i in range(n):
            rng = semillas.generador("f05", "nulidad-pura", nombre, lado, str(i))
            m, o = S.par(L, r[lado], rho, rng)
            qm, qo = S.estadisticos_filas(L, np.vstack([m, o]))
            Bs.append(np.log(qm / qo))
        Bs = np.asarray(Bs)
        out[lado] = {}
        for e in ("P50", "P90", "P99", "P99.9", "max"):
            x = Bs[:, S.IDX[e]]
            x = x[np.isfinite(x)]
            out[lado][e] = {"media": float(x.mean()), "ee_mc": float(x.std(ddof=1) / np.sqrt(x.size)), "n": int(x.size)}
    return out


DIFERENCIAS = {"V1a": ("T0", "A"), "V1b": ("T0b", "T1b"), "V2a": ("A1", "A2")}


def sesgo_estimador(reps: list[dict], efectos: dict[str, float]) -> dict:
    """§20.5: sesgo del estimador completo (generador nulo, calibración, ruido y SIMEX) por lado y en las diferencias,
    solo con los estadísticos estimables (estimable en la mayoría de las réplicas). Marca «supera_cuarto» si
    |sesgo| > ¼ del efecto de la prueba que lo usa: V3 en el lado A; V1a, V1b y V2a en sus diferencias."""

    def media(lado):
        Bl = np.array([rp["B_lados"][lado] for rp in reps], float)
        est = np.array([rp["estimable"][lado] for rp in reps], bool).mean(axis=0) >= 0.5
        return Bl, est

    out = {"lados": {}, "diferencias": {}, "hallazgos": []}
    for lado in reps[0]["B_lados"]:
        Bl, est = media(lado)
        out["lados"][lado] = {}
        for e in ("P50", "P90", "P99", "max"):
            j = S.IDX[e]
            x = Bl[:, j][np.isfinite(Bl[:, j])]
            fila = {
                "media": float(x.mean()),
                "ee_mc": float(x.std(ddof=1) / np.sqrt(x.size)),
                "estimable": bool(est[j]),
            }
            if lado == "A" and e in ("P90", "P99", "max") and est[j]:
                fila["supera_cuarto"] = bool(abs(fila["media"]) > efectos["V3a"] / 4)
                if fila["supera_cuarto"]:
                    out["hallazgos"].append(f"Sesgo del estimador en B({e}) del lado A: {fila['media']:+.4f}")
            out["lados"][lado][e] = fila
    for prueba, (l1, l2) in DIFERENCIAS.items():
        B1, e1 = media(l1)
        B2, e2 = media(l2)
        out["diferencias"][prueba] = {}
        for e in ("P50", "P90", "P99", "max"):
            j = S.IDX[e]
            d = B2[:, j] - B1[:, j]
            d = d[np.isfinite(d)]
            ok = bool(e1[j] and e2[j])
            fila = {"media": float(d.mean()), "ee_mc": float(d.std(ddof=1) / np.sqrt(d.size)), "estimable": ok}
            if ok:
                fila["supera_cuarto"] = bool(abs(fila["media"]) > efectos[prueba] / 4)
                if fila["supera_cuarto"]:
                    out["hallazgos"].append(f"Sesgo del estimador en Δ de {prueba} ({e}): {fila['media']:+.4f}")
            out["diferencias"][prueba][e] = fila
    return out
