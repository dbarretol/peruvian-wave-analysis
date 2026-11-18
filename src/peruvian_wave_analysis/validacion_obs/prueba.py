"""Prueba observada de la F5 (contrato v6, §3.4, §3.5, §3.7, §4 y §6.1). Reutiliza el motor de la etapa 2.

Por lado: el modelo en la colocación (m), la observación (o) y el nodo en los mismos pasos (§3.7) comparten el
bootstrap por bloques de un mes, con B = 9 999. SIMEX (λ ∈ {0.5, 1, 1.5, 2}, extrapolación cuadrática) se calcula para
cada variante del ruido (principal con el σ medido, σ × 0.5, σ × 2 o 4 % y 8 % en las boyas, y con la varianza de
pequeña escala si la función de estructura supera su umbral), y se suma a las réplicas como un desplazamiento fijo,
igual que en la potencia. Los errores de calibración se sortean una vez por réplica y tramo, comunes a todos los
lados (se cancelan donde se comparten), y entran linealizados; la calibración × 2 los duplica.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from ..comun import semillas
from ..validacion import config as CV
from ..validacion import escalon as E
from ..validacion import potencia as P
from ..validacion import simulacion as S

ESTS = E.ESTADISTICOS  # P50, P90, P99, P99.9, max


@dataclass
class LadoObs:
    nombre: str
    L: S.LadoSim
    o: np.ndarray
    df: pd.DataFrame
    punt: np.ndarray  # (3, 5): modelo, observación y nodo, brutos
    reps: np.ndarray  # (B, 3, 5)
    sigmas: dict[str, np.ndarray] = field(default_factory=dict)  # σ por variante de ruido, alineado con df
    med: dict[str, S.Medida] = field(default_factory=dict)  # por variante de ruido

    @property
    def bruto(self) -> np.ndarray:
        return np.log(self.punt[0]) - np.log(self.punt[1])


def ordenar(df: pd.DataFrame) -> pd.DataFrame:
    """El orden de ``LadoSim.desde`` (por tiempo, estable), para alinear o y σ con el lado."""
    return df.sort_values("tiempo", kind="stable").reset_index(drop=True)


def medir_lado(nombre: str, df: pd.DataFrame, sigmas: dict[str, np.ndarray], Bboot: int) -> LadoObs:
    """``df`` ya ordenado (``ordenar``), con ``hs_obs`` y ``hs_nodo``; ``sigmas``: σ por variante, alineado con df."""
    L = S.LadoSim.desde(nombre, df)
    o = df["hs_obs"].to_numpy(float)
    L.boot = E.LadoBoot.desde(L.t, L.m, o, df["hs_nodo"].to_numpy(float))
    punt = E.puntual(L.boot)
    reps = E.replicas(L.boot, Bboot, semillas.generador("f05", "observada", "bootstrap", nombre))
    out = LadoObs(nombre, L, o, df, punt, reps, dict(sigmas))
    for var, sig in sigmas.items():
        Qs = S.simex(L, o, np.nan_to_num(sig), semillas.generador("f05", "observada", "simex", nombre, var))
        desplaz = np.log(punt[1]) - np.log(Qs)
        B = np.log(punt[0]) - np.log(Qs)
        Bb = np.log(reps[:, 0, :]) - np.log(reps[:, 1, :]) + desplaz[None, :]
        out.med[var] = S.Medida(B, Bb, S.estimables(L, o, Qs), Qs)
    return out


def errores_calibracion(lados: dict[str, LadoObs], Bboot: int) -> dict:
    """Un error (a, b, e) por réplica y clave (tramo, época, año de boya, término por lado), común a todos los lados."""
    claves = set()
    for lo in lados.values():
        claves |= set(S._claves(lo.L, lado_boya=True))
    return S.sortear_errores(claves, 1.0, semillas.generador("f05", "observada", "calibracion"), size=Bboot)


def escalar(errores: dict, factor: float) -> dict:
    return {k: tuple(factor * np.asarray(x) for x in v) for k, v in errores.items()}


def con_cal(lo: LadoObs, var: str, errores: dict, lado_boya: bool = False) -> np.ndarray:
    return S.con_calibracion(lo.L, lo.med[var], errores, lado_boya=lado_boya)


def ic(x: np.ndarray, nivel: float = 0.95) -> list[float]:
    return list(E.ic(np.asarray(x, float), nivel))


def _por_trozos(lado: E.LadoBoot, cuentas: np.ndarray, trozo: int = 500) -> np.ndarray:
    """``escalon._estadisticos`` por trozos de réplicas (memoria acotada). Bloques sin datos: sin aviso."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.concatenate([E._estadisticos(lado, cuentas[a : a + trozo]) for a in range(0, len(cuentas), trozo)])


# --- Pruebas ---------------------------------------------------------------------------------------------


def resultado(pr: S.Prueba, reps: np.ndarray, ests, alfa: float, factor: float) -> dict:
    """Estadístico, EE, IC, p, rechazo (con el factor del crítico si se recalibró) y el estadístico que decide."""
    ok = np.isfinite(pr.est) & (pr.ee > 0)
    t = np.where(ok, np.abs(pr.est) / np.where(ok, pr.ee, 1.0), np.nan)
    j = int(np.nanargmax(t)) if ok.any() else None
    crit = pr.critico.get(alfa, np.nan)
    rech = bool(ok.any() and np.isfinite(crit) and np.nanmax(t) > factor * crit)
    return {
        "estadisticos": list(ests),
        "estimado": {e: (float(pr.est[k]) if ok[k] else None) for k, e in enumerate(ests)},
        "ee": {e: (float(pr.ee[k]) if ok[k] else None) for k, e in enumerate(ests)},
        "ic95": {e: (ic(reps[:, k]) if ok[k] else None) for k, e in enumerate(ests)},
        "t_max": float(np.nanmax(t)) if ok.any() else None,
        "critico": float(crit) if np.isfinite(crit) else None,
        "factor_critico": factor,
        "alfa": alfa,
        "p": pr.p if np.isfinite(pr.p) else None,
        "rechaza": rech,
        "decide": ests[j] if j is not None else None,
        "signo": (int(np.sign(pr.est[j])) if j is not None else 0),
        "estimables": int(ok.sum()),
    }


def diferencia(
    l1: LadoObs, l2: LadoObs, var: str, errores: dict, lado_boya: bool, alfa: float, factor: float, desplazamiento=0.0
) -> dict:
    """Δ(q) = B_lado2(q) − B_lado1(q), max-T sobre los estimables de los dos lados (§4.1)."""
    r1, r2 = con_cal(l1, var, errores, lado_boya), con_cal(l2, var, errores, lado_boya)
    pr = P._diferencia(l1.med[var], r1, l2.med[var], r2, P.ESTS_ATRIB, desplazamiento)
    i = P._idx(P.ESTS_ATRIB)
    return resultado(pr, r2[:, i] - r1[:, i], P.ESTS_ATRIB, alfa, factor)


def v3(lA: LadoObs, var: str, errores: dict, con_max: bool, factores: dict[str, float]) -> dict:
    """V3a, V3b, B_cola y B_alta del lado A con sus IC (§4.3, §3.4; con o sin máximos según §20.3)."""
    suf = "" if con_max else "_sinmax"
    mA, rA = lA.med[var], con_cal(lA, var, errores)
    v3a, v3b, sub = P._v3(mA, rA, con_max)
    ests = P.ESTS_V3A if con_max else ("P90", "P99")
    i3 = P._idx(ests)
    ok = mA.estimable[i3]
    w = np.where(ok & (np.var(rA[:, i3], axis=0, ddof=1) > 0), 1.0 / np.var(rA[:, i3], axis=0, ddof=1), 0.0)
    cola_reps = rA[:, i3] @ w / w.sum() if w.sum() > 0 else np.full(len(rA), np.nan)
    orden = ("max", "P99", "P90") if con_max else ("P99", "P90")
    alta = next((e for e in orden if mA.estimable[S.IDX[e]]), None)
    reserva = "max" if con_max else "P90"
    alto_b = "P99" if mA.estimable[S.IDX["P99"]] else reserva
    dr = rA[:, S.IDX[alto_b]] - rA[:, S.IDX["P50"]]
    return {
        "V3a": resultado(v3a, rA[:, i3], ests, CV.ALFA_FAMILIA, factores.get(f"V3a{suf}", 1.0)),
        "V3b": resultado(v3b, dr[:, None], (f"{alto_b}−P50",), CV.ALFA_FAMILIA, factores.get(f"V3b{suf}", 1.0)),
        "B_cola": {
            "estimado": sub["B_cola"] if np.isfinite(sub["B_cola"]) else None,
            "ic95": ic(cola_reps) if w.sum() > 0 else None,
            "pesos": {e: float(w[k] / w.sum()) if w.sum() > 0 else 0.0 for k, e in enumerate(ests)},
        },
        "B_alta": {
            "estimado": sub["B_alta"] if np.isfinite(sub["B_alta"]) else None,
            "estadistico": alta,
            "ic95": ic(rA[:, S.IDX[alta]]) if alta else None,
        },
        "B_q": {e: (float(mA.B[S.IDX[e]]) if mA.estimable[S.IDX[e]] else None) for e in ESTS[:4]},
    }


# --- SI y ρ observados (§5, línea de «después de leer valores») ------------------------------------------


def si_observado(lo: LadoObs, sigma: np.ndarray) -> float:
    """SI de o frente a m, sin el sesgo medio y sin el ruido de la observación: el análogo del SI del generador."""
    w = lo.L.w / lo.L.w.sum()
    m, o = lo.L.m, lo.o
    c = np.sum(w * m) / np.sum(w * o)
    var = np.sum(w * (c * o - m) ** 2) - np.sum(w * (c * np.nan_to_num(sigma)) ** 2)
    return float(np.sqrt(max(var, 0.0)) / np.sum(w * m))


def rho_observado(lo: LadoObs) -> float:
    """Fracción común a la unidad (pasada o día) del residuo en puntuaciones normales, por ANOVA de un factor."""
    L = lo.L
    xs = L.m[L._orden_m]
    pw = L.w[L._orden_m] / L.w.sum()
    u_o = np.clip(np.interp(lo.o, xs, L._F_m - pw / 2), 1e-6, 1 - 1e-6)
    r = st.norm.ppf(u_o) - st.norm.ppf(L.u_modelo())
    return icc(r, L.unidad)


def icc(x: np.ndarray, grupo: np.ndarray) -> float:
    _, g = np.unique(grupo, return_inverse=True)
    n_g = np.bincount(g)
    keep = n_g[g] >= 2
    if keep.sum() < 3:
        return float("nan")
    x, g = x[keep], np.unique(g[keep], return_inverse=True)[1]
    n_g = np.bincount(g).astype(float)
    G, N = n_g.size, x.size
    if G < 2:
        return float("nan")
    media_g = np.bincount(g, weights=x) / n_g
    msb = np.sum(n_g * (media_g - x.mean()) ** 2) / (G - 1)
    msw = np.sum((x - media_g[g]) ** 2) / (N - G)
    n0 = (N - np.sum(n_g**2) / N) / (G - 1)
    return float(np.clip((msb - msw) / (msb + (n0 - 1) * msw), 0.0, 1.0))


# --- Comprobaciones -----------------------------------------------------------------------------------------


def coherencia_t0(l_geo: LadoObs, l_otro: LadoObs, errores: dict, ests) -> dict:
    """§4.1: B de GEOSAT frente a ERS-1 y TOPEX antes de 1993; marca si los IC del 95 % no se solapan."""
    r1, r2 = con_cal(l_geo, "principal", errores), con_cal(l_otro, "principal", errores)
    filas, incoherente = {}, False
    for e in ests:
        k = S.IDX[e]
        ok = bool(l_geo.med["principal"].estimable[k] and l_otro.med["principal"].estimable[k])
        a, b = ic(r1[:, k]), ic(r2[:, k])
        solapan = bool(a[0] <= b[1] and b[0] <= a[1]) if ok else None
        filas[e] = {
            "B_geosat": float(l_geo.med["principal"].B[k]),
            "ic_geosat": a,
            "B_ers1_topex": float(l_otro.med["principal"].B[k]),
            "ic_ers1_topex": b,
            "estimable_en_los_dos": ok,
            "solapan": solapan,
        }
        incoherente |= solapan is False
    return {"estadisticos": filas, "incoherencia": bool(incoherente)}


def absorcion_v2(l1: LadoObs, l2: LadoObs, errores: dict, tol: float) -> dict:
    """§6.1, condición (iii): IC del 95 % de (B_A2 − B_A1) − S_mod,nodo en los mismos pasos, en el P90 (§19)."""
    k = S.IDX["P90"]
    r1, r2 = con_cal(l1, "principal", errores), con_cal(l2, "principal", errores)
    s_nodo = np.log(l2.reps[:, 2, k]) - np.log(l1.reps[:, 2, k])
    d = (r2[:, k] - r1[:, k]) - s_nodo
    est = (l2.med["principal"].B[k] - l1.med["principal"].B[k]) - (np.log(l2.punt[2, k]) - np.log(l1.punt[2, k]))
    lo, hi = ic(d)
    return {"estimado": float(est), "ic95": [lo, hi], "tolerancia": tol, "absorbe": bool(-tol <= lo and hi <= tol)}


def gradiente(lA: LadoObs, sig: np.ndarray, n_bandas: int, Bboot: int, x0_km: float, tol: float) -> dict:
    """§3.7: B(P90) del lado A frente a la distancia a la costa, extrapolado al nodo.

    Bandas por cuartiles de la distancia de las colocaciones del lado (geometría, fijada antes de leer valores).
    B(P90) de cada banda con su SIMEX; recta ponderada por la inversa de la varianza bootstrap de cada banda; las
    réplicas remuestrean los mismos meses en todas las bandas y en el lado entero, así que la diferencia entre B a
    23 km y B en la región tiene su IC. Válido si ese IC cae dentro de ±log(1.02).
    """
    df, L = lA.df, lA.L  # df ya ordenado por tiempo, alineado con el lado
    dist = df["dist_costa_km"].to_numpy(float)
    cortes = np.quantile(dist, np.linspace(0, 1, n_bandas + 1)[1:-1])
    banda = np.digitize(dist, cortes)
    k90 = S.IDX["P90"]
    rng = semillas.generador("f05", "observada", "gradiente")
    cuentas = rng.multinomial(L.boot.K, np.full(L.boot.K, 1.0 / L.boot.K), size=Bboot)
    rr = _por_trozos(L.boot, cuentas)
    B_region = float(lA.med["principal"].B[k90])
    reg_rep = (
        np.log(rr[:, 0, k90])
        - np.log(rr[:, 1, k90])
        + (np.log(lA.punt[1, k90]) - np.log(lA.med["principal"].Qobs[k90]))
    )
    xs, bs, vs, reps_b, filas = [], [], [], [], []
    for b in range(n_bandas):
        s = banda == b
        sub = df[s].reset_index(drop=True)
        if len(sub) < 10:
            continue
        Lb = S.LadoSim.desde(f"A_banda{b}", sub)
        ob = sub["hs_obs"].to_numpy(float)
        # mismos bloques (meses) que el lado entero: las réplicas remuestrean los mismos meses en todas las bandas
        Lb.boot = E.LadoBoot(np.vstack([Lb.m, ob]), Lb.boot.mes, L.boot.bloque[s], L.boot.K)
        punt = _por_trozos(Lb.boot, np.ones((1, L.boot.K)))[0]
        Qs = S.simex(Lb, ob, np.nan_to_num(sig[s]), semillas.generador("f05", "observada", "simex-banda", str(b)))
        est = bool(S.estimables(Lb, ob, Qs)[k90])
        rep = _por_trozos(Lb.boot, cuentas)
        Bb = np.log(rep[:, 0, k90]) - np.log(rep[:, 1, k90]) + (np.log(punt[1, k90]) - np.log(Qs[k90]))
        Bp = float(np.log(punt[0, k90]) - np.log(Qs[k90]))
        filas.append(
            {"banda": b, "dist_media_km": float(dist[s].mean()), "n": int(s.sum()), "B_P90": Bp, "estimable": est}
        )
        if est and np.isfinite(Bb).sum() > 10:
            xs.append(float(dist[s].mean()))
            bs.append(Bp)
            vs.append(float(np.nanvar(Bb, ddof=1)))
            reps_b.append(Bb)
    if len(xs) < 2:
        return {"bandas": filas, "valido": None, "estado": "no evaluable (menos de dos bandas estimables)"}
    x, w = np.array(xs), 1.0 / np.array(vs)

    def recta(y):
        X = np.vstack([np.ones_like(x), x]).T
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        return beta[0] + beta[1] * x0_km, beta[1]

    b23, pend = recta(np.array(bs))
    R = np.vstack(reps_b).T  # (B, bandas)
    ok = np.all(np.isfinite(R), axis=1) & np.isfinite(reg_rep)
    d_rep = np.array([recta(R[i])[0] for i in np.flatnonzero(ok)]) - reg_rep[ok]
    lo, hi = ic(d_rep)
    return {
        "bandas": filas,
        "pendiente_por_km": float(pend),
        "B_P90_23km": float(b23),
        "B_P90_region": B_region,
        "diferencia": float(b23 - B_region),
        "ic95": [lo, hi],
        "tolerancia": tol,
        "valido": bool(-tol <= lo and hi <= tol),
        "estado": "válido" if (-tol <= lo and hi <= tol) else "dudoso",
    }
