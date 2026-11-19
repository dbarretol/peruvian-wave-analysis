"""Dependencia temporal en el sitio (contrato de la F6, §3.5): τ, θ(u), T_(C), rachas, corte y extremograma.

Todo en E0 sin corregir (el IET es común a las ramas, B-S-03) y con las funciones por bloques de
``extremos.dependencia``: el punto es la función aplicada a los años en su orden, y cada réplica del bootstrap la
misma función aplicada a los años remuestreados (S-02).

- **Umbrales:** los cuantiles de Hs con peso mensual igual (la misma función que u_tent, §3.4). Se fijan con la
  serie completa y **no se recalculan en las réplicas**: el IC es condicional al umbral (decisión de implementación).
- **Estimación principal:** dentro del invierno austral (mayo–octubre). Sensibilidad: todo el año con umbral mensual.
  Descriptivos: todo el año con u fijo, junio–septiembre, huecos de verano, antes y después de 1993-02.
- **IC 95 %:** B = 999 años remuestreados con reemplazo, por percentiles (``bootstrap.ic_percentil``).
- **τ_int:** si la ACF no cruza cero dentro de ``KMAX_TAU`` desfases, τ_int no está definido; se publica así, con la
  ACF, y no se inventa un valor (G6 lo marca).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..comun import semillas
from ..extremos import bootstrap as BS
from ..extremos import dependencia as D
from ..homogeneidad.cuantiles import cuantiles_mensuales
from ..validacion import reanudable

INVIERNO = (5, 6, 7, 8, 9, 10)
JUN_SEP = (6, 7, 8, 9)
VERANO = (11, 12, 1, 2, 3, 4)
CUANTILES_THETA = (0.95, 0.96, 0.97, 0.98, 0.99, 0.995)
Q_PRINCIPAL = 0.99
KMAX_EXTREMOGRAMA = 120  # 15 días en pasos de 3 h
KMAX_TAU = 720  # 90 días, como en la etapa 1
FACTOR = 1.5
FACTORES_SENSIBILIDAD = (1.25, 2.0)
B = 999
TROZO = 50
INICIO_ASIMILACION = pd.Timestamp("1993-02-01")
VARIACION_THETA = 0.2

NOMBRES = (
    "tau_e_h", "tau_int_h",
    *(f"theta_q{q:g}" for q in CUANTILES_THETA),
    *(f"T_C_h_q{q:g}" for q in CUANTILES_THETA),
    "corte_h", "escala_extremograma_h",
    *(f"escala_extremograma_{f:g}_h" for f in FACTORES_SENSIBILIDAD),
)  # fmt: skip
CON_IC = ("tau_e_h", "tau_int_h", *(n for n in NOMBRES if n.startswith(("theta_", "T_C_"))), "corte_h",
          "escala_extremograma_h")  # fmt: skip


def umbrales(hs: pd.Series, qs: Sequence[float] = CUANTILES_THETA) -> dict[float, float]:
    """Cuantiles de Hs con peso mensual igual (como u_tent)."""
    return {float(q): float(cuantiles_mensuales(hs, q)[0]) for q in qs}


def umbral_mensual(hs: pd.Series, q: float = Q_PRINCIPAL) -> np.ndarray:
    """Vector de 13 (índice = mes) con el cuantil q de cada mes calendario; el índice 0 no se usa."""
    s = hs.dropna()
    u = np.full(13, np.inf)
    for m, g in s.groupby(s.index.month):
        u[int(m)] = float(np.quantile(g.to_numpy(float), q))
    return u


def vector(bloques: Sequence[D.Bloque], u: dict[float, float]) -> np.ndarray:
    """Las escalas de ``NOMBRES`` para una lista de bloques (punto o réplica)."""
    t = D.tau_bloques(bloques, kmax=KMAX_TAU)
    th, tc = [], []
    for q in CUANTILES_THETA:
        dc = D.declustering_bloques(bloques, u[q], INVIERNO)
        th.append(dc.theta)
        tc.append(dc.T_C_horas)
    u99 = u[Q_PRINCIPAL]
    corte = D.corte_natural(D.huecos_bloques(bloques, u99, INVIERNO) * D.PASO_H)
    ex = D.extremograma_bloques(bloques, u99, INVIERNO, kmax=KMAX_EXTREMOGRAMA, factor=FACTOR)
    sens = [D.escala_extremograma(ex.rho, ex.ref, f)[0] for f in FACTORES_SENSIBILIDAD]
    corte_h = float("nan") if corte.corte_h is None else corte.corte_h
    return np.array([t.tau_e_h, t.tau_int_h, *th, *tc, corte_h, ex.escala_h, *sens], dtype=float)


def trabajo_replicas(args) -> list[list[float]]:
    bloques, indices, u = args
    return [vector([bloques[i] for i in r], u).tolist() for r in indices]


def replicas(bloques, u, procesos_: int, avance, aviso=None, b: int = B) -> np.ndarray:
    """B réplicas del vector de escalas (forma B × len(NOMBRES)), reanudables."""
    rng = semillas.generador("f06", "clima", "dependencia")
    idx = BS.remuestrear_claves(list(range(len(bloques))), b, rng)
    tareas = [(f"rep_{a:04d}", (list(bloques), idx[a : a + TROZO], u)) for a in range(0, b, TROZO)]
    res = reanudable.ejecutar(trabajo_replicas, tareas, avance, procesos_, aviso)
    return np.asarray([fila for parte in res for fila in parte], dtype=float)


def intervalos(punto: np.ndarray, reps: np.ndarray) -> dict[str, dict]:
    out = {}
    for j, n in enumerate(NOMBRES):
        ic = BS.ic_percentil(reps[:, j], float(punto[j]), no_regular=False)
        out[n] = {
            "punto": ic.punto, "lo": ic.lo, "hi": ic.hi, "n_validas": ic.n_validas, "n_no_finitas": ic.n_no_finitas,
            "contiene_punto": bool(np.isfinite(ic.punto) and ic.contiene_punto),
        }  # fmt: skip
    return out


def g6(ic: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(no definidos o con IC no finito, IC que no contienen su punto). Ambos son marcas (§7, G6)."""
    sin, fuera = [], []
    for n in CON_IC:
        v = ic[n]
        if not (np.isfinite(v["punto"]) and np.isfinite(v["lo"]) and np.isfinite(v["hi"])):
            sin.append(n)
        elif not v["contiene_punto"]:
            fuera.append(n)
    return sin, fuera


# --- Descriptivos sin IC ---------------------------------------------------------------------------------


def sensibilidades(bloques, hs: pd.Series, u: dict[float, float]) -> dict:
    u99 = u[Q_PRINCIPAL]
    um = umbral_mensual(hs)
    principal = D.declustering_bloques(bloques, u99, INVIERNO)
    mensual = D.declustering_bloques(bloques, um, None)
    fijo = D.declustering_bloques(bloques, u99, None)
    junsep = D.declustering_bloques(bloques, u99, JUN_SEP)
    n_exc = sum(int(D.excedencias(b, u99).sum()) for b in bloques)
    ex_todo = D.extremograma_bloques(bloques, u99, None, kmax=KMAX_EXTREMOGRAMA, factor=FACTOR)
    huecos_v = D.huecos_bloques(bloques, u99, VERANO) * D.PASO_H

    def dc(x):
        return {"theta": x.theta, "T_C_h": x.T_C_horas, "n_int": x.n_int, "n_exc": x.n_exc}

    return {
        "invierno": dc(principal),
        "todo_el_anio_umbral_mensual": dc(mensual),
        "todo_el_anio_u_fijo": dc(fijo),
        "junio_septiembre": dc(junsep),
        "razon_T_C_mensual_frente_a_invierno": (max(mensual.T_C_horas, D.PASO_H) / max(principal.T_C_horas, D.PASO_H)),
        "excedencias_fuera_del_invierno": n_exc - principal.n_exc,
        "excedencias_totales": n_exc,
        "extremograma_todo_el_anio_escala_h": ex_todo.escala_h,
        "huecos_verano_h": _resumen(huecos_v),
        "umbral_mensual_m": {str(m): float(um[m]) for m in range(1, 13)},
    }


def _resumen(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(v.size), "p10": q[0], "p25": q[1], "mediana": q[2], "p75": q[3], "p90": q[4],
            "max": float(v.max())}  # fmt: skip


def por_tramo(hs: pd.Series, u: dict[float, float]) -> dict:
    """θ, T_(C) y huecos antes y después de 1993-02 (descriptivo; la asimilación cambia la serie)."""
    out = {}
    for nombre, parte in (("antes_1993_02", hs[hs.index < INICIO_ASIMILACION]),
                          ("desde_1993_02", hs[hs.index >= INICIO_ASIMILACION])):  # fmt: skip
        bl = D.bloques_anuales(parte)
        dc = D.declustering_bloques(bl, u[Q_PRINCIPAL], INVIERNO)
        h = D.huecos_bloques(bl, u[Q_PRINCIPAL], INVIERNO) * D.PASO_H
        out[nombre] = {"theta": dc.theta, "T_C_h": dc.T_C_horas, "n_exc": dc.n_exc, "huecos_h": _resumen(h)}
    return out


def por_anio(bloques, u99: float) -> pd.DataFrame:
    filas = []
    for b in bloques:
        dc = D.declustering_bloques([b], u99, INVIERNO)
        filas.append({"anio": b.anio, "theta": dc.theta, "n_exc": int(D.excedencias(b, u99).sum())})
    return pd.DataFrame(filas)


def autocorrelacion_entre_anios(df: pd.DataFrame) -> dict:
    """Autocorrelación de desfase 1 entre años de θ̂ por año y del número de excedencias (supuesto del bootstrap)."""
    out = {}
    for c in ("theta", "n_exc"):
        v = df[c].to_numpy(float)
        a, b = v[:-1], v[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        out[c] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    return out


def rachas_tabla(serie: pd.Series, u99: float) -> pd.DataFrame:
    """Rachas de excedencias consecutivas sobre u (sin tolerancia, todo el año) y el hueco hasta la siguiente."""
    x = serie.to_numpy(float)
    e = np.nan_to_num(x, nan=-np.inf) > u99
    d = np.diff(np.r_[0, e.astype(int), 0])
    ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    t = serie.index
    hueco = np.r_[(ini[1:] - fin[:-1]) * D.PASO_H, np.nan]
    return pd.DataFrame({
        "inicio": t[ini], "fin": t[fin], "duracion_h": (fin - ini + 1) * D.PASO_H,
        "hs_max": [float(np.nanmax(x[a : b + 1])) for a, b in zip(ini, fin, strict=True)],
        "mes": t[ini].month, "hueco_siguiente_h": hueco,
    })  # fmt: skip


def concordancia(ic: dict[str, dict], factor: float, regla_no_discrimina: bool) -> dict:
    """Regla de §3.5 con las tres escalas de la cola a P99 (invierno); τ_e no entra."""
    tc = ic[f"T_C_h_q{Q_PRINCIPAL:g}"]["punto"]
    corte = ic["corte_h"]["punto"]
    esc = ic["escala_extremograma_h"]["punto"]
    disp = {"T_C_h": max(tc, D.PASO_H), "corte_h": corte, "escala_extremograma_h": esc}
    disp = {k: v for k, v in disp.items() if np.isfinite(v)}
    razon = max(disp.values()) / min(disp.values()) if len(disp) >= 2 else float("nan")
    th = [ic[f"theta_q{q:g}"]["punto"] for q in CUANTILES_THETA]
    variacion = float(np.nanmax(th) - np.nanmin(th))
    senales = []
    if np.isfinite(razon) and razon > factor:
        senales.append("escalas de dependencia discordantes")
    if variacion > VARIACION_THETA:
        senales.append(f"θ(u) varía {variacion:.2f} entre P95 y P99.5 (> {VARIACION_THETA})")
    if not np.isfinite(corte):
        senales.append("sin corte natural de los huecos")
    return {
        "escalas_h": disp, "razon_max_min": razon, "factor": factor, "regla_no_discrimina": regla_no_discrimina,
        "concuerdan": bool(np.isfinite(razon) and razon <= factor),
        "rango_h": [min(disp.values()), max(disp.values())] if disp else None,
        "variacion_theta": variacion, "senales_para_f7": senales,
        "nota": "Con factor > 3 la regla no discrimina (etapa 0): «concuerdan» no es evidencia de concordancia.",
    }  # fmt: skip
