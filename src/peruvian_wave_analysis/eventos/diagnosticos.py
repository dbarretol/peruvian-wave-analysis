"""Diagnósticos de independencia y bondad de ajuste de la F7 (contrato §3.5). Funciones puras; la etapa con datos
está en ``diagnosticos_datos.py``. **Ningún diagnóstico elige nada:** no mueven u\\* ni el IET\\*.

1. **Prueba de Poisson sobre las rachas** (C-03, C2-04, E-27). Llegadas = inicios de las rachas con algún pico
   > u\\*; tiempo muerto = (t_inicio, t_fin + IET] de **todas** las rachas sobre u_id. La intensidad sale de una
   regresión de Poisson sobre el tiempo en riesgo:

   - ``armonico``: log λ(t) = β0 + β1 sen φ + β2 cos φ + β3 sen 2φ + β4 cos 2φ, φ = 2π·(día del año)/365.25, con
     φ tomado en el paso de 3 h de cada celda. La regresión se ajusta sobre las celdas agrupadas por su paso de 3 h
     dentro del año, lo que es exacto porque la covariable solo depende de ese paso;
   - ``estaciones``: λ constante por estación (sensibilidad de §3.7, punto 6): 2 estaciones (mayo–octubre y
     noviembre–abril, como B en la F6) o 4 (DEF, MAM, JJA, SON; rotulado: el contrato no las define).

   Dos regresiones sobre el mismo tiempo en riesgo: λ̂_todas (todas las rachas) y λ̂_marcadas (las rachas > u\\*).
   La prueba reescala con λ̂_marcadas (``extremos.reescalado``). A² decide, calibrado con un bootstrap paramétrico
   del procedimiento completo (B = 199): se simulan todas las rachas con λ̂_todas y duraciones remuestreadas, cada
   una se marca con probabilidad p̂(t) = min{1, λ̂_marcadas(t)/λ̂_todas(t)} en su inicio, y se reajustan las dos
   regresiones en cada réplica.
2. **Dispersión de los conteos anuales:** índice de dispersión con la exposición de cada año, p por simulación de
   conteos de Poisson con la misma exposición.
3. **τ de Kendall entre excesos de picos consecutivos,** con p por permutación del orden dentro de cada año
   (bilateral): todos los pares, los separados por < 7 días y los de una misma racha dividida.
4. **Separaciones entre picos** y fracción en [IET, IET + 24 h].
5. **A² de Anderson–Darling en u\\*** contra la GPD ajustada, con bootstrap paramétrico (B = 999), réplicas
   redondeadas a 0.01 m como en R3 (C2-05) y reajuste con ``gpd_perfil.ajustar_rapido`` (ADR-0034).
6. **Süveges y Davison (2010):** θ por K-gaps (ec. 3) y prueba de la matriz de información (ec. 4 y apéndice),
   T ~ χ²₁, solo en las celdas con K·F̄(u) ≤ 0.2 y F̄(u) = 1 − q nominal (C-10, C2-08).
7. **Picos de verano** (noviembre a abril) sobre u\\*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..extremos import gpd, gpd_perfil
from ..extremos import reescalado as RS
from . import umbral as U

PASO_H = 3.0
PASOS_POR_DIA = 8
ANIO_DIAS = 365.25
B_POISSON = 199
B_A2 = 999
B_PERMUTACION = 1999
B_DISPERSION = 9999
ALFA = 0.05
DIAS_PAR_CERCANO = 7.0
MIN_PARES_TAU = 3
ANCHO_FRANJA_H = 24.0
MESES_VERANO = (11, 12, 1, 2, 3, 4)
ESTACIONES_2 = {"may-oct": (5, 6, 7, 8, 9, 10), "nov-abr": (11, 12, 1, 2, 3, 4)}
ESTACIONES_4 = {"DEF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MESES_MAYO_OCTUBRE = ESTACIONES_2["may-oct"]
Q_SD = (0.99, 0.995)
KF_MAX = 0.2
ITER_MAX_GLM = 100


# --- 1. Prueba de Poisson con intensidad estacional -------------------------------------------------------------


@dataclass(frozen=True)
class Modelo:
    """Intensidad por grupo de celdas. ``celda[i]`` = grupo de la celda i; ``X`` = diseño por grupo (armónico) o
    ``None`` (constante por grupo: estaciones)."""

    tipo: str
    celda: np.ndarray
    n_grupos: int
    X: np.ndarray | None
    nombres: tuple[str, ...] = ()


def paso_del_anio(tiempos) -> np.ndarray:
    """Paso de 3 h dentro del año: (día del año − 1)·8 + hora // 3 (0 a 2927)."""
    t = pd.DatetimeIndex(tiempos)
    return (t.day_of_year.to_numpy() - 1) * PASOS_POR_DIA + t.hour.to_numpy() // int(PASO_H)


def diseno_armonico(n_grupos: int = 366 * PASOS_POR_DIA) -> np.ndarray:
    fase = 2.0 * np.pi * (np.arange(n_grupos) / PASOS_POR_DIA) / ANIO_DIAS
    return np.column_stack([np.ones(n_grupos), np.sin(fase), np.cos(fase), np.sin(2 * fase), np.cos(2 * fase)])


def modelo_armonico(tiempos) -> Modelo:
    X = diseno_armonico()
    return Modelo("armonico", paso_del_anio(tiempos), X.shape[0], X)


def modelo_estaciones(tiempos, estaciones: Mapping[str, Sequence[int]]) -> Modelo:
    meses = pd.DatetimeIndex(tiempos).month.to_numpy()
    celda = np.full(meses.size, -1)
    for j, ms in enumerate(estaciones.values()):
        celda[np.isin(meses, list(ms))] = j
    if np.any(celda < 0):
        raise ValueError("Las estaciones deben cubrir los 12 meses")
    return Modelo("estaciones", celda, len(estaciones), None, tuple(estaciones))


def _glm_poisson(y: np.ndarray, E: np.ndarray, X: np.ndarray) -> np.ndarray:
    """β de la regresión de Poisson y ~ Poisson(E·exp(Xβ)) por Newton con paso amortiguado."""
    ok = E > 0
    y, E, X = y[ok], E[ok], X[ok]
    if y.sum() <= 0:
        raise ValueError("Regresión de Poisson sin eventos")
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / E.sum())

    def ll(b):
        eta = X @ b
        return float(np.sum(y * eta - E * np.exp(eta)))

    actual = ll(beta)
    for _ in range(ITER_MAX_GLM):
        mu = E * np.exp(X @ beta)
        paso = np.linalg.solve((X * mu[:, None]).T @ X, X.T @ (y - mu))
        t = 1.0
        while t > 1e-8:
            nuevo = ll(beta + t * paso)
            if nuevo >= actual - 1e-12:
                break
            t /= 2.0
        beta = beta + t * paso
        cambio, actual = nuevo - actual, nuevo
        if np.max(np.abs(t * paso)) < 1e-9 or abs(cambio) < 1e-12:
            return beta
    raise ValueError("La regresión de Poisson no convergió")


def ajustar_intensidad(modelo: Modelo, en_riesgo: np.ndarray, llegadas: np.ndarray, paso_horas: float = PASO_H):
    """Intensidad por celda (eventos por hora) ajustada sobre las celdas en riesgo; ``llegadas`` = posiciones."""
    E = paso_horas * np.bincount(modelo.celda[en_riesgo], minlength=modelo.n_grupos).astype(float)
    y = np.bincount(modelo.celda[llegadas], minlength=modelo.n_grupos).astype(float)
    if modelo.X is None:
        lam = np.divide(y, E, out=np.zeros(modelo.n_grupos), where=E > 0)
        return lam[modelo.celda], {s: float(v) for s, v in zip(modelo.nombres, lam, strict=True)}
    beta = _glm_poisson(y, E, modelo.X)
    return np.exp(modelo.X @ beta)[modelo.celda], {"beta": [float(b) for b in beta]}


def simular_rachas(validos, lam, duraciones, celdas_muertas: int, paso_horas: float, rng: np.random.Generator):
    """Inicios de Poisson con intensidad por celda en el tiempo en riesgo y duraciones remuestreadas (en celdas)."""
    n = len(validos)
    candidatos = np.flatnonzero((rng.random(n) < lam * paso_horas) & validos)
    ini, fin = [], []
    bloqueado = -1
    for i in candidatos:
        if i > bloqueado:
            e = min(i + int(duraciones[rng.integers(duraciones.size)]), n - 1)
            ini.append(i)
            fin.append(e)
            bloqueado = e + celdas_muertas
    return np.asarray(ini, dtype=int), np.asarray(fin, dtype=int)


def _z(en_riesgo, lam, llegadas, paso_horas):
    acum = np.cumsum(np.where(en_riesgo, lam * paso_horas, 0.0))
    return np.diff(acum[llegadas])


def prueba_poisson_rachas(
    tiempos,
    validos,
    ini: np.ndarray,
    fin: np.ndarray,
    marcadas: np.ndarray,
    iet_horas: float,
    modelo: Modelo,
    rng: np.random.Generator,
    B: int = B_POISSON,
    paso_horas: float = PASO_H,
) -> dict:
    """§3.5, punto 1. ``ini``/``fin``: posiciones de todas las rachas sobre u_id; ``marcadas``: las > u\\*."""
    tiempos = pd.DatetimeIndex(tiempos)
    validos = np.asarray(validos, dtype=bool)
    ini, fin, marcadas = np.asarray(ini, int), np.asarray(fin, int), np.asarray(marcadas, bool)
    muertas = int(round(iet_horas / paso_horas))
    en_riesgo = RS.celdas_en_riesgo(validos, ini, fin, muertas)
    lam_t, par_t = ajustar_intensidad(modelo, en_riesgo, ini, paso_horas)
    lam_m, par_m = ajustar_intensidad(modelo, en_riesgo, ini[marcadas], paso_horas)
    res = RS.intervalos_reescalados(
        tiempos,
        validos,
        tiempos[ini],
        tiempos[fin],
        None,
        iet_horas,
        paso_horas,
        llegadas=tiempos[ini[marcadas]],
        intensidad=lam_m,
    )
    if res.z.size < RS.MIN_INTERVALOS:
        raise ValueError(f"Hacen falta al menos {RS.MIN_INTERVALOS} intervalos; hay {res.z.size}")
    ad0, ks0 = RS.estadistico_ad(res.z), RS.estadistico_ks(res.z)
    p_marca = np.minimum(1.0, np.divide(lam_m, lam_t, out=np.zeros_like(lam_m), where=lam_t > 0))
    dur = fin - ini
    k_ad = k_ks = validas = generadas = 0
    tope = int(np.floor(RS.FACTOR_MAX_GENERACIONES * B))
    while validas < B:
        if generadas >= tope:
            raise ValueError(f"Hicieron falta más de {tope} réplicas para tener {B} válidas")
        generadas += 1
        si, fi = simular_rachas(validos, lam_t, dur, muertas, paso_horas, rng)
        mk = rng.random(si.size) < p_marca[si]
        if mk.sum() < RS.MIN_INTERVALOS + 1:
            continue
        er = RS.celdas_en_riesgo(validos, si, fi, muertas)
        try:
            ajustar_intensidad(modelo, er, si, paso_horas)
            lm, _ = ajustar_intensidad(modelo, er, si[mk], paso_horas)
        except ValueError:
            continue
        zb = _z(er, lm, si[mk], paso_horas)
        validas += 1
        k_ad += RS.estadistico_ad(zb) >= ad0
        k_ks += RS.estadistico_ks(zb) >= ks0
    p_ad = (1 + k_ad) / (B + 1)
    return {
        "modelo": modelo.tipo if modelo.X is not None else f"{modelo.n_grupos}_estaciones",
        "ad": ad0,
        "p_ad": p_ad,
        "rechaza": bool(p_ad <= ALFA + 1e-12),
        "ks": ks0,
        "p_ks": (1 + k_ks) / (B + 1),
        "n_rachas": int(ini.size),
        "n_llegadas": int(marcadas.sum()),
        "n_intervalos": int(res.z.size),
        "B": B,
        "generadas": generadas,
        "intensidad_todas": par_t,
        "intensidad_marcadas": par_m,
    }


def modelo_por_nombre(nombre: str, tiempos) -> Modelo:
    if nombre == "armonico":
        return modelo_armonico(tiempos)
    return modelo_estaciones(tiempos, {"2_estaciones": ESTACIONES_2, "4_estaciones": ESTACIONES_4}[nombre])


def intensidad_invernal(tiempos, por_anio: float, amplitud: float = 2.2, dia_maximo: float = 196.0) -> np.ndarray:
    """Intensidad sinusoidal (eventos por hora) con máximo en julio; con amplitud 2.2, ≈ 89 % de los inicios caen
    en mayo–octubre (el contrato pide 86–94 %, §8)."""
    fase = 2 * np.pi * (paso_del_anio(tiempos) / PASOS_POR_DIA) / ANIO_DIAS
    f = np.exp(amplitud * np.cos(fase - 2 * np.pi * dia_maximo / ANIO_DIAS))
    return por_anio / (ANIO_DIAS * 24.0) * f / f.mean()


def proceso_sintetico(tiempos, rng: np.random.Generator, por_anio: float = 40.0, iet_horas: float = 48.0):
    """Proceso de §8 para el tamaño de la prueba completa: rachas de Poisson concentradas en invierno, con tiempo
    muerto, y una fracción > u\\* estacional (0.14 en mayo–octubre y 0.07 el resto: el doble en invierno)."""
    t = pd.DatetimeIndex(tiempos)
    lam = intensidad_invernal(t, por_anio)
    muertas = int(round(iet_horas / PASO_H))
    ini, fin = simular_rachas(np.ones(t.size, bool), lam, rng.integers(1, 12, 300), muertas, PASO_H, rng)
    marcadas = rng.random(ini.size) < np.where(np.isin(t[ini].month, MESES_MAYO_OCTUBRE), 0.14, 0.07)
    return ini, fin, marcadas


def rejilla_sintetica(anios: float = 10.0, inicio: str = "2000-01-01") -> pd.DatetimeIndex:
    return pd.date_range(inicio, periods=int(round(anios * ANIO_DIAS * PASOS_POR_DIA)), freq="3h")


# --- 2. Dispersión de los conteos anuales --------------------------------------------------------------------


def exposicion_por_anio(tiempos, validos) -> pd.Series:
    """Años de exposición por año civil: celdas válidas × 3 h / (365.25 × 24 h)."""
    t = pd.DatetimeIndex(tiempos)
    v = np.asarray(validos, dtype=bool)
    s = pd.Series(v.astype(float) * PASO_H / (ANIO_DIAS * 24.0), index=t.year)
    e = s.groupby(level=0).sum()
    return e[e > 0]


def dispersion_anual(anios_picos, exposicion: pd.Series, rng: np.random.Generator, B: int = B_DISPERSION) -> dict:
    """Índice de dispersión D = Σ (N_a − λ e_a)² / (λ e_a) / (A − 1), con λ = ΣN / Σe; p por simulación de
    conteos de Poisson con media λ e_a (unilateral: sobredispersión)."""
    e = exposicion.to_numpy(float)
    n = pd.Series(np.asarray(anios_picos)).value_counts().reindex(exposicion.index, fill_value=0).to_numpy(float)
    if np.asarray(anios_picos).size and not np.isin(np.asarray(anios_picos), exposicion.index).all():
        raise ValueError("Hay picos en años sin exposición")
    lam = n.sum() / e.sum()
    mu = lam * e

    def indice(c):
        return np.sum((c - mu) ** 2 / mu, axis=-1) / (e.size - 1)

    d0 = float(indice(n))
    sim = indice(rng.poisson(mu, size=(B, e.size)).astype(float))
    p = (1 + int(np.sum(sim >= d0 - 1e-12))) / (B + 1)
    return {
        "indice": d0,
        "p": p,
        "anios": int(e.size),
        "picos": int(n.sum()),
        "lambda_por_anio": float(lam),
        "media": float(n.mean()),
        "varianza": float(n.var(ddof=1)),
        "B": B,
    }


# --- 3. τ de Kendall entre picos consecutivos ----------------------------------------------------------------


def _tau(a, b) -> float:
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(stats.kendalltau(a, b).statistic)


def tau_consecutivos(t_pico, excesos, id_racha, rng: np.random.Generator, B: int = B_PERMUTACION) -> dict:
    """τ de Kendall entre (x_i, x_{i+1}) de picos ordenados en el tiempo; p bilateral por permutación de los
    excesos dentro de cada año civil (las posiciones y los subconjuntos de pares quedan fijos)."""
    t = pd.DatetimeIndex(t_pico)
    o = np.argsort(t.to_numpy(), kind="stable")
    t, x, r = t[o], np.asarray(excesos, float)[o], np.asarray(id_racha)[o]
    sep_d = np.diff(t.to_numpy()).astype("timedelta64[s]").astype(float) / 86400.0
    subconjuntos = {
        "todos": np.ones(sep_d.size, bool),
        "menos_de_7_dias": sep_d < DIAS_PAR_CERCANO,
        "misma_racha": r[1:] == r[:-1],
    }
    anios = t.year.to_numpy()
    obs = {k: _tau(x[:-1][m], x[1:][m]) for k, m in subconjuntos.items()}
    cuenta = dict.fromkeys(subconjuntos, 0)
    for _ in range(B):
        xp = x[np.lexsort((rng.random(x.size), anios))]
        for k, m in subconjuntos.items():
            if m.sum() >= MIN_PARES_TAU and np.isfinite(obs[k]):
                tb = _tau(xp[:-1][m], xp[1:][m])
                cuenta[k] += bool(np.isfinite(tb) and abs(tb) >= abs(obs[k]) - 1e-12)
    out = {}
    for k, m in subconjuntos.items():
        ok = m.sum() >= MIN_PARES_TAU and np.isfinite(obs[k])
        p = (1 + cuenta[k]) / (B + 1) if ok else None
        out[k] = {
            "pares": int(m.sum()),
            "tau": obs[k] if ok else None,
            "p": p,
            "rechaza": bool(p <= ALFA + 1e-12) if ok else None,
        }
    out["B"] = B
    return out


# --- 4. Separaciones entre picos ------------------------------------------------------------------------------


def separaciones(t_pico, iet_horas: float, ancho_h: float = ANCHO_FRANJA_H) -> dict:
    t = np.sort(pd.DatetimeIndex(t_pico).to_numpy())
    h = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
    franja = (h >= iet_horas - 1e-9) & (h <= iet_horas + ancho_h + 1e-9)
    return {
        "n": int(h.size),
        "en_franja": int(franja.sum()),
        "fraccion_en_franja": float(franja.mean()) if h.size else None,
        "franja_h": [iet_horas, iet_horas + ancho_h],
        "mediana_h": float(np.median(h)) if h.size else None,
        "minimo_h": float(h.min()) if h.size else None,
        "horas": h,
    }


# --- 5. Bondad de ajuste en u* -----------------------------------------------------------------------------------


def a2_gpd(excesos, xi: float, sigma: float) -> float:
    """A² de Anderson–Darling de los excesos contra la GPD (ξ, σ)."""
    z = np.sort(np.clip(1.0 - gpd.supervivencia(np.asarray(excesos, float), xi, sigma), 1e-300, 1.0 - 1e-16))
    n = z.size
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1]))))


def bondad_a2(picos, u: float, rng: np.random.Generator, B: int = B_A2, reajustar=gpd_perfil.ajustar_rapido) -> dict:
    x = np.asarray(picos, float)
    e = x[x > u] - u
    aj = gpd.ajustar_gpd(e)
    if not aj.valido:
        return {"n": int(e.size), "a2": None, "p": None, "B": B, "generadas": 0, "motivo": "ajuste no válido"}
    a0 = a2_gpd(e, aj.xi, aj.sigma)
    m = validas = generadas = 0
    tope = int(U.FACTOR_MAX_REPLICAS_R3 * B)
    while validas < B and generadas < tope:
        generadas += 1
        r = U.redondear_replica(u, gpd.cuantil_supervivencia(rng.random(e.size), aj.xi, aj.sigma))
        ab = reajustar(r)
        if not ab.valido:
            continue
        validas += 1
        m += a2_gpd(r, ab.xi, ab.sigma) >= a0
    p = (1 + m) / (B + 1) if validas == B else None
    return {
        "n": int(e.size),
        "xi": aj.xi,
        "sigma": aj.sigma,
        "a2": a0,
        "p": p,
        "rechaza": bool(p <= ALFA + 1e-12) if p is not None else None,
        "B": B,
        "generadas": generadas,
    }


# --- 6. Süveges y Davison (2010) ---------------------------------------------------------------------------------


def celdas_sd(
    iets_h: Sequence[float], qs: Sequence[float] = Q_SD, paso_horas: float = PASO_H, kf_max: float = KF_MAX
) -> list[dict]:
    """Celdas (q, IET) con K·F̄(u) ≤ 0.2, F̄(u) = 1 − q nominal y K = IET / 3 h (C-10, C2-08). No lee datos."""
    out = []
    for q in qs:
        for iet in iets_h:
            K = int(round(iet / paso_horas))
            kf = K * (1.0 - q)
            if kf <= kf_max + 1e-9:
                out.append({"q": q, "iet_h": float(iet), "K": K, "KF": kf})
    return out


def kgaps(posiciones, K: int, F_barra: float, bloques=None) -> dict:
    """θ por K-gaps (ec. 3) y la prueba de la matriz de información (ec. 4; apéndice). ``posiciones``: índices de
    las excedencias, crecientes; ``bloques``: si se da, solo cuentan los huecos dentro de un mismo bloque."""
    j = np.asarray(posiciones, dtype=np.int64)
    T = np.diff(j)
    if bloques is not None:
        b = np.asarray(bloques)
        T = T[b[1:] == b[:-1]]
    c = F_barra * np.maximum(T - K, 0).astype(float)
    n = c.size
    i1 = c > 0
    nc, sc = int(i1.sum()), float(c.sum())
    if n < 2 or nc == 0:
        return {"N_huecos": n, "N_C": nc, "theta": None, "T": None, "p": None}
    bb = sc + n + nc
    theta = min(1.0, (bb - np.sqrt(max(bb * bb - 8.0 * nc * sc, 0.0))) / (2.0 * sc))
    i0 = ~i1
    uno = 1.0 - theta
    inv0 = np.where(i0, 1.0 / max(uno, 1e-300) ** 2, 0.0) if uno > 0 else np.zeros(n)
    score = np.where(i0, -1.0 / uno if uno > 0 else 0.0, 0.0) + np.where(i1, 2.0 / theta, 0.0) - c
    info = float(np.mean(inv0 + np.where(i1, 2.0 / theta**2, 0.0)))
    d = np.where(i1, 2.0 / theta**2, 0.0) + c**2 - 4.0 * c / theta
    Dn = float(d.mean())
    Dp = float(np.mean(np.where(i1, -4.0 / theta**3, 0.0) + 4.0 * c / theta**2))
    Vn = float(np.mean((d - Dp / info * score) ** 2))
    Tst = n * Dn**2 / Vn if Vn > 0 else np.nan
    return {
        "N_huecos": n,
        "N_C": nc,
        "theta": float(theta),
        "T": float(Tst),
        "p": float(stats.chi2.sf(Tst, 1)) if np.isfinite(Tst) else None,
    }


# --- 7. Picos de verano ------------------------------------------------------------------------------------------


def picos_de_verano(t_pico, hs, u: float) -> int:
    t = pd.DatetimeIndex(t_pico)
    return int(np.sum(np.isin(t.month, MESES_VERANO) & (np.asarray(hs, float) > u)))


# --- Rachas para la prueba de Poisson ----------------------------------------------------------------------------


def rachas_de_era(
    eventos: pd.DataFrame, n_celdas: int, inicio_pos: int, fin_pos: int, celdas_muertas: int, validos_serie: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(ini, fin, id_racha, validos, rachas que cruzan el inicio) de las rachas que empiezan en [inicio, fin] de la
    era. Las válidas son las celdas con dato dentro de la era; una racha que empieza antes del inicio y termina
    dentro saca de la era su resto y su tiempo muerto (no puede ser llegada, porque su inicio queda fuera)."""
    g = eventos.groupby("id_racha").agg(i_ini=("i_inicio", "min"), i_fin=("i_fin", "max")).sort_index()
    validos = np.zeros(n_celdas, bool)
    validos[inicio_pos : fin_pos + 1] = validos_serie[inicio_pos : fin_pos + 1]
    cruzan = g[(g["i_ini"] < inicio_pos) & (g["i_fin"] + celdas_muertas >= inicio_pos)]
    for f in cruzan.itertuples():
        validos[inicio_pos : min(int(f.i_fin) + celdas_muertas, n_celdas - 1) + 1] = False
    dentro = g[(g["i_ini"] >= inicio_pos) & (g["i_ini"] <= fin_pos)]
    dentro = dentro[validos[dentro["i_ini"].to_numpy()]]
    return (
        dentro["i_ini"].to_numpy(int),
        dentro["i_fin"].to_numpy(int),
        dentro.index.to_numpy(),
        validos,
        int(len(cruzan)),
    )
