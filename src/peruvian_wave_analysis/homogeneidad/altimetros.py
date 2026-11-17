"""Prueba de saltos frente a pasadas de altímetros (contrato de la F4, §3.4; requisitos recibidos de la F3).

**Exposición.** Una medición cuenta si su bandera de calidad de Hs es 1, está a ≤ 200 km del nodo y sobre mar
(profundidad del archivo > 0). No se lee Hs. Las mediciones de una misión separadas más de 60 s son pasadas
distintas; la hora de una pasada es la de su primera medición. Un paso t (rejilla de 3 h) tiene pasada de una
misión en la ventana centrada si alguna pasada cae en [t − 1.5 h, t + 1.5 h), y en la ventana hacia atrás si cae en
(t − 3 h, t].

**Exposición incompleta** (§3.4 punto 2): pasos en un hueco documentado (ERS-2 con Cuiabá), pasos desde el inicio
de una fuente asimilada sin datos locales (SAR) y pasos posteriores al último dato local de cualquier misión.

**Saltos y eventos.** Salto en t si el criterio (relativo |ΔHs(t)|/Hs(t − 1) > c, o absoluto |ΔHs(t)| > k·s) se
cumple. Un evento es una racha de pasos consecutivos con salto; su exposición es la de su primer paso. Los pasos
dentro de un evento, después del primero, no pueden empezar otro y salen de los denominadores.

**Cociente de Mantel-Haenszel** de tasas (eventos por paso), con pasada / sin pasada, combinado en estratos, con
el IC de Greenland y Robins (1985) para log RR.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..sitio.exposicion import distancia_km


# --- Lectura y pasadas ------------------------------------------------------------------------
def pasadas(med: pd.DataFrame, lat0: float, lon0: float, radio_km: float, corte_s: float) -> pd.DataFrame:
    """De mediciones (mision, tiempo, lat, lon, prof_archivo_m) a pasadas (mision, tiempo, n, dist_min_km) dentro
    del radio y sobre mar."""
    d = med.copy()
    d["dist_km"] = distancia_km(lat0, lon0, d["lat"], d["lon"])
    prof = -pd.to_numeric(d["prof_archivo_m"], errors="coerce")  # BOT_DEPTH es negativo bajo el mar
    mar = prof.isna() | (prof > 0)
    d = d[(d["dist_km"] <= radio_km) & mar].sort_values(["mision", "tiempo"])
    if d.empty:
        return pd.DataFrame(columns=["mision", "tiempo", "n", "dist_min_km"])
    nuevo = d.groupby("mision")["tiempo"].diff().dt.total_seconds().fillna(np.inf) > corte_s
    d["pasada"] = nuevo.groupby(d["mision"]).cumsum()
    g = d.groupby(["mision", "pasada"])
    out = g.agg(tiempo=("tiempo", "min"), n=("tiempo", "size"), dist_min_km=("dist_km", "min")).reset_index()
    return out.drop(columns="pasada").sort_values(["tiempo", "mision"]).reset_index(drop=True)


def en_ventana(rejilla: pd.DatetimeIndex, tiempos, desde_h: float, hasta_h: float, cerrada_derecha: bool) -> np.ndarray:
    """Para cada paso de la rejilla, si algún tiempo cae en la ventana [t+desde, t+hasta) o (t+desde, t+hasta]."""
    r = pd.DatetimeIndex(rejilla).as_unit("ns").asi8  # misma unidad en los dos (pandas puede inferir segundos)
    t = np.sort(pd.DatetimeIndex(tiempos).as_unit("ns").asi8)
    h = 3_600 * 10**9
    lo, hi = r + int(desde_h * h), r + int(hasta_h * h)
    if cerrada_derecha:
        a, b = np.searchsorted(t, lo, side="right"), np.searchsorted(t, hi, side="right")
    else:
        a, b = np.searchsorted(t, lo, side="left"), np.searchsorted(t, hi, side="left")
    return b > a


def exposicion(
    rejilla: pd.DatetimeIndex,
    pas: pd.DataFrame,
    ventanas: dict[str, tuple[float, float, bool]],
) -> pd.DataFrame:
    """Una columna booleana por (ventana, misión): ``<ventana>:<mision>``."""
    cols = {}
    for nv, (a, b, cerr) in ventanas.items():
        for m, g in pas.groupby("mision"):
            cols[f"{nv}:{m}"] = en_ventana(rejilla, g["tiempo"], a, b, cerr)
    return pd.DataFrame(cols, index=rejilla)


def asimilada_en(mision: str, t: pd.DatetimeIndex, asimiladas: dict, no_verificables: Sequence[str]) -> np.ndarray:
    """Si la misión está asimilada en cada t: período de la tabla 3, o no verificable (tratada como asimilada desde
    1993)."""
    if mision in asimiladas:
        a, b = asimiladas[mision]
        dentro = t >= pd.Timestamp(a)
        if b is not None:
            dentro &= t < pd.Timestamp(b) + pd.Timedelta(days=1)
        return np.asarray(dentro)
    if mision in no_verificables:
        return np.asarray(t >= pd.Timestamp("1993-01-01"))
    return np.zeros(len(t), bool)


def con_pasada_asimilada(
    expo: pd.DataFrame, ventana: str, asimiladas: dict, no_verificables: Sequence[str], excluir=()
) -> np.ndarray:
    t = expo.index
    out = np.zeros(len(t), bool)
    for c in expo.columns:
        v, m = c.split(":", 1)
        if v != ventana or m in excluir:
            continue
        out |= expo[c].to_numpy() & asimilada_en(m, t, asimiladas, no_verificables)
    return out


def completa(
    rejilla: pd.DatetimeIndex, huecos: Sequence[tuple[str, str, str]], sin_datos_locales: dict, ultimo_local
) -> np.ndarray:
    """Exposición completa en cada paso (§3.4 punto 2)."""
    ok = np.ones(len(rejilla), bool)
    for a, b, _ in huecos:
        ok &= ~((rejilla >= pd.Timestamp(a)) & (rejilla < pd.Timestamp(b) + pd.Timedelta(days=1)))
    for inicio in sin_datos_locales.values():
        ok &= ~(rejilla >= pd.Timestamp(inicio))
    ok &= ~(rejilla > pd.Timestamp(ultimo_local))
    return ok


# --- Saltos ------------------------------------------------------------------------------------
def cocientes_relativos(hs: np.ndarray) -> np.ndarray:
    """r(t) = |Hs(t) − Hs(t−1)| / Hs(t−1); NaN en t = 0 o con dato ausente o Hs(t−1) ≤ 0."""
    hs = np.asarray(hs, float)
    r = np.full(hs.size, np.nan)
    prev = hs[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.where(prev > 0, np.abs(np.diff(hs)) / prev, np.nan)
    return r


def umbral_c_T0(hs: pd.Series, inicio, fin, cuantil: float) -> float:
    """P99 de |ΔHs(t)|/Hs(t−1) solo con pares (t−1, t) los dos dentro de [inicio, fin] (v7)."""
    r = cocientes_relativos(hs.to_numpy(float))
    t = hs.index
    dentro = (t >= pd.Timestamp(inicio)) & (t < pd.Timestamp(fin) + pd.Timedelta(days=1))
    par = dentro.copy()
    par[1:] &= dentro[:-1]
    par[0] = False
    x = r[par]
    return float(np.quantile(x[np.isfinite(x)], cuantil))


def saltos_relativos(hs: np.ndarray, c: float) -> np.ndarray:
    r = cocientes_relativos(hs)
    return np.nan_to_num(r, nan=-1.0) > c


def saltos_absolutos(hs: np.ndarray, umbral_m: float) -> np.ndarray:
    hs = np.asarray(hs, float)
    d = np.full(hs.size, np.nan)
    d[1:] = np.abs(np.diff(hs))
    return np.nan_to_num(d, nan=-1.0) > umbral_m


def inicios_de_evento(salto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inicio, elegible): inicio de cada racha de saltos; elegible = el paso puede empezar un evento (no está
    dentro de una racha ya empezada)."""
    s = np.asarray(salto, bool)
    prev = np.concatenate([[False], s[:-1]])
    return s & ~prev, ~(s & prev)


# --- Mantel-Haenszel -------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultadoMH:
    cociente: float
    lo: float
    hi: float
    eventos_con: int
    pasos_con: int
    eventos_sin: int
    pasos_sin: int
    estratos: int

    def excluye_1(self) -> bool:
        return bool(np.isfinite(self.lo) and (self.lo > 1 or self.hi < 1))


def mantel_haenszel(evento: np.ndarray, expuesto: np.ndarray, estrato: np.ndarray, nivel: float = 0.95) -> ResultadoMH:
    """Cociente de tasas por paso, combinado por estratos (Greenland y Robins, 1985). Solo pasos con estrato ≥ 0."""
    e = np.asarray(evento, bool)
    x = np.asarray(expuesto, bool)
    s = np.asarray(estrato, int)
    ok = s >= 0
    e, x, s = e[ok], x[ok], s[ok]
    if s.size == 0:
        return ResultadoMH(np.nan, np.nan, np.nan, 0, 0, 0, 0, 0)
    k = int(s.max()) + 1
    a = np.bincount(s, weights=(e & x), minlength=k)
    b = np.bincount(s, weights=(e & ~x), minlength=k)
    n1 = np.bincount(s, weights=x, minlength=k)
    n0 = np.bincount(s, weights=~x, minlength=k)
    T = n1 + n0
    usar = (n1 > 0) & (n0 > 0)
    a, b, n1, n0, T = a[usar], b[usar], n1[usar], n0[usar], T[usar]
    R = np.sum(a * n0 / T)
    S = np.sum(b * n1 / T)
    tot = (int(a.sum()), int(n1.sum()), int(b.sum()), int(n0.sum()), int(usar.sum()))
    if R <= 0 or S <= 0:
        rr = np.inf if (R > 0 and S <= 0) else (0.0 if S > 0 else np.nan)
        return ResultadoMH(rr, np.nan, np.nan, *tot)
    rr = R / S
    var = np.sum(n1 * n0 * (a + b) / T**2) / (R * S)
    zc = stats.norm.ppf(0.5 + nivel / 2)
    se = np.sqrt(var)
    return ResultadoMH(float(rr), float(rr * np.exp(-zc * se)), float(rr * np.exp(zc * se)), *tot)


def codigos_estrato(*claves: np.ndarray, validos: np.ndarray | None = None) -> np.ndarray:
    """Código entero por combinación de claves; −1 donde ``validos`` es falso o alguna clave es negativa."""
    cl = [np.asarray(c, int) for c in claves]
    n = cl[0].size
    ok = np.ones(n, bool) if validos is None else np.asarray(validos, bool).copy()
    for c in cl:
        ok &= c >= 0
    if not ok.any():
        return np.full(n, -1)
    pila = np.column_stack([c[ok] for c in cl])
    _, inv = np.unique(pila, axis=0, return_inverse=True)
    out = np.full(n, -1)
    out[ok] = inv.ravel()
    return out


def clase_por_intervalos(t: pd.DatetimeIndex, intervalos: Sequence[tuple[str, str]]) -> np.ndarray:
    """Índice del intervalo [a, b] (días incluidos) que contiene cada t; −1 si ninguno."""
    out = np.full(len(t), -1)
    for i, (a, b) in enumerate(intervalos):
        m = (t >= pd.Timestamp(a)) & (t < pd.Timestamp(b) + pd.Timedelta(days=1))
        out[np.asarray(m)] = i
    return out


def nivel_hs(hs_prev: np.ndarray, cortes: Sequence[float]) -> np.ndarray:
    """0, 1, 2 para Hs(t−1) < corte1, entre cortes, ≥ corte2; −1 si no finito."""
    h = np.asarray(hs_prev, float)
    out = np.searchsorted(np.asarray(cortes, float), h, side="right").astype(int)
    out[~np.isfinite(h)] = -1
    return out


# --- Potencia ----------------------------------------------------------------------------------
def potencia_mh(
    expuesto: np.ndarray,
    estrato: np.ndarray,
    tasa_base: float | np.ndarray,
    cociente: float,
    fraccion_oculta: float,
    nivel: float,
    replicas: int,
    rng: np.random.Generator,
) -> float:
    """Potencia de la lectura MH (IC al ``nivel`` que excluye el 1, del lado del cociente) con eventos de Bernoulli:
    tasa base en los pasos sin pasada y tasa·cociente en los pasos con pasada; luego se ocultan al azar un tercio
    de las pasadas (quedan como «sin pasada» con la tasa alta)."""
    x = np.asarray(expuesto, bool)
    s = np.asarray(estrato, int)
    ok = s >= 0
    x, s = x[ok], s[ok]
    base = np.broadcast_to(np.asarray(tasa_base, float), ok.shape)[ok]
    p = np.where(x, np.minimum(base * cociente, 1.0), base)
    rechazos = 0
    for _ in range(replicas):
        ev = rng.random(x.size) < p
        visto = x & (rng.random(x.size) >= fraccion_oculta)
        r = mantel_haenszel(ev, visto, s, nivel)
        rechazos += bool(np.isfinite(r.lo) and r.lo > 1)
    return rechazos / replicas


def calendario_desplazado(
    con_pasada: pd.Series, inicio: str, fin: str, anios: int, rejilla: pd.DatetimeIndex
) -> np.ndarray:
    """Exposición de [inicio, fin] desplazada ``anios`` atrás al mismo día del año y hora (placebo de T0). Los
    pasos de la rejilla que no reciben valor quedan sin pasada; un 29 de febrero sin destino se pierde."""
    s = con_pasada.loc[pd.Timestamp(inicio) : pd.Timestamp(fin) + pd.Timedelta(hours=21)]
    destino = s.index - pd.DateOffset(years=anios)
    m = pd.Series(s.to_numpy(bool), index=destino)
    m = m[~m.index.duplicated(keep="first")]
    return m.reindex(rejilla, fill_value=False).to_numpy(bool)


@dataclass
class ClasesPlacebo:
    clase: dict  # nombre → «cuenta», «no informativo» o «sin pasos»
    potencia: dict  # nombre → potencia al nivel de Bonferroni final (solo los que tienen pasos)
    cuentan: list
    m: int


def clasificar_placebos(nombres, con_pasos, potencia, alfa: float, minimo: float) -> ClasesPlacebo:
    """Clases de placebo (§3.4 punto 10). Un placebo sin pasos no cuenta. Los demás cuentan si su potencia frente al
    cociente alternativo, al nivel de Bonferroni 1 − α/m, llega al mínimo; m es el número de los que cuentan. Se
    busca el punto fijo empezando con todos los que tienen pasos (m decrece hasta estabilizarse).

    ``potencia(nombre, m)`` → potencia de ese placebo al nivel 1 − α/m."""
    cuentan = list(con_pasos)
    pot: dict = {}
    for _ in range(len(con_pasos) + 1):
        m = max(len(cuentan), 1)
        pot = {n: potencia(n, m) for n in con_pasos}
        nuevos = [n for n in con_pasos if pot[n] >= minimo]
        if nuevos == cuentan:
            break
        cuentan = nuevos
    clase = {
        n: ("sin pasos" if n not in con_pasos else "cuenta" if n in cuentan else "no informativo") for n in nombres
    }
    return ClasesPlacebo(clase, pot, cuentan, len(cuentan))


# --- Cociente por misión (exploratorio, posterior a ver datos; contrato §20, A1) ---------------
ASIMILADA_TABLA3 = "asimilada según la tabla 3"
FUERA_TABLA3 = "fuera del período de la tabla 3"
NO_ASIMILADA_QUID = "no asimilada según el QUID"
POR_VERIFICAR = "independencia por verificar"


def segmento_quid(
    mision: str, t: pd.DatetimeIndex, asimiladas: dict, no_asimiladas: Sequence[str], no_verificables: Sequence[str]
) -> np.ndarray:
    """Estado de la misión en cada t según el QUID, sin el trato de las no verificables como asimiladas."""
    if mision in no_verificables:
        return np.full(len(t), POR_VERIFICAR, dtype=object)
    if mision in no_asimiladas:
        return np.full(len(t), NO_ASIMILADA_QUID, dtype=object)
    if mision in asimiladas:
        dentro = asimilada_en(mision, t, asimiladas, ())
        return np.where(dentro, ASIMILADA_TABLA3, FUERA_TABLA3).astype(object)
    return np.full(len(t), "sin clasificar", dtype=object)


def esperados_con_cociente_1(evento: np.ndarray, expuesto: np.ndarray, estrato: np.ndarray) -> float:
    """Eventos esperados en los pasos con pasada si el cociente fuera 1: Σ n1·b/n0 en los estratos con los dos
    grupos. Sirve para leer un grupo sin eventos, donde el IC de Greenland y Robins no se puede calcular."""
    e = np.asarray(evento, bool)
    x = np.asarray(expuesto, bool)
    s = np.asarray(estrato, int)
    ok = s >= 0
    e, x, s = e[ok], x[ok], s[ok]
    if s.size == 0:
        return 0.0
    k = int(s.max()) + 1
    b = np.bincount(s, weights=(e & ~x), minlength=k)
    n1 = np.bincount(s, weights=x, minlength=k)
    n0 = np.bincount(s, weights=~x, minlength=k)
    usar = (n1 > 0) & (n0 > 0)
    return float(np.sum(n1[usar] * b[usar] / n0[usar]))


def cocientes_por_mision(
    evento: np.ndarray,
    validos: np.ndarray,
    pasadas: dict[str, np.ndarray],
    hora: np.ndarray,
    nivel: np.ndarray,
    t: pd.DatetimeIndex,
    asimiladas: dict,
    no_asimiladas: Sequence[str],
    no_verificables: Sequence[str],
    nivel_conf: float = 0.95,
) -> pd.DataFrame:
    """Cociente MH de cada misión, por segmento del QUID y por año (contrato §20, A1).

    Expuestos: pasos con pasada **solo de esa misión**; no expuestos: pasos **sin pasada de ninguna misión**. Se
    limita a los pasos válidos entre la primera y la última pasada de la misión. Estratos: hora × nivel de Hs(t−1).
    Exploratorio: conteos, cociente e IC, sin p-valor."""
    nombres = sorted(pasadas)
    if not nombres:
        return pd.DataFrame()
    M = np.column_stack([np.asarray(pasadas[m], bool) for m in nombres])
    cuantas = M.sum(axis=1)
    ninguna = cuantas == 0
    validos = np.asarray(validos, bool)
    anio = np.asarray(t.year)
    filas = []
    for j, m in enumerate(nombres):
        solo = M[:, j] & (cuantas == 1)
        con = np.flatnonzero(M[:, j])
        if con.size == 0:
            continue
        activa = np.zeros(len(t), bool)
        activa[con[0] : con[-1] + 1] = True
        base = validos & activa & (solo | ninguna)
        seg = segmento_quid(m, t, asimiladas, no_asimiladas, no_verificables)
        grupos = [("segmento", s, seg == s) for s in sorted(set(seg[base & solo]))]
        grupos += [("anio", str(y), anio == y) for y in sorted(set(anio[base & solo].tolist()))]
        for agregado, clave, g in grupos:
            v = base & g
            est = codigos_estrato(hora, nivel, validos=v)
            r = mantel_haenszel(evento, solo, est, nivel_conf)
            ix = np.flatnonzero(v & solo)
            filas.append(
                {
                    "mision": m,
                    "agregado": agregado,
                    "clave": clave,
                    "desde": str(t[ix[0]]) if ix.size else "",
                    "hasta": str(t[ix[-1]]) if ix.size else "",
                    "pasos_con": r.pasos_con,
                    "eventos_con": r.eventos_con,
                    "pasos_sin": r.pasos_sin,
                    "eventos_sin": r.eventos_sin,
                    "estratos": r.estratos,
                    "cociente": r.cociente,
                    "ic_lo": r.lo,
                    "ic_hi": r.hi,
                    "eventos_esperados_con_cociente_1": esperados_con_cociente_1(evento, solo, est),
                }
            )
    return pd.DataFrame(filas)
