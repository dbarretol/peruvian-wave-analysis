"""Descriptivos de la F6: climatología mensual, dirección, Tp condicional, peralte y bimodalidad circular.

Contrato de la F6, §3.1, §3.6, §3.7 y §6.1 (candidata C). Funciones puras sobre arreglos; la etapa con datos las
aplica a cada rama.

- **Dirección:** grados «desde» (PUM), de 0 a 360. Media circular = atan2(Σ sin, Σ cos); dispersión circular = 1 − R,
  con R la longitud media del vector resultante. Rosa en 16 sectores de 22.5° centrados en 0° (N), 22.5°, …
- **Tp condicional (§3.6):** cuantiles de Tp por intervalos de Hs; los intervalos de la cola se unen desde arriba hasta
  reunir el mínimo, y el último llega al máximo. Tp censurado por la derecha en el tope (`tp_saturado`): un cuantil q
  se publica como «≥ tope» si la fracción censurada del intervalo supera 1 − q.
- **Peralte (§3.7):** Sp = 2π Hs / (g Tp²); límite de DNV-RP-C205 (2017), §3.5.4: 1/15 con Tp ≤ 8 s, 1/25 con
  Tp ≥ 15 s, interpolado linealmente entre ambos.
- **Bimodalidad circular (§6.1, C):** densidad de `VPED` con núcleo de von Mises de ancho equivalente 10° (κ = 1/σ²,
  σ = 10° en radianes) en una rejilla de 0.5°. Las modas son los máximos locales; la masa de cada moda es la de su
  cuenca (entre los mínimos vecinos). Es candidata si hay al menos dos modas y la segunda en masa reúne ≥ 10 % de
  los picos; los cortes son los mínimos que limitan la cuenca de esa segunda moda. Con más de dos modas se avisa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

G = 9.80665
SECTOR = 22.5
CUANTILES_TP = (0.05, 0.25, 0.5, 0.75, 0.95)
ANCHO_VM_GRADOS = 10.0
PASO_REJILLA = 0.5
FRAC_SEGUNDA_MODA = 0.10


# --- Dirección -----------------------------------------------------------------------------------------


def media_circular(grados) -> tuple[float, float]:
    """(dirección media en [0, 360), dispersión circular 1 − R). NaN si no hay datos."""
    a = np.deg2rad(np.asarray(grados, dtype=float))
    a = a[np.isfinite(a)]
    if not a.size:
        return float("nan"), float("nan")
    s, c = np.sin(a).mean(), np.cos(a).mean()
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0), float(1.0 - np.hypot(s, c))


def sector(grados, ancho: float = SECTOR) -> np.ndarray:
    """Índice de sector (0 = N, centrado en 0°) de cada dirección."""
    g = np.asarray(grados, dtype=float)
    return (np.floor(((g + ancho / 2) % 360.0) / ancho)).astype(int)


def rosa(grados, ancho: float = SECTOR) -> pd.DataFrame:
    """Conteo y fracción por sector (todos los sectores, también los vacíos)."""
    g = np.asarray(grados, dtype=float)
    g = g[np.isfinite(g)]
    n = int(round(360 / ancho))
    conteo = np.bincount(sector(g, ancho), minlength=n)
    return pd.DataFrame({
        "sector": np.arange(n), "centro_grados": np.arange(n) * ancho, "n": conteo,
        "fraccion": conteo / g.size if g.size else np.full(n, np.nan),
    })  # fmt: skip


# --- Climatología mensual -------------------------------------------------------------------------------


def clima_mensual(df: pd.DataFrame, hs: str = "VHM0", tp: str = "VTPK", direccion: str = "VPED") -> pd.DataFrame:
    """Por mes calendario: n, media, P50, P90, P99 y máximo de Hs; P50 y P90 de Tp; dirección media y dispersión.

    ``df`` lleva un índice de tiempo y solo los registros válidos.
    """
    filas = []
    for mes, g in df.groupby(df.index.month):
        h, t = g[hs].to_numpy(float), g[tp].to_numpy(float)
        dm, disp = media_circular(g[direccion])
        filas.append({
            "mes": int(mes), "n": int(np.isfinite(h).sum()), "anios": int(g.index.year.nunique()),
            "hs_media": float(np.nanmean(h)), "hs_p50": float(np.nanquantile(h, 0.5)),
            "hs_p90": float(np.nanquantile(h, 0.9)), "hs_p99": float(np.nanquantile(h, 0.99)),
            "hs_max": float(np.nanmax(h)), "tp_p50": float(np.nanquantile(t, 0.5)),
            "tp_p90": float(np.nanquantile(t, 0.9)), "dir_media": dm, "dir_dispersion": disp,
        })  # fmt: skip
    return pd.DataFrame(filas)


# --- Tp condicional -------------------------------------------------------------------------------------


def intervalos_cola(hs, ancho: float, minimo: int) -> np.ndarray:
    """Bordes de los intervalos de Hs de ``ancho``; los de la cola se unen desde arriba hasta tener ≥ ``minimo``
    valores, y el último borde es el máximo (incluido)."""
    h = np.asarray(hs, dtype=float)
    h = h[np.isfinite(h)]
    if not h.size:
        return np.array([])
    lo = np.floor(h.min() / ancho) * ancho
    bordes = list(np.arange(lo, h.max() + ancho, ancho))
    if bordes[-1] <= h.max():
        bordes.append(bordes[-1] + ancho)
    bordes = np.round(np.array(bordes), 10)
    conteo = np.histogram(h, bordes)[0]
    # une desde arriba: el intervalo superior absorbe a los de abajo hasta reunir el mínimo
    k = len(conteo) - 1
    acumulado = conteo[k]
    while acumulado < minimo and k > 0:
        k -= 1
        acumulado += conteo[k]
    out = np.append(bordes[: k + 1], h.max())
    return out


def cuantiles_censurados(tp, censurado, qs=CUANTILES_TP) -> dict[str, object]:
    """Cuantiles de Tp; ``censurado`` marca los valores en el tope. Un cuantil q vale «≥ tope» si la fracción censurada
    supera 1 − q (el tope es el menor Tp censurado)."""
    tp = np.asarray(tp, dtype=float)
    c = np.asarray(censurado, dtype=bool)
    ok = np.isfinite(tp)
    tp, c = tp[ok], c[ok]
    frac = float(c.mean()) if tp.size else float("nan")
    tope = float(tp[c].min()) if c.any() else float("nan")
    out: dict[str, object] = {"n": int(tp.size), "fraccion_censurada": frac}
    for q in qs:
        clave = f"tp_q{int(round(100 * q)):02d}"
        if not tp.size:
            out[clave] = float("nan")
        elif frac > 1 - q:
            out[clave] = f"≥ {tope:g}"
        else:
            out[clave] = float(np.quantile(tp, q))
    return out


def tp_condicional(hs, tp, censurado, ancho: float, minimo: int, qs=CUANTILES_TP) -> pd.DataFrame:
    hs = np.asarray(hs, dtype=float)
    tp = np.asarray(tp, dtype=float)
    censurado = np.asarray(censurado, dtype=bool)
    bordes = intervalos_cola(hs, ancho, minimo)
    filas = []
    for i in range(len(bordes) - 1):
        lo, hi = bordes[i], bordes[i + 1]
        dentro = (hs >= lo) & ((hs <= hi) if i == len(bordes) - 2 else (hs < hi))
        filas.append(
            {"hs_desde": float(lo), "hs_hasta": float(hi), **cuantiles_censurados(tp[dentro], censurado[dentro], qs)}
        )
    return pd.DataFrame(filas)


# --- Peralte --------------------------------------------------------------------------------------------


def peralte(hs, tp) -> np.ndarray:
    tp = np.asarray(tp, dtype=float)
    return 2 * np.pi * np.asarray(hs, dtype=float) / (G * tp**2)


def limite_peralte_dnv(tp) -> np.ndarray:
    """DNV-RP-C205 (2017), §3.5.4: 1/15 con Tp ≤ 8 s, 1/25 con Tp ≥ 15 s, lineal entre ambos."""
    return np.interp(np.asarray(tp, dtype=float), [8.0, 15.0], [1 / 15, 1 / 25])


# --- Bimodalidad circular -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bimodalidad:
    candidata: bool
    n_modas: int
    modas_grados: tuple[float, ...]  # ordenadas por masa, de mayor a menor
    masas: tuple[float, ...]
    cortes_grados: tuple[float, float] | None  # límites de la cuenca de la segunda moda (en sentido horario)
    mas_de_dos_modas: bool
    n: int


def densidad_von_mises(grados, ancho_grados: float = ANCHO_VM_GRADOS, paso: float = PASO_REJILLA):
    """(rejilla en grados, densidad por grado) con núcleo de von Mises de κ = 1/σ², σ = ancho en radianes."""
    a = np.deg2rad(np.asarray(grados, dtype=float))
    a = a[np.isfinite(a)]
    kappa = 1.0 / np.deg2rad(ancho_grados) ** 2
    rejilla = np.arange(0.0, 360.0, paso)
    r = np.deg2rad(rejilla)
    # exp(κ(cos − 1)) evita el desborde de exp(κ cos); la normalización se hace numéricamente
    k = np.exp(kappa * (np.cos(r[:, None] - a[None, :]) - 1.0)).sum(axis=1)
    return rejilla, k / (k.sum() * paso)


def bimodalidad(grados, ancho_grados: float = ANCHO_VM_GRADOS, frac_min: float = FRAC_SEGUNDA_MODA) -> Bimodalidad:
    g = np.asarray(grados, dtype=float)
    g = g[np.isfinite(g)]
    if g.size < 2:
        return Bimodalidad(False, 0, (), (), None, False, int(g.size))
    rejilla, d = densidad_von_mises(g, ancho_grados)
    izq, der = np.roll(d, 1), np.roll(d, -1)
    tol = 1e-12 * d.max()
    maximos = np.flatnonzero((d > izq + tol) & (d >= der - tol))
    minimos = np.flatnonzero((d < izq - tol) & (d <= der + tol))
    if maximos.size < 2 or minimos.size < 2:
        moda = (float(rejilla[maximos[0]]),) if maximos.size else ()
        return Bimodalidad(False, int(maximos.size), moda, (1.0,) if moda else (), None, False, int(g.size))
    # cuenca de cada máximo: del mínimo anterior al siguiente (circular)
    masas, cuencas = [], []
    for m in maximos:
        antes = minimos[minimos < m]
        despues = minimos[minimos > m]
        a = antes[-1] if antes.size else minimos[-1]
        b = despues[0] if despues.size else minimos[0]
        idx = np.arange(a, b + (len(d) if b < a else 0)) % len(d)
        masas.append(float(d[idx].sum() * PASO_REJILLA))
        cuencas.append((float(rejilla[a]), float(rejilla[b])))
    orden = np.argsort(masas)[::-1]
    masas_o = tuple(masas[i] for i in orden)
    modas_o = tuple(float(rejilla[maximos[i]]) for i in orden)
    candidata = masas_o[1] >= frac_min
    return Bimodalidad(
        bool(candidata), int(maximos.size), modas_o, masas_o, cuencas[orden[1]], bool(maximos.size > 2), int(g.size)
    )


def en_arco(grados, desde: float, hasta: float) -> np.ndarray:
    """True si la dirección cae en el arco [desde, hasta) recorrido en sentido creciente (circular)."""
    g = np.asarray(grados, dtype=float) % 360.0
    if desde <= hasta:
        return (g >= desde) & (g < hasta)
    return (g >= desde) | (g < hasta)
