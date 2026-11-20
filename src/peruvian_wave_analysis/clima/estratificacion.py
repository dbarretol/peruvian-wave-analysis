"""Mezclas de dos estratos GPD: frontera del efecto relevante, potencia de las pruebas y α de familia.

Contrato de la F6, §4, §5 y §6.1 (v5). Lo usan la etapa 1, con los conteos reales, y la reproducción de la simulación
previa (``simulacion_previa``), sin datos del sitio.

**Mezcla.** Estrato grande con GPD(ξ₁, σ₁) y estrato chico con GPD(ξ₂, r σ₁) y fracción f de los picos, sobre el
umbral u. El RL100 es el de ADR-0022 (Σ λ_s S_s(z − u) = 1/100, ``retorno.nivel_retorno_multiestrato``).

**Frontera (§5.1, §6.1 punto 1).** ΔRL100 = RL100 del modelo sin estratos (una GPD ajustada a la mezcla) frente al
verdadero, en % del verdadero, con una muestra grande (60 000) y la media de 8 semillas. La frontera de σ es el menor r
de 1.1 a 4.0 (paso 0.1) con |media| ≥ 2 %; si no llega, r = 4 rotulado. La de ξ hacia arriba es el menor ξ₂ de
ξ₁ + 0.025 a ξ₁ + 0.3 (paso 0.025); hacia abajo, el mayor ξ₂ de ξ₁ − 0.025 a ξ₁ − 0.2; si no llega, no hay frontera
de ese lado y se rotula.

**Pruebas (§4).** Anderson–Darling de k muestras con la variante de rangos medios fijada (``variant="midrank"``) y la
razón de verosimilitudes ξ común frente a ξ por estrato (χ²₁). **α de familia:** el menor de {0.05, 0.075, 0.10} con
el que todas las candidatas, en ambas bases, llegan al 80 % de potencia con α por prueba = α_fam / k (Bonferroni);
si ninguno, 0.10 y marca G7.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.stats import genpareto

from ..extremos.gpd import ajustar_gpd, sigma_dado_xi
from ..extremos.retorno import EstratoRetorno, nivel_retorno_multiestrato

U_SIM = 3.2  # m, P99 publicado de la F5 (§5)
ANIOS = 46.0
SIGMA1 = 0.5  # m
T_RL = 100.0
N_GRANDE = 60_000
SEMILLAS_FRONTERA = 8
EFECTO_PCT = 2.0
RS = tuple(float(r) for r in np.round(np.arange(1.1, 4.001, 0.1), 2))
PASOS_XI_ARRIBA = tuple(float(d) for d in np.round(np.arange(0.025, 0.3001, 0.025), 3))
PASOS_XI_ABAJO = tuple(float(d) for d in np.round(-np.arange(0.025, 0.2001, 0.025), 3))
ALFAS_FAM = (0.05, 0.075, 0.10)
POTENCIA_OBJETIVO = 0.80
XI_COMUN_COTAS = (-0.9, 1.0)

SIGMA, XI_ARRIBA, XI_ABAJO = "sigma", "xi+", "xi-"


def rl_mezcla(componentes: Sequence[tuple[float, float, float]], u: float = U_SIM, T: float = T_RL) -> float:
    """RL_T (ADR-0022) de una mezcla; ``componentes`` = [(λ por año, ξ, σ), …]. NaN si no hay nivel."""
    return nivel_retorno_multiestrato(u, [EstratoRetorno(lam, xi, s) for lam, xi, s in componentes], T).valor


def muestra(xi1: float, xi2: float, r: float, n1: int, n2: int, rng: np.random.Generator, sigma1: float = SIGMA1):
    a = genpareto.rvs(xi1, scale=sigma1, size=n1, random_state=rng)
    b = genpareto.rvs(xi2, scale=sigma1 * r, size=n2, random_state=rng)
    return a, b


def partir(n: int, f: float) -> tuple[int, int]:
    """(n₁, n₂) con n₂ = round(f n)."""
    n2 = int(round(f * n))
    return n - n2, n2


def delta_grande(xi1: float, f: float, r: float, xi2: float, lam: float, rng: np.random.Generator) -> float:
    """ΔRL100 (%) del modelo sin estratos frente a la mezcla verdadera, con una muestra de ``N_GRANDE``."""
    n1, n2 = partir(N_GRANDE, f)
    a, b = muestra(xi1, xi2, r, n1, n2, rng)
    verdadero = rl_mezcla([(lam * (1 - f), xi1, SIGMA1), (lam * f, xi2, SIGMA1 * r)])
    aj = ajustar_gpd(np.concatenate([a, b]))
    if not aj.valido:
        return float("nan")
    return 100.0 * (rl_mezcla([(lam, aj.xi, aj.sigma)]) / verdadero - 1.0)


def rejilla(tipo: str, xi1: float) -> tuple[float, ...]:
    """Valores de r (σ) o de ξ₂ (forma) que se recorren, en el orden en que se busca la frontera."""
    if tipo == SIGMA:
        return RS
    pasos = PASOS_XI_ARRIBA if tipo == XI_ARRIBA else PASOS_XI_ABAJO
    return tuple(float(np.round(xi1 + d, 3)) for d in pasos)


def parametros(tipo: str, xi1: float, valor: float) -> tuple[float, float]:
    """(r, ξ₂) del punto ``valor`` de la rejilla de ``tipo``."""
    return (valor, xi1) if tipo == SIGMA else (1.0, valor)


@dataclass(frozen=True)
class Frontera:
    tipo: str
    xi1: float
    f: float
    r: float | None  # None si no hay frontera de ese lado
    xi2: float | None
    media_pct: float
    sd_pct: float
    rotulo: str = ""

    def como_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("tipo", "xi1", "f", "r", "xi2", "media_pct", "sd_pct", "rotulo")}


def elegir_frontera(tipo: str, xi1: float, f: float, fila: Sequence[tuple[float, Sequence[float]]]) -> Frontera:
    """Frontera desde la fila de la rejilla: [(valor, [ΔRL100 de cada semilla]), …] en el orden de ``rejilla``."""
    medias = [(v, float(np.nanmean(d)), float(np.nanstd(d))) for v, d in fila]
    for v, m, sd in medias:
        if abs(m) >= EFECTO_PCT:
            r, xi2 = parametros(tipo, xi1, v)
            return Frontera(tipo, xi1, f, r, xi2, m, sd)
    v, m, sd = medias[-1]
    if tipo == SIGMA:
        return Frontera(tipo, xi1, f, v, xi1, m, sd, f"sin frontera hasta r = {v:g}")
    return Frontera(tipo, xi1, f, None, None, m, sd, f"no llega al {EFECTO_PCT:g} % hasta ξ₂ = {v:g}")


# --- Pruebas -------------------------------------------------------------------------------------------


def anderson_darling(a, b) -> float:
    """Valor p de Anderson–Darling de k muestras, variante de rangos medios fijada (scipy lo acota a [0.001, 0.25])."""
    with warnings.catch_warnings():  # la cota del valor p es esperada (§4) y se publica como tal
        warnings.filterwarnings("ignore", message="p-value (floored|capped)", category=UserWarning)
        return float(stats.anderson_ksamp([np.asarray(a), np.asarray(b)], variant="midrank").pvalue)


@dataclass(frozen=True)
class AjusteXiComun:
    valido: bool
    xi: float
    sigma1: float
    sigma2: float
    loglik: float


def ajustar_xi_comun(a, b) -> AjusteXiComun:
    """GPD con ξ común y σ por estrato: perfil en ξ de ℓ₁(ξ, σ̂₁(ξ)) + ℓ₂(ξ, σ̂₂(ξ)) en (−0.9, 1)."""

    def nl(xi):
        s = sigma_dado_xi(a, xi)[1] + sigma_dado_xi(b, xi)[1]
        return -s if np.isfinite(s) else 1e300

    r = optimize.minimize_scalar(nl, bounds=XI_COMUN_COTAS, method="bounded", options={"xatol": 1e-6})
    xi = float(r.x)
    s1, s2 = sigma_dado_xi(a, xi)[0], sigma_dado_xi(b, xi)[0]
    ok = bool(np.isfinite(r.fun) and r.fun < 1e299 and np.isfinite(s1) and np.isfinite(s2))
    return AjusteXiComun(ok, xi, float(s1), float(s2), -float(r.fun) if ok else -np.inf)


def razon_verosimilitudes(a, b, comun: AjusteXiComun | None = None) -> float:
    """2 [ℓ(ξ por estrato) − ℓ(ξ común)]; NaN si algún ajuste falla."""
    a1, a2 = ajustar_gpd(a), ajustar_gpd(b)
    comun = comun or ajustar_xi_comun(a, b)
    if not (a1.valido and a2.valido and comun.valido):
        return float("nan")
    return float(max(2.0 * (a1.loglik + a2.loglik - comun.loglik), 0.0))


def replica_pruebas(xi1, xi2, r, n1, n2, rng, con_lrt: bool = True) -> dict:
    a, b = muestra(xi1, xi2, r, n1, n2, rng)
    out = {"ad_p": anderson_darling(a, b)}
    if con_lrt:
        out["lrt"] = razon_verosimilitudes(a, b)
    return out


def replica_modelos(xi1, xi2, r, n1, n2, rng, anios: float = ANIOS, por_estrato: bool = True) -> dict:
    """RL100 de los modelos sin estratos, con ξ común y (si ``por_estrato``) con ξ por estrato, más las pruebas."""
    a, b = muestra(xi1, xi2, r, n1, n2, rng)
    l1, l2 = n1 / anios, n2 / anios
    au = ajustar_gpd(np.concatenate([a, b]))
    co = ajustar_xi_comun(a, b)
    out = {
        "ad_p": anderson_darling(a, b),
        "unico": rl_mezcla([(l1 + l2, au.xi, au.sigma)]) if au.valido else float("nan"),
        "comun": rl_mezcla([(l1, co.xi, co.sigma1), (l2, co.xi, co.sigma2)]) if co.valido else float("nan"),
    }
    if por_estrato:
        a1, a2 = ajustar_gpd(a), ajustar_gpd(b)
        ok = a1.valido and a2.valido
        out["sep"] = rl_mezcla([(l1, a1.xi, a1.sigma), (l2, a2.xi, a2.sigma)]) if ok else float("nan")
        out["lrt"] = float(max(2.0 * (a1.loglik + a2.loglik - co.loglik), 0.0)) if ok and co.valido else float("nan")
    return out


def tasa_rechazo(p, alfa: float) -> float:
    """Fracción de valores p < α (un NaN cuenta como no rechazo)."""
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.nan_to_num(p, nan=1.0) < alfa))


def rechazo_lrt(lrt, alfa: float) -> float:
    lrt = np.asarray(lrt, dtype=float)
    return float(np.mean(np.nan_to_num(lrt, nan=0.0) > stats.chi2.ppf(1.0 - alfa, 1)))


# --- α de familia ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EleccionAlfa:
    alfa_fam: float
    alcanza: bool  # algún nivel da el 80 % en todos los casos
    k: int
    por_nivel: dict  # α_fam → mínima potencia entre los casos
    bajo_objetivo_con_tope: tuple  # casos (candidata, base) bajo el 80 % aun con α_fam = 0.10: marca G7


def elegir_alfa_fam(
    potencias: Mapping[tuple[str, float], Mapping[float, float]],
    alfas: Sequence[float] = ALFAS_FAM,
    objetivo: float = POTENCIA_OBJETIVO,
) -> EleccionAlfa:
    """``potencias[(candidata, ξ₁)][α_fam]`` = potencia con α por prueba = α_fam / k (k = número de candidatas)."""
    if not potencias:
        raise ValueError("Sin casos de potencia")
    k = len({c for c, _ in potencias})
    por_nivel = {a: float(min(p[a] for p in potencias.values())) for a in alfas}
    tope = max(alfas)
    bajo = tuple(sorted(c for c, p in potencias.items() if p[tope] < objetivo))
    for a in sorted(alfas):
        if por_nivel[a] >= objetivo:
            return EleccionAlfa(a, True, k, por_nivel, bajo)
    return EleccionAlfa(tope, False, k, por_nivel, bajo)
