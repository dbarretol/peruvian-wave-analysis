"""Escenarios del estudio de verificación por simulación de la biblioteca de extremos.

Cada réplica es una función pura de (escenario, índice, repetición): su generador se deriva de esos nombres
(``semillas``), no del orden de ejecución, así que el resultado no depende del número de procesos.

Conteo de la cobertura de un IC, por lado: falla si el valor verdadero queda fuera de ese lado. Un límite abierto
por ``tope`` o por ``frontera`` cubre su lado (el IC llega hasta el borde de la búsqueda o del espacio de
parámetros); uno abierto por ``sin_convergencia`` o ``borde_de_factibilidad`` es desconocido y no cubre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..comun import semillas
from ..extremos import bootstrap, decluster, gpd, perfil, reescalado, retorno
from ..extremos.estratos import Estratificacion
from ..extremos.perfil import FRONTERA, TOPE
from ..extremos.retorno import EstratoRetorno

T_RETORNO = 100.0
ANIOS = 40
PASO_H = 3.0
IET_H = 24.0
B_POISSON = 99
B_BOOT = 199
CAUSAS_QUE_CUBREN = (TOPE, FRONTERA)


@dataclass(frozen=True)
class Escenario:
    id: str
    tipo: str  # "rl", "rl_multi", "poisson_param", "poisson_continua", "boot_ar3h", "boot_tasa_ar", "boot_rl"
    con_banda: bool
    descripcion: str
    params: dict = field(default_factory=dict)


ESCENARIOS: tuple[Escenario, ...] = (
    Escenario(
        "A1", "rl", True, "RL100 y ξ de un estrato, ξ = −0.4", {"xi": -0.4, "sigma": 1.0, "tasa": 10.0, "anios": ANIOS}
    ),
    Escenario(
        "A2", "rl", True, "RL100 y ξ de un estrato, ξ = −0.2", {"xi": -0.2, "sigma": 1.0, "tasa": 10.0, "anios": ANIOS}
    ),
    Escenario(
        "A3", "rl", True, "RL100 y ξ de un estrato, ξ = 0", {"xi": 0.0, "sigma": 1.0, "tasa": 10.0, "anios": ANIOS}
    ),
    Escenario(
        "A4", "rl", True, "RL100 y ξ de un estrato, ξ = 0.1", {"xi": 0.1, "sigma": 1.0, "tasa": 10.0, "anios": ANIOS}
    ),
    Escenario(
        "B",
        "rl_multi",
        True,
        "RL100 con dos estratos",
        {"xi": (-0.2, 0.0), "sigma": (1.0, 0.7), "tasa": (6.0, 4.0), "anios": ANIOS},
    ),
    Escenario("D1", "poisson_param", True, "Reescalado: nula paramétrica con huecos", {"tasas_anio": (30.0, 12.0)}),
    Escenario(
        "D2",
        "poisson_continua",
        True,
        "Reescalado: nula no circular (serie continua declusterizada)",
        {"tasas_anio": (30.0, 12.0), "frac_replica": 0.0},
    ),
    Escenario(
        "E",
        "poisson_continua",
        False,
        "Reescalado: potencia, 30 % de tormentas con réplica",
        {"tasas_anio": (30.0, 12.0), "frac_replica": 0.3},
    ),
    Escenario(
        "I1", "rl", False, "Informativo: n ≈ 40 picos, ξ = 0", {"xi": 0.0, "sigma": 1.0, "tasa": 1.0, "anios": ANIOS}
    ),
    Escenario(
        "I2",
        "rl",
        False,
        "Informativo: n ≈ 40 picos, ξ = −0.2",
        {"xi": -0.2, "sigma": 1.0, "tasa": 1.0, "anios": ANIOS},
    ),
    Escenario(
        "I3",
        "rl",
        False,
        "Informativo: zona no regular, ξ = −0.7",
        {"xi": -0.7, "sigma": 1.0, "tasa": 10.0, "anios": ANIOS},
    ),
    Escenario(
        "BS1", "boot_ar3h", False, "Informativo: bootstrap por años de la media de una AR(1) a 3 h", {"phi": 0.8}
    ),
    Escenario(
        "BS2",
        "boot_tasa_ar",
        False,
        "Informativo: bootstrap por años con dependencia entre años",
        {"phi": 0.5, "sd": 0.3, "tasa": 10.0, "anios": ANIOS},
    ),
    Escenario(
        "BS3",
        "boot_rl",
        False,
        "Informativo: IC percentil del RL100 por años, ξ = −0.2",
        {"xi": -0.2, "sigma": 1.0, "tasa": 10.0, "anios": 30},
    ),
)
POR_ID = {e.id: e for e in ESCENARIOS}


def _cubre(verdadero: float, lo: float, hi: float, causa_lo, causa_hi) -> tuple[bool, bool]:
    """(cubre por abajo, cubre por arriba)."""
    abajo = (verdadero >= lo) if causa_lo is None else (causa_lo in CAUSAS_QUE_CUBREN)
    arriba = (verdadero <= hi) if causa_hi is None else (causa_hi in CAUSAS_QUE_CUBREN)
    return bool(abajo), bool(arriba)


def _rng(esc: str, i: int, repeticion: bool) -> np.random.Generator:
    return semillas.generador("verificacion", esc, *(["repeticion"] if repeticion else []), str(i))


# --- Perfiles ---------------------------------------------------------------------------------


def _replica_rl(p: dict, rng: np.random.Generator) -> dict:
    xi, sigma, lam, E = p["xi"], p["sigma"], p["tasa"], float(p["anios"])
    n = int(rng.poisson(lam * E))
    x = stats.genpareto.rvs(c=xi, scale=sigma, size=n, random_state=rng)
    rl_v = retorno.nivel_retorno(0.0, xi, sigma, lam, T_RETORNO)
    a = gpd.ajustar_gpd(x)
    fila = {"n": n, "valido": a.valido, "xi_hat": a.xi, "frontera": a.frontera}
    if not a.valido or n / E * T_RETORNO <= 1:
        return fila | {"ajuste_fallido": True}
    r = perfil.perfil_rl(x, n, E, T_RETORNO, 0.0, a)
    c = perfil.perfil_xi(x, a)
    rl_ab, rl_ar = _cubre(rl_v, r.lo, r.hi, r.causa_lo, r.causa_hi)
    xi_ab, xi_ar = _cubre(xi, c.lo, c.hi, c.causa_lo, c.causa_hi)
    return fila | {
        "ajuste_fallido": False,
        "rl_hat": r.punto,
        "rl_lo": r.lo,
        "rl_hi": r.hi,
        "rl_causa_lo": r.causa_lo,
        "rl_causa_hi": r.causa_hi,
        "rl_cubre_abajo": rl_ab,
        "rl_cubre_arriba": rl_ar,
        "xi_lo": c.lo,
        "xi_hi": c.hi,
        "xi_causa_lo": c.causa_lo,
        "xi_causa_hi": c.causa_hi,
        "xi_cubre_abajo": xi_ab,
        "xi_cubre_arriba": xi_ar,
        "xi_no_regular": c.no_regular,
    }


def _replica_rl_multi(p: dict, rng: np.random.Generator) -> dict:
    E = float(p["anios"])
    xs, ns, ajustes = [], [], []
    for xi, sigma, lam in zip(p["xi"], p["sigma"], p["tasa"], strict=True):
        n = int(rng.poisson(lam * E))
        xs.append(stats.genpareto.rvs(c=xi, scale=sigma, size=n, random_state=rng))
        ns.append(n)
        ajustes.append(gpd.ajustar_gpd(xs[-1]))
    rl_v = retorno.nivel_retorno_multiestrato(
        0.0, [EstratoRetorno(l_, xi, s) for xi, s, l_ in zip(p["xi"], p["sigma"], p["tasa"], strict=True)], T_RETORNO
    )
    fila = {"n": sum(ns), "valido": all(a.valido for a in ajustes)}
    if not fila["valido"]:
        return fila | {"ajuste_fallido": True}
    r = perfil.perfil_rl_multiestrato(xs, ns, [E] * len(xs), T_RETORNO, 0.0, ajustes)
    ab, ar = _cubre(rl_v, r.lo, r.hi, r.causa_lo, r.causa_hi)
    return fila | {
        "ajuste_fallido": False,
        "rl_hat": r.punto,
        "rl_lo": r.lo,
        "rl_hi": r.hi,
        "rl_causa_lo": r.causa_lo,
        "rl_causa_hi": r.causa_hi,
        "rl_cubre_abajo": ab,
        "rl_cubre_arriba": ar,
        "n_rescates": r.n_rescates,
        "n_sin_resolver": r.n_sin_resolver,
    }


# --- Reescalado --------------------------------------------------------------------------------

_SEMESTRES = Estratificacion.desde({"ene-jun": list(range(1, 7)), "jul-dic": list(range(7, 13))})


def _rejilla() -> pd.DatetimeIndex:
    return pd.date_range("1980-01-01", periods=int(ANIOS * 365.25 * 24 / PASO_H), freq="3h")


def _replica_poisson_param(p: dict, rng: np.random.Generator) -> dict:
    rejilla = _rejilla()
    estr = _SEMESTRES.etiqueta(rejilla)
    codigo = _SEMESTRES.codigo(rejilla)
    validos = np.ones(rejilla.size, bool)
    for inicio in (0.2, 0.5, 0.8):  # tres huecos de 10 días
        k = int(inicio * rejilla.size)
        validos[k : k + 80] = False
    inten = np.array(p["tasas_anio"]) / (365.25 * 24)
    duraciones = [np.arange(0, 12)] * 2  # 0 a 11 celdas tras el inicio
    muertas = int(IET_H / PASO_H)
    ini, fin = reescalado.simular_nula(validos, codigo, inten, duraciones, muertas, PASO_H, rng)
    r = reescalado.prueba_poisson(rejilla, validos, rejilla[ini], rejilla[fin], estr, IET_H, PASO_H, rng, B_POISSON)
    return {
        "n": int(ini.size),
        "p_ad": r.p_ad,
        "p_ks": r.p_ks,
        "rechaza_ad": r.rechaza,
        "rechaza_ks": r.p_ks <= 0.05,
        "generadas": r.generadas,
    }


def poner_tormenta(hs: np.ndarray, i: int, largo: int, rng: np.random.Generator) -> None:
    """Racha continua sobre u = 1 desde la celda ``i``; si llega al final de la serie, se corta allí. Siempre consume
    ``largo`` números del generador, de modo que el corte no altera el resto de la réplica."""
    valores = 1.0 + rng.random(largo)
    tramo = hs[i : i + largo]
    hs[i : i + largo] = np.maximum(tramo, valores[: tramo.size])


def serie_continua(p: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Serie de 3 h sin huecos: tormentas con llegadas Poisson por estrato en tiempo calendario, cada una una racha
    continua sobre u = 1 de 1 a 12 celdas, sin bajadas internas; las que se solapan se funden. Con ``frac_replica``
    > 0, esa fracción de tormentas va seguida de una réplica que empieza U(IET, 2·IET) horas después de su fin."""
    rejilla = _rejilla()
    codigo = _SEMESTRES.codigo(rejilla)
    inten = (np.array(p["tasas_anio"]) / (365.25 * 24))[codigo]
    hs = np.zeros(rejilla.size)
    inicios = np.flatnonzero(rng.random(rejilla.size) < inten * PASO_H)
    for i in inicios:
        largo = int(rng.integers(1, 13))
        poner_tormenta(hs, i, largo, rng)
        if p["frac_replica"] > 0 and rng.random() < p["frac_replica"]:
            fin = i + largo - 1
            j = fin + int(np.ceil(rng.uniform(IET_H, 2 * IET_H) / PASO_H))
            poner_tormenta(hs, j, int(rng.integers(1, 13)), rng)
    return pd.DataFrame({"Hs": hs}, index=rejilla)


def _replica_poisson_continua(p: dict, rng: np.random.Generator) -> dict:
    df = serie_continua(p, rng)
    picos = decluster.decluster_rachas(df, u=0.5, iet_horas=IET_H, paso_horas=PASO_H)
    validos = np.ones(len(df), bool)
    r = reescalado.prueba_poisson(
        df.index,
        validos,
        pd.DatetimeIndex(picos["t_inicio"]),
        pd.DatetimeIndex(picos["t_fin"]),
        _SEMESTRES.etiqueta(df.index),
        IET_H,
        PASO_H,
        rng,
        B_POISSON,
    )
    return {
        "n": int(len(picos)),
        "p_ad": r.p_ad,
        "p_ks": r.p_ks,
        "rechaza_ad": r.rechaza,
        "rechaza_ks": r.p_ks <= 0.05,
        "generadas": r.generadas,
    }


# --- Bootstrap (informativos) ------------------------------------------------------------------------

_ANUAL = Estratificacion.anual()


def _replica_boot_ar3h(p: dict, rng: np.random.Generator) -> dict:
    rejilla = _rejilla()
    phi = p["phi"]
    e = rng.normal(0.0, np.sqrt(1 - phi**2), rejilla.size)
    y = np.empty(rejilla.size)
    y[0] = rng.normal()
    for k in range(1, y.size):
        y[k] = phi * y[k - 1] + e[k]
    serie = pd.Series(y, index=rejilla)
    bloques = bootstrap.bloques_anuales(serie, _ANUAL)
    reps = bootstrap.bootstrap_bloques(
        bloques, lambda bloques_rep: float(np.concatenate([b.to_numpy() for b in bloques_rep]).mean()), B_BOOT, rng
    )
    ic = bootstrap.ic_percentil(reps, float(y.mean()))
    ee_iid = y.std(ddof=1) / np.sqrt(y.size)
    return {
        "bloques_cubre_abajo": ic.lo <= 0.0,
        "bloques_cubre_arriba": 0.0 <= ic.hi,
        "iid_cubre_abajo": y.mean() - 1.96 * ee_iid <= 0.0,
        "iid_cubre_arriba": 0.0 <= y.mean() + 1.96 * ee_iid,
    }


def _replica_boot_tasa_ar(p: dict, rng: np.random.Generator) -> dict:
    phi, sd, n_a = p["phi"], p["sd"], p["anios"]
    var_eta = sd**2 / (1 - phi**2)
    eta = np.empty(n_a)
    eta[0] = rng.normal(0, np.sqrt(var_eta))
    for k in range(1, n_a):
        eta[k] = phi * eta[k - 1] + rng.normal(0, sd)
    cuentas = rng.poisson(p["tasa"] * np.exp(eta - var_eta / 2))  # E[λ_y] = tasa
    serie = pd.Series(cuentas.astype(float), index=pd.date_range("1980-07-01", periods=n_a, freq="12MS"))
    reps = bootstrap.bootstrap_bloques(
        bootstrap.bloques_anuales(serie, _ANUAL),
        lambda bloques_rep: float(np.mean([b.iloc[0] for b in bloques_rep])),
        B_BOOT,
        rng,
    )
    ic = bootstrap.ic_percentil(reps, float(cuentas.mean()))
    return {
        "bloques_cubre_abajo": ic.lo <= p["tasa"],
        "bloques_cubre_arriba": p["tasa"] <= ic.hi,
        "autocorr_lag1": float(np.corrcoef(cuentas[:-1], cuentas[1:])[0, 1]),
    }


def _replica_boot_rl(p: dict, rng: np.random.Generator) -> dict:
    xi, sigma, lam, n_a = p["xi"], p["sigma"], p["tasa"], p["anios"]
    n_por_anio = rng.poisson(lam, n_a)
    excesos = [stats.genpareto.rvs(c=xi, scale=sigma, size=k, random_state=rng) for k in n_por_anio]
    rl_v = retorno.nivel_retorno(0.0, xi, sigma, lam, T_RETORNO)

    def rl(lista) -> float:
        x = np.concatenate(lista)
        a = gpd.ajustar_gpd(x)
        return retorno.nivel_retorno(0.0, a.xi, a.sigma, x.size / n_a, T_RETORNO) if a.valido else np.nan

    a0 = gpd.ajustar_gpd(np.concatenate(excesos))
    punto = rl(excesos)
    reps = bootstrap.bootstrap_bloques(dict(enumerate(excesos)), rl, B_BOOT, rng)
    ic = bootstrap.ic_percentil(reps, punto, no_regular=a0.no_regular)
    return {
        "cubre_abajo": ic.lo <= rl_v,
        "cubre_arriba": rl_v <= ic.hi,
        "n_no_finitas": ic.n_no_finitas,
        "contiene_punto": ic.contiene_punto,
    }


_FUNCIONES = {
    "rl": _replica_rl,
    "rl_multi": _replica_rl_multi,
    "poisson_param": _replica_poisson_param,
    "poisson_continua": _replica_poisson_continua,
    "boot_ar3h": _replica_boot_ar3h,
    "boot_tasa_ar": _replica_boot_tasa_ar,
    "boot_rl": _replica_boot_rl,
}


def correr_replica(esc_id: str, i: int, repeticion: bool = False) -> dict:
    esc = POR_ID[esc_id]
    return {"replica": i} | _FUNCIONES[esc.tipo](esc.params, _rng(esc_id, i, repeticion))
