"""Etapa 3 de la estratificación (contrato de la F6, §4 y §6) con los excesos de los picos provisionales.

Se llama solo después de verificar que la etapa 1 está publicada, con su gate aprobado y la huella de su código sin
cambios (``run.verificar_etapa1``). Usa el α de familia que la etapa 1 congeló.

- **Prueba confirmatoria:** Anderson–Darling de dos muestras (variante de rangos medios) entre los estratos de cada
  candidata, sobre los excesos sobre u_tent de E0 sin corregir, una sola vez, con α por prueba = α_fam / k.
- **Desenlace (§6.1, punto 2):** «establecido» si rechaza; «descartado» si no rechaza y la potencia de la etapa 1
  llega al 80 % con α_fam en ambas bases; si no, «no concluyente».
- **Escenarios (§6.1, puntos 2 y 3; ADR-0028, punto 7):** con «establecido» o «no concluyente», sin estratos y
  estratificado con ξ común. Si algún estrato tiene menos de 30 picos en alguna rama (ADR-0006, G4), el estratificado
  se rotula «estrato con menos de 30 picos» y, según ADR-0029, queda fuera de la envolvente y del valor alto, como
  sensibilidad descriptiva, junto con el modelo de ξ por estrato de la cota alta (ADR-0028, punto 3).
- **Descriptivos (§4):** Anderson–Darling entre meses dentro de cada estrato (meses con < 5 picos unidos al vecino),
  señal de dispersión (1.8), ΔRL100 combinado con ξ por estrato y con ξ común frente al sin estratos, razón de
  verosimilitudes con su valor p asintótico («sin potencia y con tamaño inflado»), por rama.
- **Población mixta (§6.2)** y **covariables (§6.3)**.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..extremos.gpd import ajustar_gpd
from ..extremos.perfil import perfil_rl, perfil_rl_multiestrato
from . import estratificacion as E
from . import picos as P

MIN_PICOS = 30  # ADR-0006
MIN_PICOS_MES = 5
DISPERSION = 1.8
COTA_P = (0.001, 0.25)
T_RL = 100.0
VEINTE = 20
SENAL_GRUPO_MENOR = 4
SEMESTRE_COVARIABLE = 0.75
ESTABLECIDO, DESCARTADO, NO_CONCLUYENTE = "establecido", "descartado", "no concluyente"
ROTULO_MENOS_30 = "estrato con menos de 30 picos"


def ad_k(muestras: Sequence) -> tuple[float, float]:
    """(estadístico normalizado, valor p) de Anderson–Darling de k muestras, variante de rangos medios."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="p-value (floored|capped)", category=UserWarning)
        r = stats.anderson_ksamp([np.asarray(m, float) for m in muestras], variant="midrank")
    return float(r.statistic), float(r.pvalue)


def p_publicado(p: float) -> float | str:
    if p <= COTA_P[0]:
        return f"≤ {COTA_P[0]:g}"
    if p >= COTA_P[1]:
        return f"≥ {COTA_P[1]:g}"
    return p


# --- Prueba y desenlace --------------------------------------------------------------------------------------


def desenlace(rechaza: bool, potencias: Sequence[float]) -> str:
    """§6.1: potencias = potencia de la etapa 1 con α_fam en cada base."""
    if rechaza:
        return ESTABLECIDO
    return DESCARTADO if potencias and min(potencias) >= E.POTENCIA_OBJETIVO else NO_CONCLUYENTE


def prueba(picos: pd.DataFrame, columna: str, estratos: Sequence[str], alfa_por_prueba: float) -> dict:
    grupos = [picos.loc[picos[columna] == s, "exceso"].to_numpy(float) for s in estratos]
    est, p = ad_k(grupos)
    return {
        "estratos": list(estratos), "n": [int(g.size) for g in grupos], "estadistico": est, "p": p,
        "p_publicado": p_publicado(p), "alfa_por_prueba": alfa_por_prueba, "rechaza": bool(p < alfa_por_prueba),
    }  # fmt: skip


def escenarios(desenlace_: str, menos_de_30: bool, excluir_menos_de_30: bool = True) -> list[dict]:
    """Escenarios que salen de la regla de escala, con su papel en la envolvente."""
    out = [{"nombre": "sin_estratos", "envolvente": True, "rotulo": ""}]
    if desenlace_ in (ESTABLECIDO, NO_CONCLUYENTE):
        fuera = menos_de_30 and excluir_menos_de_30
        out.append({
            "nombre": "estratificado_xi_comun", "envolvente": not fuera,
            "rotulo": ROTULO_MENOS_30 if menos_de_30 else "",
            "papel": "sensibilidad descriptiva (ADR-0029)" if fuera else "escenario de la envolvente",
        })  # fmt: skip
    return out


# --- Descriptivos -------------------------------------------------------------------------------------------


def grupos_de_meses(meses: pd.Series, minimo: int = MIN_PICOS_MES) -> dict[int, int]:
    """Mes → grupo: en orden circular desde el primer mes con picos, se juntan meses hasta reunir ``minimo``; un
    resto final con menos se une al último grupo."""
    c = meses.value_counts()
    presentes = [m for m in range(1, 13) if c.get(m, 0) > 0]
    if not presentes:
        return {}
    # empieza después del mayor hueco circular, para no partir un estrato que cruza diciembre–enero
    huecos = [((presentes[(i + 1) % len(presentes)] - m) % 12 or 12, i) for i, m in enumerate(presentes)]
    _, i0 = max(huecos)
    orden = presentes[i0 + 1 :] + presentes[: i0 + 1]
    grupos, actual, n, g = {}, [], 0, 0
    for m in orden:
        actual.append(m)
        n += int(c[m])
        if n >= minimo:
            grupos.update({x: g for x in actual})
            actual, n, g = [], 0, g + 1
    if actual:
        destino = g - 1 if g > 0 else 0
        grupos.update({x: destino for x in actual})
    return grupos


def dentro_de_estratos(picos: pd.DataFrame, columna: str, estratos: Sequence[str]) -> list[dict]:
    out = []
    for s in estratos:
        sub = picos[picos[columna] == s]
        gm = grupos_de_meses(sub["mes"])
        g = sub["mes"].map(gm)
        muestras = [sub.loc[g == k, "exceso"].to_numpy(float) for k in sorted(set(gm.values()))]
        sd = {int(k): float(np.std(m, ddof=1)) if m.size > 1 else float("nan")
              for k, m in zip(sorted(set(gm.values())), muestras, strict=True)}  # fmt: skip
        fila = {"estrato": s, "n": int(len(sub)), "grupos_de_meses": {str(m): int(k) for m, k in gm.items()},
                "sd_por_grupo": sd}  # fmt: skip
        if len(muestras) >= 2:
            est, p = ad_k(muestras)
            fila |= {"estadistico": est, "p_publicado": p_publicado(p)}
        v = np.array([x for x in sd.values() if np.isfinite(x) and x > 0])
        fila["razon_dispersion_max"] = float(v.max() / v.min()) if v.size >= 2 else float("nan")
        fila["senal_dispersion"] = bool(v.size >= 2 and v.max() / v.min() >= DISPERSION)
        out.append(fila)
    return out


def modelos(picos: pd.DataFrame, columna: str, estratos: Sequence[str], u: float, anios: float) -> dict:
    """RL100 sin estratos, con ξ común y con ξ por estrato (ADR-0022); ΔRL100 en % frente al sin estratos; LRT."""
    a, b = (picos.loc[picos[columna] == s, "exceso"].to_numpy(float) for s in estratos)
    la, lb = a.size / anios, b.size / anios
    au = ajustar_gpd(np.r_[a, b])
    co = E.ajustar_xi_comun(a, b)
    a1, a2 = ajustar_gpd(a), ajustar_gpd(b)
    rl_u = E.rl_mezcla([(la + lb, au.xi, au.sigma)], u, T_RL) if au.valido else float("nan")
    rl_c = E.rl_mezcla([(la, co.xi, co.sigma1), (lb, co.xi, co.sigma2)], u, T_RL) if co.valido else float("nan")
    ok = a1.valido and a2.valido
    rl_s = E.rl_mezcla([(la, a1.xi, a1.sigma), (lb, a2.xi, a2.sigma)], u, T_RL) if ok else float("nan")
    lrt = float(max(2.0 * (a1.loglik + a2.loglik - co.loglik), 0.0)) if ok and co.valido else float("nan")
    ic_u = _ic(lambda: perfil_rl(np.r_[a, b], a.size + b.size, anios, T_RL, u, au)) if au.valido else None
    ic_s = (
        _ic(lambda: perfil_rl_multiestrato([a, b], [a.size, b.size], [anios, anios], T_RL, u, [a1, a2])) if ok else None
    )

    def gpd(x):
        return {"valido": x.valido, "xi": x.xi, "sigma": x.sigma, "n": x.n, "causa": x.causa}

    def pct(v):
        return 100.0 * (v / rl_u - 1.0) if np.isfinite(v) and np.isfinite(rl_u) else float("nan")

    return {
        "u": u, "anios": anios, "n": {estratos[0]: int(a.size), estratos[1]: int(b.size)},
        "sin_estratos": gpd(au) | {"rl100": rl_u, "ic_perfil_rl100": ic_u},
        "xi_comun": {"valido": co.valido, "xi": co.xi, "sigma": [co.sigma1, co.sigma2], "rl100": rl_c,
                     "delta_pct": pct(rl_c)},
        "xi_por_estrato": {"ajustes": [gpd(a1), gpd(a2)], "rl100": rl_s, "delta_pct": pct(rl_s),
                           "no_calculable": not ok, "ic_perfil_rl100": ic_s},
        "lrt": {"estadistico": lrt, "p_asintotico": float(stats.chi2.sf(lrt, 1)) if np.isfinite(lrt) else None,
                "rotulo": "sin potencia y con tamaño inflado (§5)"},
    }  # fmt: skip


def _ic(f) -> dict | None:
    """IC 95 % de perfil del RL100 (límites abiertos como None, con su causa); None si el perfil falla."""
    try:
        r = f()
    except (ValueError, FloatingPointError, RuntimeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"punto": r.punto, "lo": r.lo, "hi": r.hi, "causa_lo": r.causa_lo, "causa_hi": r.causa_hi}


def alarma_adr0029(mod: Mapping, efecto_pct: float = E.EFECTO_PCT) -> list[str]:
    """Modelos descriptivos cuyo RL100 supera al sin estratos en más del efecto relevante (aviso al usuario)."""
    out = []
    for clave in ("xi_comun", "xi_por_estrato"):
        d = mod[clave].get("delta_pct", float("nan"))
        if np.isfinite(d) and d > efecto_pct:
            texto = f"{clave}: ΔRL100 = {d:+.1f} % (> {efecto_pct:g} %)"
            ic, ic_u = mod[clave].get("ic_perfil_rl100"), mod.get("sin_estratos", {}).get("ic_perfil_rl100")
            if ic and "lo" in ic:
                texto += f"; IC95 de perfil de este modelo [{ic['lo']:.2f}, {ic['hi']:.2f}] m"
            elif clave == "xi_comun":
                texto += "; sin IC de perfil para ξ común"
            if ic_u and "lo" in ic_u:
                texto += f"; sin estratos {ic_u['punto']:.2f} m [{ic_u['lo']:.2f}, {ic_u['hi']:.2f}]"
            out.append(texto)
    return out


# --- Población mixta y covariables -------------------------------------------------------------------------------


def poblacion_mixta(picos_por_sep: Mapping[str, pd.DataFrame], arco) -> dict:
    """§6.2 con cada separación. Sin arco (C no es bimodal) no hay grupos: población única."""
    if arco is None:
        return {"grupos": None, "senal": False, "veredicto": "población única",
                "motivo": "VPED de los picos no es bimodal (§6.1): no hay grupo menor"}  # fmt: skip
    por = {}
    for nombre, pk in picos_por_sep.items():
        top = pk.nlargest(VEINTE, "Hs")
        menor = P.estrato_C(top["VPED"], arco) == P.C_MENOR
        mayor_es_menor = bool(menor[np.argmax(top["Hs"].to_numpy())]) if len(top) else False
        n_menor = int(menor.sum())
        por[nombre] = {"n_menor_en_20": n_menor, "mayor_en_grupo_menor": mayor_es_menor,
                       "senal": bool(n_menor >= SENAL_GRUPO_MENOR or mayor_es_menor)}  # fmt: skip
    senales = {v["senal"] for v in por.values()}
    return {"grupos": "C", "por_separacion": por, "senal": any(senales), "cambia_entre_separaciones": len(senales) > 1,
            "veredicto": "señal de población mixta" if any(senales) else "población única"}  # fmt: skip


def covariables(picos: pd.DataFrame, origen_swell_sur: bool) -> dict:
    inv = float(np.mean(np.isin(picos["mes"], P.INVIERNO))) if len(picos) else float("nan")
    semestre = max(inv, 1 - inv)
    out = {
        "tiempo": "propuesta (tendencia; viene de la F4)",
        "escalon_2016_03": "propuesta (E2; viene de la F4)",
        "fraccion_picos_invierno": inv,
        "estacion": "propuesta como covariable de la tasa" if semestre > SEMESTRE_COVARIABLE else "no propuesta",
    }
    out["indices_regionales"] = (
        "ENSO (Niño 1+2 u ONI) y SAM: los extremos son swell del sur; la justificación se verifica en la F8"
        if origen_swell_sur
        else "sin propuesta: el origen de los extremos no es swell del sur dominante"
    )
    return out
