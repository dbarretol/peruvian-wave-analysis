"""IET\\*: barrido y regla, ``eventos-iet`` (contrato de la F7, §3.3; tolerancia calibrada en §5.3).

En E0 sin corregir, para cada IET de la rejilla (42 h solo como vecino de abajo de 48 h):

1. se identifican los eventos con §3.2 (u_id = P95 de la F3, bajada intermedia de 0.5 m) y se toman los de pico en
   el período del modelo (hasta el 2025-12-31, ADR-0031);
2. se ajusta el modelo de §0 (GPD sin estratos y Poisson) a los picos sobre los tres umbrales de referencia, P98, P99
   y P99.5 de los estados de E0 sin corregir **del período del modelo**, con peso mensual;
3. se publican, como descriptivos (precisión del 2026-11-20): ξ̂ con su IC de perfil, σ\\* = σ̂ − ξ̂u con su IC de
   perfil (``perfil_sigma_mod``), el RL100 con su IC de perfil, el número de picos sobre cada umbral, los eventos y
   cuántos salen de rachas divididas, y cuántos de los fusionados y dudosos de la F6 quedan separados.

**Regla (§3.3), con las mismas funciones que la calibración de §5.3** (``ruido_iet.cambios`` y ``ruido_iet.regla_iet``):
IET\\* es el menor IET ≥ 48 h cuyo RL100 cambia ≤ tolerancia frente a cada vecino en los tres umbrales. La tolerancia
se lee de ``simulacion-previa/ruido-iet/ruido_iet.json`` (1 %: ningún proceso superó el 20 %). Si ninguno cumple, el
de menor cambio máximo con la marca «IET sensible». Se publica el par (IET\\* y el siguiente de la rejilla).

**Cuantil del sitio en cada proceso de §5.3 (descriptivo; fila 21 del contrato):** para cada IET, la fracción de
réplicas de cada proceso con cambio máximo ≤ el del sitio. No reclasifica nada ni cambia el IET\\*.

Los diagnósticos de independencia por IET (§3.5, también en E1 sin corregir, P11) van en el bloque de diagnósticos.

Salida en ``processed/F07-eventos/iet/marcona-200m/`` (carpeta hermana de la corrida final): ``barrido_iet.csv``,
``iet.json`` y ``figuras/``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .. import rutas
from ..comun import etapa, json_estricto, orquestador
from ..comun.errores import GateFallido
from ..datos import carga
from ..datos.run import ruta_candidatos
from ..extremos import perfil as PF
from ..extremos.gpd import XI_MIN, AjusteGPD, loglik
from ..homogeneidad.cuantiles import cuantiles_mensuales
from . import conteo_previo as CP
from . import identificacion as ID
from . import ruido_iet as RI
from . import umbral as U

ETAPA = orquestador.Etapa("eventos-iet", Path(__file__).resolve(), Path("."))
FASE = "F07-eventos"
NOMBRE = CP.NOMBRE
IETS_H = RI.IETS_H  # 42, 48, …, 120: la misma rejilla que la calibración
Q_REFERENCIA = RI.Q_REFERENCIA  # P98, P99, P99.5
T_RL = 100.0
NIVEL = 0.95
CLASES_F6 = ("fusionado", "dudoso")


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "iet" / NOMBRE


def ruta_ruido(raiz: Path) -> Path:
    return Path(raiz) / "processed" / FASE / "simulacion-previa" / "ruido-iet"


def ruta_f06(raiz: Path) -> Path:
    return Path(raiz) / "processed" / "F06-clima-de-oleaje" / NOMBRE


# --- IC de perfil de σ* -------------------------------------------------------------------------------


def _loglik_sigma_mod(x: np.ndarray, u: float, sigma_mod: float) -> float:
    """max_ξ ℓ(ξ, σ = σ* + ξu) con ξ ∈ [−1, 1.5] y el soporte respetado."""
    xmax = float(x.max())
    hi = PF.XI_MAX
    # σ = σ* + ξu > 0 y, con ξ < 0, soporte σ + ξ·xmax > 0, es decir σ* + ξ(u + xmax) > 0 (u > 0, xmax > 0)
    cota = -sigma_mod / (u + xmax) if sigma_mod > 0 else -sigma_mod / u
    lo = max(XI_MIN, cota + 1e-12)
    if lo >= hi:
        return -np.inf

    def f(xi):
        s = sigma_mod + xi * u
        v = loglik(x, xi, s) if s > 0 else -np.inf
        return -v if np.isfinite(v) else 1e300

    rej = np.linspace(lo, hi, 61)
    vals = np.array([f(t) for t in rej])
    k = int(np.argmin(vals))
    a, b = rej[max(k - 1, 0)], rej[min(k + 1, rej.size - 1)]
    r = optimize.minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    best = min(float(r.fun), float(vals[k]))
    return -best if best < 1e299 else -np.inf


def perfil_sigma_mod(excesos, u: float, ajuste: AjusteGPD, nivel: float = NIVEL) -> PF.IntervaloPerfil:
    """IC por perfil de σ* = σ − ξu: para cada σ* fijo se maximiza en ξ. Topes: σ̂* ± 20·σ̂."""
    x = np.asarray(excesos, dtype=float)
    q = float(stats.chi2.ppf(nivel, 1))
    l_hat = ajuste.loglik
    punto = ajuste.sigma - ajuste.xi * u

    def deficit(sm: float) -> float:
        v = _loglik_sigma_mod(x, u, sm)
        return 2.0 * (l_hat - v) if np.isfinite(v) else np.inf

    ancho = PF.FACTOR_TOPE * ajuste.sigma
    hi, c_hi = PF._limite(deficit, punto, q, punto + ancho)
    lo, c_lo = PF._limite(deficit, punto, q, punto - ancho)
    return PF.IntervaloPerfil(punto, lo, hi, c_lo, c_hi, nivel, l_hat)


# --- Barrido ------------------------------------------------------------------------------------------


def umbrales_referencia(hs_periodo: pd.Series) -> np.ndarray:
    return cuantiles_mensuales(hs_periodo, list(Q_REFERENCIA))


def ajuste_descriptivo(picos: np.ndarray, u: float, anios: float) -> dict:
    """Ajuste en u con ξ̂, σ*, RL100 e IC de perfil (los IC fallidos quedan NaN con su causa)."""
    a = U.ajustar_en(picos, u, anios, (T_RL,))
    exc = picos[picos > u] - u
    fila = {"u_m": u, "n_sobre_u": a.n, "lambda_por_anio": a.tasa_por_anio, "valido": bool(a.valido),
            "xi": np.nan, "xi_lo": np.nan, "xi_hi": np.nan, "sigma": np.nan, "sigma_mod": np.nan,
            "sigma_mod_lo": np.nan, "sigma_mod_hi": np.nan, "rl100": a.rl_valor(T_RL), "rl100_lo": np.nan,
            "rl100_hi": np.nan, "causa_ic": ""}  # fmt: skip
    if not a.valido:
        fila["causa_ic"] = "ajuste_no_valido"
        return fila
    causas = []
    fila |= {"xi": a.ajuste.xi, "sigma": a.ajuste.sigma, "sigma_mod": a.sigma_mod}
    px = PF.perfil_xi(exc, a.ajuste, NIVEL)
    fila |= {"xi_lo": px.lo, "xi_hi": px.hi}
    causas += [f"xi:{c}" for c in (px.causa_lo, px.causa_hi) if c]
    ps = perfil_sigma_mod(exc, u, a.ajuste, NIVEL)
    fila |= {"sigma_mod_lo": ps.lo, "sigma_mod_hi": ps.hi}
    causas += [f"sigma_mod:{c}" for c in (ps.causa_lo, ps.causa_hi) if c]
    try:
        pr = PF.perfil_rl(exc, a.n, anios, T_RL, u, a.ajuste, NIVEL)
        fila |= {"rl100_lo": pr.lo, "rl100_hi": pr.hi}
        causas += [f"rl100:{c}" for c in (pr.causa_lo, pr.causa_hi) if c]
    except ValueError as exc_:
        causas.append(f"rl100:{exc_}")
    fila["causa_ic"] = ";".join(causas)
    return fila


def separados_f6(eventos: pd.DataFrame, revision: pd.DataFrame, hs: np.ndarray, u_tent: float) -> dict:
    """Casos fusionados y dudosos de la revisión visual de la F6 que la F7 deja separados.

    Cada caso de la F6 es **un** evento suyo (u_tent = su P99 de E0 sin corregir, separación de 48 h) con dos o tres
    rachas sobre u_tent (tipo «casi partidos»; los de tipo «mayor» repiten casos y se cuentan una vez por ``t_pico``).
    En la F7, el caso queda **separado** si, dentro de la racha de la F7 que contiene su pico, los estados sobre u_tent
    caen en dos o más eventos de la F7 (división por la bajada intermedia).
    """
    out = {}
    tiempos = pd.DatetimeIndex(eventos["t_inicio"]), pd.DatetimeIndex(eventos["t_fin"])
    for clase in CLASES_F6:
        sub = revision[revision["clasificacion_visual"] == clase].drop_duplicates("t_pico")
        sep, sin_evento = 0, 0
        for r in sub.itertuples(index=False):
            t = pd.Timestamp(r.t_pico)
            k = np.flatnonzero((tiempos[0] <= t) & (t <= tiempos[1]))
            if not k.size:
                sin_evento += 1
                continue
            partes = eventos[eventos["id_racha"] == eventos["id_racha"].iloc[k[0]]]
            con_tent = 0
            for a, b in zip(partes["i_inicio"], partes["i_fin"], strict=True):
                seg = hs[int(a) : int(b) + 1]
                con_tent += int(np.any(np.nan_to_num(seg, nan=-np.inf) > u_tent))
            sep += int(con_tent >= 2)
        out[clase] = {"casos": int(len(sub)), "separados": sep, "pico_fuera_de_eventos": sin_evento}
    return out


def barrido_iet(
    hs0: pd.Series, u_id: float, revision: pd.DataFrame, u_tent: float, iets: Sequence[float] = IETS_H
) -> tuple:
    """(tabla larga IET × umbral, matriz RL100 (IET, umbral), umbrales de referencia, años de exposición)."""
    inicio, fin = pd.Timestamp(hs0.index[0]), CP.FIN_MODELO
    periodo = hs0[(hs0.index >= inicio) & (hs0.index <= fin)]
    us = umbrales_referencia(periodo)
    anios = CP.anios_exposicion(hs0, inicio, fin)
    filas, rl = [], np.full((len(iets), len(us)), np.nan)
    v = hs0.to_numpy(dtype=float)
    for j, iet in enumerate(iets):
        ev = ID.en_periodo(ID.identificar(hs0, u_id, iet), inicio, fin)
        picos = v[ev["i_pico"].to_numpy()]
        sep = separados_f6(ev, revision, v, u_tent)
        for k, (q, u) in enumerate(zip(Q_REFERENCIA, us, strict=True)):
            f = ajuste_descriptivo(picos, float(u), anios)
            rl[j, k] = f["rl100"]
            filas.append({"iet_h": iet, "cuantil_referencia": q, "eventos": int(len(ev)),
                          "eventos_de_rachas_divididas": int(ev["dividido_por_bajada"].sum()),
                          "fusionados_f6_separados": sep["fusionado"]["separados"],
                          "dudosos_f6_separados": sep["dudoso"]["separados"]} | f)  # fmt: skip
    return pd.DataFrame(filas), rl, us, anios


def cuantil_en_procesos(maximos_sitio: np.ndarray, replicas: pd.DataFrame, iets: Sequence[float] = IETS_H) -> dict:
    """Fracción de réplicas de cada proceso con cambio máximo ≤ el del sitio, por IET (descriptivo; fila 21)."""
    out = {}
    for p, g in replicas.groupby("proceso", sort=True):
        out[p] = {}
        for j, iet in enumerate(iets):
            x = g.loc[np.isclose(g["iet_h"], iet), "maximo_cambio"].to_numpy(dtype=float)
            out[p][f"{iet:g}"] = float(np.mean(x <= maximos_sitio[j])) if x.size else None
    return out


def figuras(tabla: pd.DataFrame, iet_estrella: float, carpeta: Path) -> list[str]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    carpeta.mkdir(parents=True, exist_ok=True)
    hechas = []
    paneles = (("rl100", "RL100 (m)"), ("xi", "ξ̂"), ("sigma_mod", "σ* = σ̂ − ξ̂u (m)"))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, etiqueta) in zip(axs, paneles, strict=True):
        for q, g in tabla.groupby("cuantil_referencia", sort=True):
            g = g.sort_values("iet_h")
            ax.plot(g["iet_h"], g[col], marker="o", label=f"P{100 * q:g}")
            ax.fill_between(g["iet_h"], g[f"{col}_lo"], g[f"{col}_hi"], alpha=0.12)
        ax.axvline(iet_estrella, color="k", ls="--", lw=0.8)
        ax.set_xlabel("IET (h)")
        ax.set_ylabel(etiqueta)
    axs[0].legend(title="Umbral de referencia")
    fig.suptitle("E0 sin corregir: RL100, ξ̂ y σ* frente al IET, con IC de perfil del 95 % (descriptivo)")
    fig.tight_layout()
    fig.savefig(carpeta / "rl100_xi_sigma_frente_al_iet.png", dpi=130)
    plt.close(fig)
    hechas.append("rl100_xi_sigma_frente_al_iet.png")
    return hechas


def ejecutar(raiz_datos=None) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = CP.rutas_insumos(raiz, NOMBRE)
    d6, dr = ruta_f06(raiz), ruta_ruido(raiz)
    etapa.exigir_publicada(d6)
    etapa.exigir_publicada(dr)
    ins = {
        "serie_nodo": ins["serie_nodo"],
        "resumen_candidatos": ruta_candidatos(raiz, NOMBRE) / "resumen_candidatos.json",
        "declustering_dudoso": d6 / "declustering_dudoso.csv",
        "resumen_f06": d6 / "resumen_f06.json",
        "ruido_iet": dr / "ruido_iet.json",
        "replicas_ruido_iet": dr / "replicas.csv",
    }
    etapa.exigir_insumos(ins)
    serie = carga.cargar_serie(raiz, NOMBRE)
    if "tiempo" in serie.columns:
        serie = serie.set_index("tiempo")
    hs0 = serie["VHM0"].astype(float)
    u_id = float(json_estricto.leer(ins["resumen_candidatos"])["p95_hs_m"])
    tol = float(json_estricto.leer(ins["ruido_iet"])["tolerancia"]["tolerancia"])
    revision = pd.read_csv(ins["declustering_dudoso"])
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        u_tent = float(json_estricto.leer(ins["resumen_f06"])["u_tent_m"]["E0_sin_corregir"])
        tabla, rl, us, anios = barrido_iet(hs0, u_id, revision, u_tent)
        maximos = RI.cambios(rl)
        iet_est, sensible = RI.regla_iet(maximos, IETS_H, tol)
        candidatos = [float(x) for x in IETS_H if x >= RI.IET_MIN_H]
        j = candidatos.index(iet_est)
        siguiente = candidatos[j + 1] if j + 1 < len(candidatos) else None
        tabla["maximo_cambio_rl100"] = tabla["iet_h"].map(dict(zip(IETS_H, maximos, strict=True)))
        tabla.to_csv(st / "barrido_iet.csv", index=False, float_format="%.6g", lineterminator="\n")
        cuantiles = cuantil_en_procesos(maximos, pd.read_csv(ins["replicas_ruido_iet"]))
        figs = figuras(tabla, iet_est, st / "figuras")
        nulos = tabla.loc[~np.isfinite(tabla["rl100"]), ["iet_h", "cuantil_referencia"]].to_dict(orient="records")
        res = {
            "u_id_m": u_id,
            "umbrales_referencia_m": {f"{q:g}": float(u) for q, u in zip(Q_REFERENCIA, us, strict=True)},
            "anios_exposicion_e0": anios,
            "rejilla_h": list(IETS_H),
            "tolerancia": tol,
            "maximo_cambio_rl100": {f"{i:g}": float(m) for i, m in zip(IETS_H, maximos, strict=True)},
            "iet_estrella_h": iet_est,
            "iet_sensible": bool(sensible),
            "par": {"iet_estrella_h": iet_est, "siguiente_h": siguiente,
                    "nota": "El siguiente es sensibilidad obligatoria de la F9, no escenario (§3.3)."},  # fmt: skip
            "cuantil_del_sitio_en_procesos_5_3": {
                "rotulo": "Descriptivo (fila 21): el IET* lo decide la regla del 1 %; ningún cuantil reclasifica nada.",
                "por_proceso_y_iet": cuantiles,
            },
            "figuras": figs,
            "rl100_nulos": nulos,
            "nota_umbrales": "P98, P99 y P99.5 de los estados de E0 sin corregir del período del modelo, con peso "
                             "mensual (§3.3). La calibración de §5.3 usó cuantiles sin peso sobre la serie sintética.",
        }  # fmt: skip
        json_estricto.escribir(st / "iet.json", res)
        gate = etapa.Gate(True, {"iet_estrella_h": iet_est, "iet_sensible": bool(sensible)}, [])
        return etapa.publicar(st, salida, gate, raices, ins, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-iet", description="Barrido y regla del IET de la F7 (§3.3)")
    ap.add_argument("--datos")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = ejecutar(a.datos)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
