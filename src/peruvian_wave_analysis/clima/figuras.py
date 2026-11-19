"""Figuras de la F6 (contrato §2) y comprobación de G3: toda figura incluye el máximo de lo que muestra.

``revisar`` recorre los artistas de cada panel (líneas, puntos, barras) y comprueba que el máximo y el mínimo finitos
de sus datos caen dentro de los límites de los ejes. Se llama antes de guardar cada figura; los problemas se suman al
gate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import estilo
from ..extremos import gev as G

PROBLEMAS: list[str] = []


def _ejes_de(a, ax) -> tuple[bool, bool]:
    """Qué coordenadas del artista están en datos: (x, y). axhline/axvline/axvspan mezclan datos y ejes."""
    t = a.get_transform()
    if t == ax.transData:
        return True, True
    if t == ax.get_xaxis_transform():
        return True, False
    if t == ax.get_yaxis_transform():
        return False, True
    return False, False


def _datos(ax) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for a in ax.get_children():
        if not a.get_visible() or a is ax.patch:
            continue
        en_x, en_y = _ejes_de(a, ax)
        if isinstance(a, Line2D):
            x, y = _num(a.get_xdata(orig=True)), _num(a.get_ydata(orig=True))
        elif isinstance(a, PathCollection):
            o = np.asarray(a.get_offsets(), float)
            if not len(o):
                continue
            x, y = o[:, 0], o[:, 1]
        elif isinstance(a, Rectangle):
            x = np.array([a.get_x(), a.get_x() + a.get_width()], float)
            y = np.array([a.get_y(), a.get_y() + a.get_height()], float)
        else:
            continue
        if en_x:
            xs.append(np.asarray(x, float))
        if en_y:
            ys.append(np.asarray(y, float))
    cat = lambda v: np.concatenate(v) if v else np.empty(0)  # noqa: E731
    return cat(xs), cat(ys)


def _num(v):
    v = np.asarray(v)
    if np.issubdtype(v.dtype, np.datetime64):
        return mdates.date2num(v)
    try:
        return v.astype(float)
    except (TypeError, ValueError):
        return mdates.date2num(pd.to_datetime(v))


def revisar(fig, nombre: str) -> list[str]:
    """G3 y estilo (``estilo.revisar_figura``) de una figura."""
    out = []
    for i, ax in enumerate(fig.get_axes()):
        if getattr(ax, "name", "") == "polar":
            continue
        x, y = _datos(ax)
        for eje, v, lim in (("x", x, ax.get_xlim()), ("y", y, ax.get_ylim())):
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            lo, hi = sorted(lim)
            tol = 1e-9 * max(hi - lo, 1e-12)
            if v.max() > hi + tol or v.min() < lo - tol:
                out.append(f"{nombre}, panel {i}: el eje {eje} [{lo:g}, {hi:g}] no incluye [{v.min():g}, {v.max():g}]")
    return out


def _guardar(fig, ruta: Path, categoricos: set[str] | None = None) -> None:
    PROBLEMAS.extend(revisar(fig, ruta.name))
    PROBLEMAS.extend(f"{ruta.name}: {p}" for p in estilo.revisar_figura(fig, categoricos))
    estilo.guardar_figura(fig, ruta)


# --- Figuras ----------------------------------------------------------------------------------------------


def rosa(rosas: pd.DataFrame, ruta: Path) -> None:
    fig, axs = estilo.nueva_figura(1, 2, figsize=(9.0, 4.2), subplot_kw={"projection": "polar"},
                                   layout="constrained")  # fmt: skip
    for ax, (pob, g) in zip(axs, rosas.groupby("poblacion", sort=False), strict=True):
        th = np.deg2rad(g["centro_grados"].to_numpy(float))
        ax.bar(th, g["fraccion"].to_numpy(float), width=np.deg2rad(22.5), color="#4a77a3", edgecolor="k", lw=0.4)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, float(np.nanmax(g["fraccion"])) * 1.08)
        ax.set_title(f"{pob} (n = {int(g['n'].sum())})")
    fig.suptitle("Dirección de procedencia del pico (VPED), fracción por sector de 22.5°")
    PROBLEMAS.extend(f"{ruta.name}: {p}" for p in estilo._textos_fuera(fig))
    estilo.guardar_figura(fig, ruta)


def ciclo_anual(cm: pd.DataFrame, ruta: Path) -> None:
    fig, axs = estilo.nueva_figura(1, 2, figsize=(10.0, 3.8), layout="constrained")
    for rama, g in cm.groupby("rama", sort=False):
        axs[0].plot(g["mes"], g["hs_p99"], "o-", ms=3, label=f"P99, {rama}")
    g = cm[cm["rama"] == cm["rama"].iloc[0]]
    axs[0].plot(g["mes"], g["hs_max"], "k^", ms=4, label="máximo (E0 sin corregir)")
    axs[0].plot(g["mes"], g["hs_p50"], "k--", label="P50 (E0 sin corregir)")
    axs[0].set_xlabel("Mes (-)")
    axs[0].set_ylabel("Hs (m)")
    axs[0].legend(fontsize=6)
    axs[1].plot(g["mes"], g["dir_media"], "o-", color="k")
    axs[1].set_xlabel("Mes (-)")
    axs[1].set_ylabel("Dirección media de VPED (°)")
    fig.suptitle("Ciclo anual")
    _guardar(fig, ruta)


def dominancia(dom: pd.DataFrame, ruta: Path) -> None:
    d = dom[(dom["cortes"] == "1/3–2/3") & (dom["grupo"] == "mes")]
    fig, ax = estilo.nueva_figura(figsize=(8.0, 3.6), layout="constrained")
    x = d["valor"].astype(int).to_numpy()
    abajo = np.zeros(len(d))
    for c, color in (("swell", "#4a77a3"), ("mixto", "#c98500"), ("mar de viento", "#d55181")):
        v = d[c].to_numpy(float)
        ax.bar(x, v, bottom=abajo, color=color, label=c)
        abajo += v
    ax.set_xlabel("Mes (-)")
    ax.set_ylabel("Fracción de estados (-)")
    ax.set_ylim(0, max(1.0, float(abajo.max())) * 1.02)
    ax.legend(fontsize=7)
    ax.set_title("Clase física por mes (cortes 1/3 y 2/3)")
    _guardar(fig, ruta)


def tp_hs(serie: pd.DataFrame, pk: pd.DataFrame, ruta: Path) -> None:
    v = serie.dropna(subset=["VHM0", "VTPK"])
    fig, ax = estilo.nueva_figura(figsize=(7.0, 4.6), layout="constrained")
    ax.plot(v["VTPK"], v["VHM0"], ".", ms=1, color="0.7", label="estados")
    ax.plot(pk["VTPK"], pk["Hs"], "o", ms=3, color="#eb6834", label="picos provisionales")
    ax.set_xlabel("Tp (s)")
    ax.set_ylabel("Hs (m)")
    estilo.asegurar_visible(ax, v["VHM0"], "y")
    ax.legend(fontsize=7)
    ax.set_title("Hs frente a Tp")
    _guardar(fig, ruta)


def acf(r: np.ndarray, tau_e_h: float, ruta: Path) -> None:
    fig, ax = estilo.nueva_figura(figsize=(7.0, 3.6), layout="constrained")
    k = np.arange(r.size) * 3.0 / 24.0
    ax.plot(k, r, color="k")
    ax.axhline(np.exp(-1), color="#eb6834", ls="--", label="1/e")
    ax.axhline(0, color="0.5", lw=0.6)
    if np.isfinite(tau_e_h):
        ax.axvline(tau_e_h / 24, color="#6250d6", ls=":", label=f"τ_e = {tau_e_h:g} h")
    ax.set_xlabel("Desfase (días)")
    ax.set_ylabel("ACF de la anomalía (-)")
    ax.legend(fontsize=7)
    ax.set_title("Autocorrelación de la anomalía estacional de Hs, dentro de cada año")
    _guardar(fig, ruta)


def theta_u(ic: dict, qs, ruta: Path) -> None:
    fig, axs = estilo.nueva_figura(1, 2, figsize=(10.0, 3.6), layout="constrained")
    x = np.array([100 * q for q in qs])
    for ax, pref, et in ((axs[0], "theta_q", "θ (-)"), (axs[1], "T_C_h_q", "T_(C) (h)")):
        p = np.array([ic[f"{pref}{q:g}"]["punto"] for q in qs])
        lo = np.array([ic[f"{pref}{q:g}"]["lo"] for q in qs])
        hi = np.array([ic[f"{pref}{q:g}"]["hi"] for q in qs])
        ax.errorbar(x, p, yerr=[p - lo, hi - p], fmt="o-", color="k", capsize=3)
        ax.plot(x, lo, "_", color="k")
        ax.plot(x, hi, "_", color="k")
        ax.set_xlabel("Cuantil del umbral (%)")
        ax.set_ylabel(et)
    fig.suptitle("Índice extremal y largo de racha por umbral, invierno (IC del 95 %, bootstrap de años)")
    _guardar(fig, ruta)


def huecos(h_inv: np.ndarray, corte_h: float, ruta: Path) -> None:
    fig, ax = estilo.nueva_figura(figsize=(7.0, 3.6), layout="constrained")
    h = np.asarray(h_inv, float)
    h = h[h > 0]
    if h.size:
        bordes = np.logspace(np.log10(h.min() * 0.9), np.log10(h.max() * 1.1), 40)
        ax.hist(h, bordes, color="0.5", edgecolor="k", lw=0.3)
        ax.set_xscale("log")
    if np.isfinite(corte_h):
        ax.axvline(corte_h, color="#eb6834", ls="--", label=f"corte natural {corte_h:.0f} h")
        ax.legend(fontsize=7)
    ax.set_xlabel("Hueco entre rachas (h)")
    ax.set_ylabel("Número de huecos (-)")
    ax.set_title("Huecos entre rachas sobre P99, invierno")
    _guardar(fig, ruta)


def extremograma(rho: np.ndarray, ref: np.ndarray, escala_h: float, ruta: Path) -> None:
    fig, ax = estilo.nueva_figura(figsize=(7.0, 3.6), layout="constrained")
    k = np.arange(rho.size) * 3.0
    ax.plot(k[1:], rho[1:], color="k", label="ρ(k)")
    ax.plot(k[1:], 1.5 * ref[1:], color="#eb6834", ls="--", label="1.5 × referencia estacional")
    if np.isfinite(escala_h):
        ax.axvline(escala_h, color="#6250d6", ls=":", label=f"escala {escala_h:g} h")
    ax.set_yscale("log")
    ax.set_xlabel("Desfase (h)")
    ax.set_ylabel("ρ(k) (-)")
    ax.legend(fontsize=7)
    ax.set_title("Extremograma con t en invierno, u = P99")
    _guardar(fig, ruta)


def gev(tabla: pd.DataFrame, mxs: dict, ruta: Path) -> None:
    fig, axs = estilo.nueva_figura(2, len(mxs), figsize=(3.2 * len(mxs), 6.2), layout="constrained", squeeze=False)
    for j, (rama, mx) in enumerate(mxs.items()):
        f = tabla[tabla["rama"] == rama].iloc[0]
        x = np.sort(mx["hs_max"].to_numpy(float))
        pp = estilo.posiciones_trazado(x.size)
        T_emp = -1.0 / np.log(pp)  # ADR-0022: p = exp(−1/T), igual que la curva
        ax = axs[0, j]
        ax.plot(T_emp, x, "o", ms=3, color="k", label="máximos")
        if f["valido"]:
            T = np.logspace(np.log10(1.05), np.log10(200), 60)
            ax.plot(T, G.cuantil(np.exp(-1.0 / T), f["mu"], f["sigma"], f["xi"]), color="#4a77a3", label="GEV")
            ax.errorbar([100], [f["rl100"]], yerr=[[f["rl100"] - f["rl100_lo"]], [f["rl100_hi"] - f["rl100"]]]
                        if np.isfinite(f["rl100_lo"]) and np.isfinite(f["rl100_hi"]) else None,
                        fmt="s", color="#eb6834", capsize=3, label="RL100 (IC de perfil)")  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("Período de retorno (años)")
        ax.set_ylabel("Hs máxima anual (m)")
        ax.set_title(rama, fontsize=8)
        ax.legend(fontsize=6)
        ax2 = axs[1, j]
        if f["valido"]:
            teo = G.cuantil(pp, f["mu"], f["sigma"], f["xi"])
            ax2.plot(teo, x, "o", ms=3, color="k")
            lim = [min(teo.min(), x.min()), max(teo.max(), x.max())]
            ax2.plot(lim, lim, color="0.5", lw=0.7)
        ax2.set_xlabel("Cuantil de la GEV (m)")
        ax2.set_ylabel("Máximo observado (m)")
    fig.suptitle("GEV de máximos anuales por rama (ADR-0022)")
    _guardar(fig, ruta)


def picos_estrato(pk: pd.DataFrame, ruta: Path) -> None:
    fig, ax = estilo.nueva_figura(figsize=(8.0, 3.8), layout="constrained")
    for est, color in (("invierno", "#4a77a3"), ("verano", "#eb6834")):
        g = pk[pk["estrato_B"] == est]
        ax.plot(g.index, g["Hs"], "o", ms=3, color=color, label=f"{est} (n = {len(g)})")
    ax.set_xlabel("Fecha (-)")
    ax.set_ylabel("Hs del pico (m)")
    ax.legend(fontsize=7)
    ax.set_title("Picos provisionales por estación (candidata B)")
    _guardar(fig, ruta)


def evento(serie: pd.DataFrame, t_pico: pd.Timestamp, u: float, sep_h: float, titulo: str, ruta: Path,
           t_otro: pd.Timestamp | None = None) -> None:  # fmt: skip
    ini, fin = t_pico - pd.Timedelta(hours=72), (t_otro or t_pico) + pd.Timedelta(hours=72)
    d = serie.loc[ini:fin]
    fig, axs = estilo.nueva_figura(3, 1, figsize=(8.0, 6.6), sharex=True, layout="constrained")
    ax = axs[0]
    ax.plot(d.index, d["VHM0"], color="k", label="Hs total")
    for col, color in (("VHM0_WW", "#d55181"), ("VHM0_SW1", "#4a77a3"), ("VHM0_SW2", "#c98500")):
        if col in d:
            ax.plot(d.index, d[col], color=color, lw=0.8, label=col.replace("VHM0_", "Hs "))
    ax.axhline(u, color="#eb6834", ls="--", lw=0.8, label=f"u_tent = {u:.2f} m")
    for t in (t_pico, t_otro):
        if t is not None and t in d.index:
            ax.plot([t], [d.at[t, "VHM0"]], "v", color="#e34948", ms=7)
    ax.axvspan(t_pico, t_pico + pd.Timedelta(hours=sep_h), color="0.9", zorder=0,
               label=f"separación mínima {sep_h:g} h")  # fmt: skip
    ax.set_ylabel("Hs (m)")
    estilo.asegurar_visible(ax, d["VHM0"], "y")
    ax.legend(fontsize=6, ncol=3)
    axs[1].plot(d.index, d["VTPK"], color="k")
    axs[1].set_ylabel("Tp (s)")
    axs[2].plot(d.index, d["VPED"], ".", ms=3, color="k")
    axs[2].set_ylabel("VPED (°)")
    axs[2].set_ylim(0, 360)
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    for a in axs:
        a.set_xlabel("Fecha (-)")
        a.xaxis.set_major_locator(loc)
        a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.suptitle(titulo)
    _guardar(fig, ruta)
