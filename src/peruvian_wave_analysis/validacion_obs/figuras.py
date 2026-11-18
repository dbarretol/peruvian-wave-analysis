"""Figuras de la etapa 3 (contrato v6, §2): Q-Q por lado con los máximos, cocientes por cuantil con IC, mapa de las
colocaciones y cociente frente a la distancia a la costa. PNG deterministas (``estilo.guardar_figura``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import estilo  # noqa: E402
from ..homogeneidad import cuantiles as Q  # noqa: E402
from ..validacion import config as CV  # noqa: E402

COLOR_MODELO = "#6250d6"
COLOR_OBS = "#eb6834"


def qq(lados: dict[str, tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]], ruta: Path) -> Path:
    """Un panel por lado: cuantiles con peso mensual del modelo frente a los de la observación, y medias de los
    máximos mensuales."""
    estilo.aplicar()
    n = len(lados)
    cols = min(4, n)
    filas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(3.2 * cols, 3.2 * filas), squeeze=False, layout="constrained")
    qs = np.r_[np.linspace(0.01, 0.99, 99), 0.995, 0.999]
    for ax, (nombre, (t, m, o)) in zip(axs.flat, lados.items(), strict=False):
        w = Q.pesos_mensuales(t)
        qm, qo = Q.cuantil_ponderado(m, w, qs), Q.cuantil_ponderado(o, w, qs)
        ax.plot(qo, qm, ".", ms=3, color=COLOR_MODELO, label="cuantiles")
        mes = pd.Series(t.year * 12 + t.month)
        mx_m = pd.Series(m).groupby(mes.to_numpy()).max()
        mx_o = pd.Series(o).groupby(mes.to_numpy()).max()
        ax.plot(mx_o, mx_m, "x", ms=3, color=COLOR_OBS, label="máximos mensuales")
        lim = [0, float(np.nanmax([qo.max(), qm.max(), mx_o.max(), mx_m.max()])) * 1.05]
        ax.plot(lim, lim, "-", lw=0.8, color=estilo.TINTA_TENUE)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(nombre)
        ax.set_xlabel("Hs observada (m)")
        ax.set_ylabel("Hs del modelo (m)")
    for ax in list(axs.flat)[n:]:
        ax.set_visible(False)
    axs.flat[0].legend(loc="upper left", fontsize=7)
    return estilo.guardar_figura(fig, ruta)


def cocientes(tabla: pd.DataFrame, ruta: Path) -> Path:
    """R(q) descontado con su IC del 95 % por lado principal. ``tabla``: filas de ``sesgo_por_cuantil.csv``."""
    estilo.aplicar()
    ests = ["P50", "P90", "P99", "P99.9", "max", "cola"]
    lados = list(dict.fromkeys(tabla["lado"]))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ancho = 0.8 / max(1, len(lados))
    for j, lado in enumerate(lados):
        x = tabla[tabla["lado"] == lado].set_index("estadistico")
        for k, e in enumerate(ests):
            if e not in x.index or not bool(x.loc[e, "estimable"]):
                continue
            r, lo, hi = x.loc[e, "R_descontado"], x.loc[e, "R_ic95_lo"], x.loc[e, "R_ic95_hi"]
            pos = k - 0.4 + ancho * (j + 0.5)
            ax.errorbar(pos, r, yerr=[[r - lo], [hi - r]], fmt="o", ms=3, capsize=2, label=lado if k == 0 else None)
    ax.axhline(1.0, lw=0.8, color=estilo.TINTA_TENUE)
    ax.set_xticks(range(len(ests)), ["P50", "P90", "P99", "P99.9", "máximos", "cola combinada"])
    ax.set_ylabel("Cociente modelo / observación")
    ax.legend(ncol=4, fontsize=7)
    return estilo.guardar_figura(fig, ruta)


def mapa(so: pd.DataFrame, ruta: Path) -> Path:
    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    x = so[so["costa_min_km"] == CV.COSTA_MIN_KM]
    for mision, g in x.groupby("mision", sort=True):
        ax.plot(g["lon"], g["lat"], ".", ms=1.5, label=f"{mision} ({len(g)})")
    ax.plot(CV.NODO[1], CV.NODO[0], "*", ms=12, color=estilo.TINTA, label="nodo")
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.set_aspect(1 / np.cos(np.radians(CV.NODO[0])))
    ax.legend(fontsize=7, markerscale=4)
    return estilo.guardar_figura(fig, ruta)


def distancia(grad: dict, ruta: Path) -> Path:
    """B(P90) por banda de distancia a la costa, la recta y el valor extrapolado a 23 km (§3.7)."""
    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    b = pd.DataFrame(grad.get("bandas", []))
    if not b.empty:
        ok = b["estimable"].astype(bool)
        ax.plot(b.loc[ok, "dist_media_km"], b.loc[ok, "B_P90"], "o", color=COLOR_OBS, label="bandas estimables")
        ax.plot(
            b.loc[~ok, "dist_media_km"], b.loc[~ok, "B_P90"], "o", mfc="none", color=COLOR_OBS, label="no estimables"
        )
    if grad.get("B_P90_23km") is not None and not b.empty:
        x = np.array([23.0, float(b["dist_media_km"].max())])
        y = grad["B_P90_23km"] + grad["pendiente_por_km"] * (x - 23.0)
        ax.plot(x, y, "-", color=COLOR_MODELO, label="recta ponderada")
        ax.plot(23.0, grad["B_P90_23km"], "s", color=COLOR_MODELO, label="extrapolado al nodo (23 km)")
    ax.axhline(0.0, lw=0.8, color=estilo.TINTA_TENUE)
    ax.set_xlabel("Distancia a la costa (km)")
    ax.set_ylabel("B(P90) = log(modelo / observación)")
    ax.legend(fontsize=7)
    return estilo.guardar_figura(fig, ruta)
