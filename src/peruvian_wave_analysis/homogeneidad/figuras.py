"""Figuras de la F4 (contrato §2): cuantiles por tramo, indicadores anuales con la cronología, serie anual de
cuantiles altos y exposición a altímetros. Se escriben después de ``potencia.csv`` (G4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .. import estilo


def _lineas(ax, fechas: Sequence[pd.Timestamp], anual: bool) -> None:
    for f in fechas:
        x = f.year + (f.dayofyear - 1) / 365.25 if anual else f
        ax.axvline(x, color="0.6", lw=0.7, ls="--")


def cuantiles_por_tramo(cpt: pd.DataFrame, tramos: Sequence[str], ruta: Path) -> None:
    d = cpt[(cpt["variable"] == "VHM0") & (cpt["estacion"] == "todo") & cpt["grupo"].isin(tramos)]
    fig, axs = estilo.nueva_figura(1, 4, figsize=(12.0, 3.6), layout="constrained")
    for ax, est in zip(axs, ("P50", "P99", "P99.9", "media_maximos"), strict=True):
        g = d[d["estadistico"] == est].set_index("grupo").reindex(tramos)
        x = np.arange(len(tramos))
        ax.errorbar(
            x, g["valor"], yerr=[g["valor"] - g["ic_lo"], g["ic_hi"] - g["valor"]], fmt="o", color="k", capsize=3
        )
        ax.set_xticks(x, tramos)
        ax.set_xlabel("Tramo")
        ax.set_ylabel("Hs (m)")
        ax.set_title("media de máximos anuales" if est == "media_maximos" else est)
    fig.suptitle("Hs por tramo con peso mensual igual (IC del 95 %, bootstrap de años)")
    estilo.guardar_figura(fig, ruta)


def indicadores(ind: pd.DataFrame, fechas: Sequence[pd.Timestamp], ruta: Path) -> None:
    fig, axs = estilo.nueva_figura(3, 1, figsize=(9.0, 8.0), sharex=True, layout="constrained")
    x = ind.index.to_numpy(float) + 0.5
    axs[0].bar(x, ind["excursiones_k_rev"], width=0.8, color="0.4", label="excursiones de k_rev")
    axs[0].plot(x, ind["candidatos_detector"], "o", color="r", ms=3, label="candidatos del detector")
    axs[0].set_ylabel("Número por año (-)")
    axs[0].legend(fontsize=7)
    axs[1].plot(x, ind["variabilidad_3h_p99"], "o-", color="k", ms=3)
    axs[1].set_ylabel("P99 de |ΔHs|/Hs (-)")
    if "discrepancia_energia_mediana" in ind:
        axs[2].plot(x, ind["discrepancia_energia_mediana"], "o-", color="k", ms=3)
    axs[2].set_ylabel("Discrepancia de energía (-)")
    axs[2].set_xlabel("Año (-)")
    for ax in axs:
        _lineas(ax, fechas, anual=True)
    fig.suptitle("Indicadores anuales (líneas: fechas de la cronología)")
    estilo.guardar_figura(fig, ruta)


def cuantiles_anuales(an: pd.DataFrame, fechas: Sequence[pd.Timestamp], ruta: Path) -> None:
    fig, ax = estilo.nueva_figura(figsize=(9.0, 4.0), layout="constrained")
    x = an.index.to_numpy(float) + 0.5
    for c, et in (("hs_p50", "P50"), ("hs_p99", "P99"), ("hs_max", "máximo")):
        ax.plot(x, an[c], "o-", ms=3, label=et)
    _lineas(ax, fechas, anual=True)
    ax.set_xlabel("Año (-)")
    ax.set_ylabel("Hs (m)")
    ax.legend(fontsize=7)
    ax.set_title("Cuantiles anuales de Hs con peso mensual (líneas: cronología)")
    estilo.guardar_figura(fig, ruta)


def exposicion(expo: pd.DataFrame, ruta: Path) -> None:
    m = expo[["asimilada:c", "completa"]].astype(float).resample("MS").mean()
    fig, ax = estilo.nueva_figura(figsize=(9.0, 3.5), layout="constrained")
    ax.plot(m.index, m["asimilada:c"], color="k", lw=0.8, label="con pasada asimilada (ventana centrada)")
    ax.fill_between(m.index, 0, 1, where=m["completa"] < 1, color="r", alpha=0.15, lw=0, label="exposición incompleta")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fecha (UTC)")
    ax.set_ylabel("Fracción de pasos (-)")
    ax.legend(fontsize=7)
    ax.set_title("Exposición a pasadas de altímetros a ≤ 200 km del nodo, por mes")
    estilo.guardar_figura(fig, ruta)
