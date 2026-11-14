"""Mapa de selección de sitio: batimetría, isóbatas configurables y malla de WAVERYS.

Uso mínimo::

    from peruvian_wave_analysis.sitio import ConfigMapa, generar_mapa
    r = generar_mapa(ConfigMapa(nombre="marcona", latitud=-15.4, longitud=-75.3))
    r.tabla.head()

Salidas en ``WAVE-DATA/interim/F01-sitio-y-fuentes/mapa-sitio/<nombre>/``:
``mapa.png``, ``candidatos.csv`` y ``metadatos.json`` (configuración, origen y hash de los insumos).
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .. import __version__, estilo, rutas
from . import batimetria, candidatos, waverys
from .config import ConfigMapa

FASE = "F01-sitio-y-fuentes"
NIVELES_PROFUNDIDAD = [0, 25, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 11000]
_HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]


@dataclass
class ResultadoMapa:
    figura: matplotlib.figure.Figure
    tabla: pd.DataFrame
    batimetria: batimetria.Batimetria
    malla: waverys.Malla
    salidas: dict[str, Path] = field(default_factory=dict)


def _fmt_lon(v, _pos=None) -> str:
    return f"{abs(v):.1f}° {'O' if v < 0 else 'E'}"


def _fmt_lat(v, _pos=None) -> str:
    return f"{abs(v):.1f}° {'S' if v < 0 else 'N'}"


def _barra_de_escala(ax, cfg: ConfigMapa) -> None:
    o, e, s, n = cfg.recuadro
    ancho_km = (e - o) * 111.32 * np.cos(np.radians(cfg.latitud))
    largo = min((5, 10, 20, 25, 50, 100, 200), key=lambda L: abs(L - ancho_km / 5))
    largo_grados = largo / (111.32 * np.cos(np.radians(cfg.latitud)))
    x0, y0 = o + 0.05 * (e - o), s + 0.05 * (n - s)
    ax.plot([x0, x0 + largo_grados], [y0, y0], color=estilo.TINTA, lw=3, solid_capstyle="butt", zorder=8)
    ax.text(x0 + largo_grados / 2, y0 + 0.015 * (n - s), f"{largo} km", ha="center", va="bottom",
            fontsize=8, color=estilo.TINTA, path_effects=_HALO, zorder=8)


def _flecha_norte(ax) -> None:
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.86), xycoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold", color=estilo.TINTA,
                arrowprops=dict(arrowstyle="-|>", color=estilo.TINTA, lw=1.5), path_effects=_HALO, zorder=8)


def dibujar(cfg: ConfigMapa, bat: batimetria.Batimetria, malla: waverys.Malla, tabla: pd.DataFrame):
    estilo.aplicar()
    o, e, s, n = cfg.recuadro
    elev = bat.elevacion
    lon, lat, z = elev["lon"].values, elev["lat"].values, elev.values
    extension = [lon[0], lon[-1], lat[0], lat[-1]]

    fig, ax = plt.subplots(figsize=(11, 8.5), layout="constrained")
    fig.set_facecolor(estilo.SUPERFICIE)
    ax.set_facecolor(estilo.SUPERFICIE)

    # Profundidad (rampa secuencial azul por clases) y tierra.
    prof = np.where(z < 0, -z, np.nan)
    pmax = np.nanmax(prof) if np.isfinite(prof).any() else 1.0
    niveles = [v for v in NIVELES_PROFUNDIDAD if v < pmax] + [next(v for v in NIVELES_PROFUNDIDAD if v >= pmax)]
    cmap = LinearSegmentedColormap.from_list("profundidad", estilo.RAMPA_PROFUNDIDAD)
    norma = BoundaryNorm(niveles, cmap.N)
    im = ax.imshow(prof, origin="lower", extent=extension, cmap=cmap, norm=norma, interpolation="nearest",
                   aspect="auto", zorder=1)
    ax.imshow(np.where(z >= 0, 1.0, np.nan), origin="lower", extent=extension, cmap=ListedColormap([estilo.TIERRA]),
              interpolation="nearest", aspect="auto", zorder=1)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.01, aspect=40,
                      spacing="uniform", ticks=niveles)
    cb.set_label("Profundidad (m)")
    cb.ax.tick_params(labelsize=7)

    # Costa e isóbatas (etiquetadas sobre la línea: la identidad no depende solo del color).
    ax.contour(lon, lat, z, levels=[0], colors=estilo.TINTA, linewidths=0.8, zorder=3)
    manejadores = [Line2D([], [], color=estilo.TINTA, lw=0.8, label="Línea de costa (0 m)")]
    for k, d in enumerate(cfg.isobatas):
        color = estilo.COLORES_ISOBATAS[k % len(estilo.COLORES_ISOBATAS)]
        ls = estilo.ESTILOS_ISOBATAS[k % len(estilo.ESTILOS_ISOBATAS)]
        lw = 2.0 if d == cfg.isobata_principal else 1.4
        cs = ax.contour(lon, lat, z, levels=[-d], colors=[color], linewidths=lw, linestyles=[ls], zorder=4)
        if cs.allsegs[0]:
            etiquetas = ax.clabel(cs, fmt={-d: f"{d:g} m"}, fontsize=7, inline=True, inline_spacing=2)
            for t in etiquetas:
                t.set_color(estilo.TINTA)
        principal = " (principal)" if d == cfg.isobata_principal else ""
        manejadores.append(Line2D([], [], color=color, lw=lw, ls=ls, label=f"Isóbata {d:g} m{principal}"))

    # Malla de WAVERYS: recuadro de cada celda, nodos válidos y enmascarados.
    h = malla.paso / 2
    for x in np.unique(np.concatenate([malla.lon - h, malla.lon + h])):
        ax.axvline(x, color=estilo.TINTA_TENUE, lw=0.4, alpha=0.7, zorder=5)
    for y in np.unique(np.concatenate([malla.lat - h, malla.lat + h])):
        ax.axhline(y, color=estilo.TINTA_TENUE, lw=0.4, alpha=0.7, zorder=5)
    LON, LAT = np.meshgrid(malla.lon, malla.lat)
    ax.scatter(LON[malla.valido], LAT[malla.valido], s=10, c=estilo.TINTA, linewidths=0, zorder=6)
    ax.scatter(LON[~malla.valido], LAT[~malla.valido], s=14, c=estilo.TINTA_SECUNDARIA, marker="x", linewidths=0.8, zorder=6)
    manejadores += [
        Line2D([], [], color=estilo.TINTA_TENUE, lw=0.6, label=f"Borde de celda WAVERYS ({malla.paso:.1f}°)"),
        Line2D([], [], color=estilo.TINTA, marker="o", ls="", ms=4, label="Nodo WAVERYS con dato"),
        Line2D([], [], color=estilo.TINTA_SECUNDARIA, marker="x", ls="", ms=5, label="Nodo WAVERYS sin dato (tierra)"),
    ]

    # Candidatos numerados (mismo número que en candidatos.csv).
    top = tabla[tabla["candidato"].notna() & (tabla["candidato"] <= cfg.n_candidatos)]
    ax.scatter(top["lon"], top["lat"], s=90, facecolors="none", edgecolors=estilo.TINTA, linewidths=1.6, zorder=7)
    for _, f in top.iterrows():
        ax.annotate(str(int(f["candidato"])), (f["lon"], f["lat"]), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=estilo.TINTA, path_effects=_HALO, zorder=8)
    if len(top):
        manejadores.append(Line2D([], [], color=estilo.TINTA, marker="o", mfc="none", ls="", ms=9, mew=1.6,
                                  label=f"Candidato (n.º = fila de la tabla), hasta {cfg.n_candidatos}"))

    # Coordenada de referencia.
    ax.scatter([cfg.longitud], [cfg.latitud], s=220, marker="*", c=estilo.TINTA, edgecolors="white",
               linewidths=1.2, zorder=9)
    manejadores.append(Line2D([], [], color=estilo.TINTA, marker="*", ls="", ms=12, mec="white",
                              label=f"Referencia ({_fmt_lat(cfg.latitud)}, {_fmt_lon(cfg.longitud)})"))

    _barra_de_escala(ax, cfg)
    _flecha_norte(ax)
    ax.set_xlim(o, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians(cfg.latitud)))
    for eje in (ax.xaxis, ax.yaxis):  # marcas en pasos de 1, 2 o 5: evita redondear 0.25 a un decimal
        eje.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.set_title(f"Mapa de selección de sitio: {cfg.nombre}", loc="left", fontweight="bold")

    leyenda = ax.legend(handles=manejadores, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    leyenda.get_frame().set_edgecolor(estilo.TINTA_TENUE)
    fuentes = "\n".join([
        "Fuentes",
        bat.etiqueta,
        malla.etiqueta,
        "Zonas de exclusión y colisión",
        "de usos: no evaluadas.",
        "Candidatos: nodo válido, celda sin tierra,",
        "8 vecinas válidas; orden por distancia",
        f"a la isóbata de {cfg.isobata_principal:g} m.",
    ])
    ax.text(1.02, 0.0, fuentes, transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            color=estilo.TINTA_SECUNDARIA, linespacing=1.4)
    return fig


def _guardar(cfg: ConfigMapa, raiz: Path, res: ResultadoMapa) -> dict[str, Path]:
    destino = rutas.interim(raiz, FASE, "mapa-sitio", cfg.nombre)
    salidas = {"mapa": destino / "mapa.png", "tabla": destino / "candidatos.csv", "metadatos": destino / "metadatos.json"}
    res.figura.savefig(salidas["mapa"], facecolor=res.figura.get_facecolor(), bbox_inches="tight",
                      pad_inches=0.15, metadata={"Software": None})
    res.tabla.to_csv(salidas["tabla"], index=False, float_format="%.4f", encoding="utf-8")
    meta = {
        "generado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version_codigo": __version__,
        "python": platform.python_version(),
        "configuracion": cfg.como_dict(),
        "insumos": {"batimetria": res.batimetria.origen, "waverys": res.malla.origen},
        "n_nodos": int(len(res.tabla)),
        "n_nodos_validos": int(res.tabla["valido"].sum()),
        "n_cumplen_criterios_basicos": int(res.tabla["cumple_criterios_basicos"].sum()),
        "no_evaluado": ["zonas de exclusión", "zonas con colisión de usos", "observaciones cercanas"],
    }
    salidas["metadatos"].write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return salidas


def generar_mapa(cfg: ConfigMapa, guardar: bool = True) -> ResultadoMapa:
    """Descarga (o reutiliza) los insumos, arma la tabla de candidatos, dibuja el mapa y guarda las salidas."""
    raiz = rutas.directorio_datos(cfg.directorio_datos)
    bat = batimetria.obtener(cfg, raiz)
    malla = waverys.obtener(cfg, raiz)
    tabla, _ = candidatos.tabla(cfg, bat.elevacion, malla)
    fig = dibujar(cfg, bat, malla, tabla)
    res = ResultadoMapa(figura=fig, tabla=tabla, batimetria=bat, malla=malla)
    if guardar:
        res.salidas = _guardar(cfg, raiz, res)
    return res
