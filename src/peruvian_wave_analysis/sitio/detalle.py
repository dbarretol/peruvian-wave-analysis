"""Mapa de detalle del nodo elegido: su celda de WAVERYS y las ocho vecinas, con la batimetría a resolución completa.

Lo usa el mapa definitivo (ADR-0011) para documentar el nodo en la tesis. No decide nada: muestra.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import box as caja
from shapely.ops import unary_union

from .. import estilo
from . import batimetria, candidatos, waverys, zonas as zonas_mod
from .config import ConfigMapa
from .mapa import _HALO, _barra_de_escala, _dibujar_zonas, _dms, _flecha_norte

NIVELES_DETALLE = [
    0,
    10,
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    250,
    300,
    400,
    500,
    600,
    800,
    1000,
    1250,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000,
    6000,
    8000,
]
MARGEN_GRADOS = 0.08  # alrededor del bloque de 3 × 3 celdas


def _fmt(v: float, hemisferios: str) -> str:
    return f"{abs(v):.1f}° {hemisferios[1] if v < 0 else hemisferios[0]}"


def vecindario(malla: waverys.Malla, elev, lat: float, lon: float) -> list[dict]:
    """El nodo y sus ocho vecinas: posición, dato en WAVERYS, profundidad en el nodo y fracción de tierra de la
    celda."""
    i = int(np.argmin(np.abs(malla.lat - lat)))
    j = int(np.argmin(np.abs(malla.lon - lon)))
    filas = []
    for di in (1, 0, -1):  # de norte a sur
        for dj in (-1, 0, 1):  # de oeste a este
            a, b = i + di, j + dj
            if not (0 <= a < malla.lat.size and 0 <= b < malla.lon.size):
                continue
            la, lo = float(malla.lat[a]), float(malla.lon[b])
            pmin, pmed, ftierra = candidatos._estadisticas_celda(elev, la, lo, malla.paso)
            filas.append(
                {
                    "lat": round(la, 4),
                    "lon": round(lo, 4),
                    "es_nodo_elegido": di == dj == 0,
                    "valido": bool(malla.valido[a, b]),
                    "prof_nodo_m": round(float(-elev.sel(lat=la, lon=lo, method="nearest")), 1),
                    "prof_min_celda_m": None if np.isnan(pmin) else round(pmin, 1),
                    "prof_media_celda_m": None if np.isnan(pmed) else round(pmed, 1),
                    "frac_tierra_celda": None if np.isnan(ftierra) else round(ftierra, 4),
                }
            )
    return filas


def dibujar_detalle(
    cfg: ConfigMapa,
    bat: batimetria.Batimetria,
    malla: waverys.Malla,
    zonas: zonas_mod.Zonas | None,
    fila: dict,
    aviso: str | None = None,
):
    """Figura del bloque de 3 × 3 celdas centrado en el nodo elegido (``fila``: su fila de la tabla de nodos)."""
    estilo.aplicar()
    la0, lo0 = float(fila["lat"]), float(fila["lon"])
    semiancho = 1.5 * malla.paso + MARGEN_GRADOS
    o, e, s, n = lo0 - semiancho, lo0 + semiancho, la0 - semiancho, la0 + semiancho
    vista = replace(cfg, latitud=la0, longitud=lo0, semiancho_grados=semiancho)

    elev = bat.elevacion.sel(lat=slice(s - 0.02, n + 0.02), lon=slice(o - 0.02, e + 0.02))
    lon, lat, z = elev["lon"].values, elev["lat"].values, elev.values
    paso_bat = float(np.median(np.diff(lat)))
    extension = [lon[0] - paso_bat / 2, lon[-1] + paso_bat / 2, lat[0] - paso_bat / 2, lat[-1] + paso_bat / 2]

    fig, ax = plt.subplots(figsize=(10.5, 8.5), layout="constrained")
    fig.set_facecolor(estilo.SUPERFICIE)
    ax.set_facecolor(estilo.SUPERFICIE)

    # Batimetría píxel a píxel (sin suavizar) con clases finas y líneas de profundidad.
    prof = np.where(z < 0, -z, np.nan)
    pmin, pmax = (np.nanmin(prof), np.nanmax(prof)) if np.isfinite(prof).any() else (0.0, 1.0)
    niveles = [v for v in NIVELES_DETALLE if v < pmax]
    niveles = [v for v in niveles if v >= max(0, pmin) or v == niveles[-1]] or [0]
    niveles = sorted({0, *niveles, next((v for v in NIVELES_DETALLE if v >= pmax), NIVELES_DETALLE[-1])})
    cmap = LinearSegmentedColormap.from_list("profundidad", estilo.RAMPA_PROFUNDIDAD)
    im = ax.imshow(
        prof,
        origin="lower",
        extent=extension,
        cmap=cmap,
        norm=BoundaryNorm(niveles, cmap.N),
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )
    ax.imshow(
        np.where(z >= 0, 1.0, np.nan),
        origin="lower",
        extent=extension,
        cmap=ListedColormap([estilo.TIERRA]),
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )
    cb = fig.colorbar(
        im, ax=ax, orientation="horizontal", shrink=0.7, pad=0.01, aspect=45, spacing="uniform", ticks=niveles
    )
    cb.set_label("Profundidad (m)")
    cb.ax.tick_params(labelsize=6.5, labelrotation=45 if len(niveles) > 14 else 0)

    lineas_prof = [v for v in niveles[1:-1] if v not in cfg.isobatas]
    if lineas_prof:
        cs = ax.contour(
            lon, lat, z, levels=sorted(-v for v in lineas_prof), colors=estilo.TINTA_TENUE, linewidths=0.5, zorder=3
        )
        for t in ax.clabel(cs, fmt=lambda v: f"{-v:g}", fontsize=6, inline=False):
            t.set_color(estilo.TINTA_SECUNDARIA)
            t.set_path_effects(_HALO)

    # Los nombres de las zonas no se colocan sobre los nodos ni sobre sus rótulos.
    h = malla.paso / 2
    vecinas = vecindario(malla, bat.elevacion, la0, lo0)
    evitar = unary_union(
        [caja(v["lon"] - 0.4 * h, v["lat"] - 0.6 * h, v["lon"] + 0.4 * h, v["lat"] + 0.25 * h) for v in vecinas]
    )
    manejadores_zonas = _dibujar_zonas(ax, zonas, vista, evitar=evitar) if zonas is not None else []

    ax.contour(lon, lat, z, levels=[0], colors=estilo.TINTA, linewidths=0.9, zorder=4)
    manejadores = [Line2D([], [], color=estilo.TINTA, lw=0.9, label="Línea de costa (0 m)")]
    for k, d in enumerate(cfg.isobatas):
        color = estilo.COLORES_ISOBATAS[k % len(estilo.COLORES_ISOBATAS)]
        ls = estilo.ESTILOS_ISOBATAS[k % len(estilo.ESTILOS_ISOBATAS)]
        lw = 2.2 if d == cfg.isobata_principal else 1.5
        cs = ax.contour(lon, lat, z, levels=[-d], colors=[color], linewidths=lw, linestyles=[ls], zorder=4)
        if cs.allsegs[0]:
            for t in ax.clabel(cs, fmt={-d: f"{d:g} m"}, fontsize=7.5, inline=False):
                t.set_color(estilo.TINTA)
                t.set_path_effects(_HALO)
        principal = " (principal)" if d == cfg.isobata_principal else ""
        manejadores.append(Line2D([], [], color=color, lw=lw, ls=ls, label=f"Isóbata {d:g} m{principal}"))
    if lineas_prof:
        manejadores.append(Line2D([], [], color=estilo.TINTA_TENUE, lw=0.5, label="Otras profundidades (m)"))
    manejadores += manejadores_zonas

    # Celdas de WAVERYS: la del nodo elegido en trazo grueso; en cada nodo, su profundidad y la tierra de su celda.
    for v in vecinas:
        elegido = v["es_nodo_elegido"]
        ax.add_patch(
            Rectangle(
                (v["lon"] - h, v["lat"] - h),
                malla.paso,
                malla.paso,
                facecolor="none",
                edgecolor=estilo.TINTA,
                lw=2.4 if elegido else 0.8,
                ls="-" if elegido else "--",
                zorder=6 if elegido else 5,
            )
        )
        if v["valido"]:
            ax.scatter([v["lon"]], [v["lat"]], s=40 if elegido else 22, c=estilo.TINTA, linewidths=0, zorder=7)
            tierra = v["frac_tierra_celda"] or 0.0
            texto = f"{v['prof_nodo_m']:.0f} m" + (f"\n{100 * tierra:.0f} % tierra" if tierra > 0 else "")
        else:
            ax.scatter([v["lon"]], [v["lat"]], s=40, c=estilo.TINTA_SECUNDARIA, marker="x", linewidths=1.2, zorder=7)
            texto = "sin dato\nen WAVERYS"
        ax.annotate(
            texto,
            (v["lon"], v["lat"]),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8 if elegido else 7,
            fontweight="bold" if elegido else "normal",
            color=estilo.TINTA,
            path_effects=_HALO,
            zorder=8,
        )
    ax.scatter([lo0], [la0], s=260, facecolors="none", edgecolors=estilo.TINTA, linewidths=2.2, zorder=8)
    manejadores += [
        Line2D([], [], color=estilo.TINTA, lw=2.4, label=f"Celda del nodo elegido ({malla.paso:.1f}°)"),
        Line2D([], [], color=estilo.TINTA, lw=0.8, ls="--", label="Celdas vecinas"),
        Line2D([], [], color=estilo.TINTA, marker="o", mfc="none", ms=12, mew=2.2, ls="", label="Nodo elegido"),
        Line2D(
            [],
            [],
            color=estilo.TINTA,
            marker="o",
            ls="",
            ms=4,
            label="Nodo con dato: profundidad\n(y % de tierra en su celda)",
        ),
        Line2D(
            [], [], color=estilo.TINTA_SECUNDARIA, marker="x", ls="", ms=6, mew=1.2, label="Nodo sin dato en WAVERYS"
        ),
    ]
    if not np.isfinite(prof).any() or (z >= 0).any():
        manejadores.insert(1, Patch(facecolor=estilo.TIERRA, edgecolor="none", label="Tierra"))

    _barra_de_escala(ax, vista)
    _flecha_norte(ax)
    ax.set_xlim(o, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians(la0)))
    for eje in (ax.xaxis, ax.yaxis):
        eje.set_major_locator(MultipleLocator(malla.paso / 2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, "EO")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _fmt(v, "NS")))
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.set_title(f"Detalle del nodo elegido: {cfg.nombre}", loc="left", fontweight="bold")

    leyenda = ax.legend(handles=manejadores, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    leyenda.get_frame().set_edgecolor(estilo.TINTA_TENUE)

    con_dato = sum(v["valido"] for v in vecinas if not v["es_nodo_elegido"])
    presentes = sum(not v["es_nodo_elegido"] for v in vecinas)
    lineas = [
        "Nodo elegido (ADR-0007)",
        f"  {_fmt(la0, 'NS')}, {_fmt(lo0, 'EO')}",
        f"  ({_dms(la0, 'NS')}, {_dms(lo0, 'EO')})",
        f"  Profundidad en el nodo: {fila['prof_nodo_m']:.0f} m",
        f"  Celda: mínima {fila['prof_min_celda_m']:.0f} m, media {fila['prof_media_celda_m']:.0f} m,",
        f"  tierra {100 * fila['frac_tierra_celda']:.1f} %",
        f"  Vecinas con dato: {con_dato} de {presentes}",
    ]
    for d in cfg.isobatas:
        clave = f"dist_iso_{d:g}m_km"
        if clave in fila and fila[clave] == fila[clave]:
            lineas.append(f"  A la isóbata de {d:g} m: {fila[clave]:.2f} km")
    lineas.append(f"  A la costa: {fila['dist_costa_km']:.1f} km")
    if fila.get("dist_exclusion_km") == fila.get("dist_exclusion_km") and fila.get("dist_exclusion_km") is not None:
        lineas.append(f"  A la zona de exclusión más cercana: {fila['dist_exclusion_km']:.1f} km")
    if fila.get("zonas_colision"):
        lineas.append("  Colisión de usos (declarada):")
        lineas += [f"    {c.split(': ', 1)[-1]}" for c in str(fila["zonas_colision"]).split("; ")]
    lineas += [
        "",
        "Fuentes",
        bat.etiqueta,
        f"  resolución {paso_bat * 3600:.0f}″ (≈ {paso_bat * 111.2:.2f} km)",
        malla.etiqueta,
    ]
    if aviso:
        lineas += ["", aviso]
    notas = ax.annotate(
        "\n".join(lineas),
        xy=(0, 0),
        xycoords=leyenda,
        xytext=(0, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        color=estilo.TINTA_SECUNDARIA,
        linespacing=1.35,
    )
    notas.set_in_layout(False)
    return fig, vecinas
