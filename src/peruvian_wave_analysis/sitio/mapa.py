"""Mapa exploratorio de selección de sitio (ADR-0011): batimetría GMRT, isóbatas, malla de WAVERYS y zonas.

Uso mínimo::

    from peruvian_wave_analysis.sitio import ConfigMapa, generar_mapa
    r = generar_mapa(ConfigMapa(nombre="marcona", latitud=-15.4, longitud=-75.3))
    r.tabla.head()

Salidas en ``WAVE-DATA/interim/F01-sitio-y-fuentes/mapa-sitio/<nombre>/``: ``mapa.png``,
``candidatos.csv``, ``metadatos.json`` y ``siguiente_paso.md`` (qué recorte de GEBCO descargar
para el mapa definitivo).
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import box as caja

from .. import __version__, estilo, rutas
from . import batimetria, candidatos, waverys
from . import zonas as zonas_mod
from .config import ConfigMapa

FASE = "F01-sitio-y-fuentes"
URL_GEBCO = "https://download.gebco.net/"
NIVELES_PROFUNDIDAD = [0, 25, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 11000]
_HALO = [patheffects.withStroke(linewidth=2.5, foreground="white")]


@dataclass
class ResultadoMapa:
    figura: matplotlib.figure.Figure
    tabla: pd.DataFrame
    batimetria: batimetria.Batimetria
    malla: waverys.Malla
    zonas: zonas_mod.Zonas | None
    salidas: dict[str, Path] = field(default_factory=dict)
    recuadro_gebco: tuple[float, float, float, float] | None = None
    figura_detalle: matplotlib.figure.Figure | None = None  # solo el mapa definitivo


def _fmt_lon(v, _pos=None) -> str:
    return f"{abs(v):.1f}° {'O' if v < 0 else 'E'}"


def _fmt_lat(v, _pos=None) -> str:
    return f"{abs(v):.1f}° {'S' if v < 0 else 'N'}"


def recuadro_gebco(cfg: ConfigMapa) -> tuple[float, float, float, float]:
    """Recuadro (oeste, este, sur, norte) a descargar de GEBCO: el de la batimetría, redondeado hacia afuera a 0.1°."""
    o, e, s, n = batimetria.recuadro_con_margen(cfg)
    return (math.floor(o * 10) / 10, math.ceil(e * 10) / 10, math.floor(s * 10) / 10, math.ceil(n * 10) / 10)


def _barra_de_escala(ax, cfg: ConfigMapa) -> None:
    o, e, s, n = cfg.recuadro
    km_por_grado = 111.32 * np.cos(np.radians(cfg.latitud))
    largo = min((5, 10, 20, 25, 50, 100, 200), key=lambda L: abs(L - (e - o) * km_por_grado / 5))
    x0, y0 = o + 0.05 * (e - o), s + 0.05 * (n - s)
    ax.plot([x0, x0 + largo / km_por_grado], [y0, y0], color=estilo.TINTA, lw=3, solid_capstyle="butt", zorder=8)
    ax.text(
        x0 + largo / km_por_grado / 2,
        y0 + 0.015 * (n - s),
        f"{largo} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color=estilo.TINTA,
        path_effects=_HALO,
        zorder=8,
    )


def _flecha_norte(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.95, 0.95),
        xytext=(0.95, 0.86),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=estilo.TINTA,
        arrowprops=dict(arrowstyle="-|>", color=estilo.TINTA, lw=1.5),
        path_effects=_HALO,
        zorder=8,
    )


def _trayecto(geom) -> MplPath | None:
    """Polígonos de shapely (con huecos) → un solo Path de matplotlib."""
    caminos = []
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type == "Polygon":
            for anillo in [g.exterior, *g.interiors]:
                caminos.append(MplPath(np.asarray(anillo.coords)[:, :2], closed=True))
        elif g.geom_type in ("MultiPolygon", "GeometryCollection"):
            sub = _trayecto(g)
            if sub is not None:
                caminos.append(sub)
    return MplPath.make_compound_path(*caminos) if caminos else None


def _es_punto(z) -> bool:
    return z.geometria.geom_type in ("Point", "MultiPoint")


def _etiquetar(ax, geom, texto: str, color: str, vista) -> None:
    """Nombre de la zona en un punto interior de su parte visible."""
    visible = geom.intersection(vista)
    if visible.is_empty or visible.area < 0.002:  # unos 25 km²: zonas menores no se etiquetan
        return
    p = visible.representative_point()
    ax.text(
        p.x,
        p.y,
        texto,
        fontsize=6.5,
        color=color,
        ha="center",
        va="center",
        zorder=7,
        clip_on=True,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
    )


def _nombre_corto(z) -> str:
    n = z.nombre.split(": ", 1)[-1]
    n = n.replace("Reserva Nacional ", "RN ").replace("Sistema de Islas, Islotes y Puntas Guaneras - ", "RNSIIPG ")
    n = n.split(" · ")[0]
    return n if len(n) <= 32 else n[:31] + "…"


def _dibujar_zonas(ax, zonas: zonas_mod.Zonas, cfg: ConfigMapa, evitar=None) -> list:
    """Color por familia (verde SERNANP, pardo hidrocarburos); exclusión opaca con borde rojo; nombres sobre las
    zonas."""

    o, e, s, n = cfg.recuadro
    m = 0.12 * (e - o)  # las etiquetas se colocan lejos del borde para que no se salgan del marco
    vista = caja(o + m, s + 0.04 * (n - s), e - m, n - 0.04 * (n - s))
    if evitar is not None:  # zona donde no se colocan nombres (p. ej. alrededor del nodo elegido)
        vista = vista.difference(evitar)
    manejadores, etiquetados = [], set()
    for familia, titulo in zonas_mod.FAMILIAS.items():
        st = estilo.ESTILO_FAMILIA[familia]
        col = [z for z in zonas.de_clase(zonas_mod.COLISION) if z.familia == familia and not _es_punto(z)]
        for z in col:
            t = _trayecto(z.geometria)
            if t is None:
                continue
            if st["relleno"] != "none":
                ax.add_patch(PathPatch(t, facecolor=st["relleno"], edgecolor="none", alpha=st["alfa"], zorder=2))
            ax.add_patch(
                PathPatch(
                    t, facecolor="none", edgecolor=st["borde"], hatch=st["trama"], ls=st["linea"], lw=1.1, zorder=2
                )
            )
            nombre = _nombre_corto(z)
            if st["etiqueta"] and nombre not in etiquetados:
                etiquetados.add(nombre)
                _etiquetar(ax, z.geometria, nombre, st["borde"], vista)
        if col:
            relleno = matplotlib.colors.to_rgba(st["relleno"], st["alfa"]) if st["relleno"] != "none" else "none"
            manejadores.append(
                Patch(
                    facecolor=relleno,
                    edgecolor=st["borde"],
                    hatch=st["trama"],
                    ls=st["linea"],
                    lw=1.1,
                    label=f"{titulo} ({len(col)})",
                )
            )
    for familia, titulo in zonas_mod.FAMILIAS.items():
        st = estilo.ESTILO_FAMILIA[familia]
        exc = [z for z in zonas.de_clase(zonas_mod.EXCLUSION) if z.familia == familia]
        poligonos = [z for z in exc if not _es_punto(z)]
        puntos = [z for z in exc if _es_punto(z)]
        for z in poligonos:
            t = _trayecto(z.geometria)
            if t is not None:
                ax.add_patch(
                    PathPatch(
                        t, facecolor=st["borde"], edgecolor=estilo.COLOR_BORDE_EXCLUSION, alpha=0.85, lw=1.3, zorder=3
                    )
                )
        for z in puntos:
            for g in getattr(z.geometria, "geoms", [z.geometria]):
                ax.plot(
                    g.x,
                    g.y,
                    marker="^",
                    ms=7,
                    color=st["borde"],
                    mec=estilo.COLOR_BORDE_EXCLUSION,
                    mew=1.2,
                    ls="",
                    zorder=6,
                )
        if poligonos:
            manejadores.append(
                Patch(
                    facecolor=st["borde"],
                    edgecolor=estilo.COLOR_BORDE_EXCLUSION,
                    lw=1.3,
                    alpha=0.85,
                    label=f"EXCLUSIÓN · {titulo} ({len(poligonos)})",
                )
            )
        if puntos:
            manejadores.append(
                Line2D(
                    [],
                    [],
                    marker="^",
                    ms=7,
                    color=st["borde"],
                    mec=estilo.COLOR_BORDE_EXCLUSION,
                    mew=1.2,
                    ls="",
                    label=f"EXCLUSIÓN · pozo petrolero ({len(puntos)})",
                )
            )
    if zonas.de_clase(zonas_mod.EXCLUSION) and cfg.resguardo_km > 0:
        t = _trayecto(zonas_mod.exclusion_ampliada(zonas, cfg.longitud, cfg.latitud, cfg.resguardo_km))
        if t is not None:
            ax.add_patch(
                PathPatch(t, facecolor="none", edgecolor=estilo.COLOR_BORDE_EXCLUSION, ls=":", lw=1.2, zorder=3)
            )
        manejadores.append(
            Line2D(
                [],
                [],
                color=estilo.COLOR_BORDE_EXCLUSION,
                ls=":",
                lw=1.2,
                label=f"Resguardo de {cfg.resguardo_km:g} km alrededor de la exclusión",
            )
        )
    if not manejadores:
        manejadores.append(Patch(facecolor="none", edgecolor="none", label="Sin zonas en el recuadro"))
    else:
        manejadores.insert(
            0, Patch(facecolor="none", edgecolor="none", label="Zonas (colisión de usos salvo EXCLUSIÓN):")
        )
    return manejadores


def _dms(v: float, hemisferios: str) -> str:
    """15°23′20″S: grados, minutos y segundos redondeados al segundo."""
    s = round(abs(v) * 3600)
    return f"{s // 3600}°{s % 3600 // 60:02d}′{s % 60:02d}″{hemisferios[1] if v < 0 else hemisferios[0]}"


def _bordes_de_celda(malla: waverys.Malla, elev) -> list:
    """Segmentos únicos del borde de las celdas con algo de agua según la batimetría.

    Una celda sin batimetría (fuera del recorte) se dibuja: no se sabe si es tierra.
    """
    h = malla.paso / 2
    lados = set()
    for la in malla.lat:
        for lo in malla.lon:
            frac_tierra = candidatos._estadisticas_celda(elev, float(la), float(lo), malla.paso)[2]
            if frac_tierra >= 1:
                continue
            x0, x1, y0, y1 = (round(float(v), 6) for v in (lo - h, lo + h, la - h, la + h))
            lados |= {((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))}
    return [list(s) for s in lados]


def dibujar(
    cfg: ConfigMapa,
    bat: batimetria.Batimetria,
    malla: waverys.Malla,
    tabla: pd.DataFrame,
    zonas: zonas_mod.Zonas | None,
    modo: str = "exploratorio",
    nodo_elegido: tuple[float, float] | None = None,
    aviso: str | None = None,
):
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
        im, ax=ax, orientation="horizontal", shrink=0.6, pad=0.01, aspect=40, spacing="uniform", ticks=niveles
    )
    cb.set_label("Profundidad (m)")
    cb.ax.tick_params(labelsize=7)

    manejadores_zonas = _dibujar_zonas(ax, zonas, cfg) if zonas is not None else []

    # Costa e isóbatas (etiquetadas sobre la línea: la identidad no depende solo del color).
    ax.contour(lon, lat, z, levels=[0], colors=estilo.TINTA, linewidths=0.8, zorder=3)
    manejadores = [Line2D([], [], color=estilo.TINTA, lw=0.8, label="Línea de costa (0 m)")]
    for k, d in enumerate(cfg.isobatas):
        color = estilo.COLORES_ISOBATAS[k % len(estilo.COLORES_ISOBATAS)]
        ls = estilo.ESTILOS_ISOBATAS[k % len(estilo.ESTILOS_ISOBATAS)]
        lw = 2.0 if d == cfg.isobata_principal else 1.4
        cs = ax.contour(lon, lat, z, levels=[-d], colors=[color], linewidths=lw, linestyles=[ls], zorder=4)
        if cs.allsegs[0]:
            # inline=False: la etiqueta va encima, sin cortar la línea (un corte parece una discontinuidad).
            for t in ax.clabel(cs, fmt={-d: f"{d:g} m"}, fontsize=7, inline=False):
                t.set_color(estilo.TINTA)
                t.set_path_effects(_HALO)
        principal = " (principal)" if d == cfg.isobata_principal else ""
        manejadores.append(Line2D([], [], color=color, lw=lw, ls=ls, label=f"Isóbata {d:g} m{principal}"))
    manejadores += manejadores_zonas

    # Malla de WAVERYS: borde de las celdas que tienen algo de agua y nodos con dato.
    # Las celdas enteramente en tierra y los nodos sin dato no se dibujan: solo recargan la vista.
    ax.add_collection(
        LineCollection(_bordes_de_celda(malla, elev), colors=estilo.TINTA_TENUE, linewidths=0.4, alpha=0.7, zorder=5)
    )
    LON, LAT = np.meshgrid(malla.lon, malla.lat)
    ax.scatter(LON[malla.valido], LAT[malla.valido], s=10, c=estilo.TINTA, linewidths=0, zorder=6)
    manejadores += [
        Line2D(
            [],
            [],
            color=estilo.TINTA_TENUE,
            lw=0.6,
            label=f"Borde de celda WAVERYS ({malla.paso:.1f}°;\nse omiten las celdas solo de tierra)",
        ),
        Line2D([], [], color=estilo.TINTA, marker="o", ls="", ms=4, label="Nodo WAVERYS con dato"),
    ]

    # Nodos que cumplen todo salvo las vecinas: se marcan para que se vea por qué no son candidatos.
    solo_vecinas = tabla[tabla["motivo_descarte"].str.startswith("3:")]
    if len(solo_vecinas):
        ax.scatter(
            solo_vecinas["lon"],
            solo_vecinas["lat"],
            s=70,
            marker="s",
            facecolors="none",
            edgecolors=estilo.TINTA_SECUNDARIA,
            linewidths=1.2,
            zorder=7,
        )
        manejadores.append(
            Line2D(
                [],
                [],
                marker="s",
                mfc="none",
                mec=estilo.TINTA_SECUNDARIA,
                mew=1.2,
                ls="",
                ms=8,
                label=f"Descartado por vecinas (< {cfg.min_vecinas}) ({len(solo_vecinas)})",
            )
        )

    # Candidatos numerados (mismo número que en candidatos.csv). El mapa definitivo ya no tiene candidatos.
    if modo == "exploratorio":
        top = tabla[tabla["candidato"].notna()]
        ax.scatter(top["lon"], top["lat"], s=90, facecolors="none", edgecolors=estilo.TINTA, linewidths=1.6, zorder=7)
        for _, f in top.iterrows():
            ax.annotate(
                str(int(f["candidato"])),
                (f["lon"], f["lat"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=estilo.TINTA,
                path_effects=_HALO,
                zorder=8,
            )
        manejadores.append(
            Line2D(
                [],
                [],
                color=estilo.TINTA,
                marker="o",
                mfc="none",
                ls="",
                ms=9,
                mew=1.6,
                label=f"Candidato n.º (fila de la tabla): {len(top)} de hasta {cfg.n_candidatos}",
            )
        )
    if nodo_elegido is not None:
        ax.annotate(
            "Nodo elegido",
            (nodo_elegido[1], nodo_elegido[0]),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=estilo.TINTA,
            path_effects=_HALO,
            zorder=8,
        )
        ax.scatter(
            [nodo_elegido[1]],
            [nodo_elegido[0]],
            s=420,
            facecolors="none",
            edgecolors=estilo.TINTA,
            linewidths=2.6,
            zorder=8,
        )
        manejadores.append(
            Line2D(
                [],
                [],
                color=estilo.TINTA,
                marker="o",
                mfc="none",
                ls="",
                ms=15,
                mew=2.6,
                label=f"Nodo elegido ({_fmt_lat(nodo_elegido[0])}, {_fmt_lon(nodo_elegido[1])})",
            )
        )

    ax.scatter(
        [cfg.longitud], [cfg.latitud], s=220, marker="*", c=estilo.TINTA, edgecolors="white", linewidths=1.2, zorder=9
    )
    manejadores.append(
        Line2D(
            [],
            [],
            color=estilo.TINTA,
            marker="*",
            ls="",
            ms=12,
            mec="white",
            label=f"Referencia ({_fmt_lat(cfg.latitud)}, {_fmt_lon(cfg.longitud)})",
        )
    )
    for p in cfg.puntos:
        ax.scatter(
            [p.longitud],
            [p.latitud],
            s=70,
            marker=p.marcador,
            c=estilo.TINTA_SECUNDARIA,
            edgecolors="white",
            linewidths=1.2,
            zorder=9,
        )
        dx, dy = p.desplazamiento
        ax.annotate(
            p.etiqueta,
            (p.longitud, p.latitud),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=estilo.TINTA,
            path_effects=_HALO,
            zorder=9,
            ha="left" if dx >= 0 else "right",
            va="top" if dy < 0 else "bottom",
        )
        manejadores.append(
            Line2D(
                [],
                [],
                color=estilo.TINTA_SECUNDARIA,
                marker=p.marcador,
                ls="",
                ms=7,
                mec="white",
                label=f"{p.nombre} ({_dms(p.latitud, 'NS')}, {_dms(p.longitud, 'EO')})",
            )
        )

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
    titulo = "Mapa exploratorio de selección de sitio" if modo == "exploratorio" else "Mapa definitivo del sitio"
    ax.set_title(f"{titulo}: {cfg.nombre}", loc="left", fontweight="bold")

    leyenda = ax.legend(handles=manejadores, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    leyenda.get_frame().set_edgecolor(estilo.TINTA_TENUE)

    lineas = ["Fuentes", bat.etiqueta, malla.etiqueta]
    if zonas is not None:
        fechas = sorted({c["fecha_acceso_utc"][:10] for c in zonas.capas.values() if c["estado"] == "ok"})
        lineas.append(f"Zonas: SERNANP, GEOCATMIN,\n  acceso {', '.join(fechas) or '—'}")
        no_eval = zonas.no_evaluadas + list(zonas_mod.NO_CUBIERTAS)
        descartadas = sorted({d.split(" · ")[0] for c in zonas.capas.values() for d in c.get("descartadas", [])})
        if descartadas:
            filas = [", ".join(descartadas[k : k + 3]) for k in range(0, len(descartadas), 3)]
            lineas.append("Quitadas (lotes sin contrato\n  ni referencia pública):\n  " + "\n  ".join(filas))
    else:
        no_eval = ["todas las zonas"]
    lineas += ["", "No evaluado:"] + [f"  {x}" for x in no_eval]
    extra = ""
    if cfg.max_tierra < 1:
        extra += f", ≤ {100 * cfg.max_tierra:g} % de tierra"
    if cfg.min_vecinas:
        extra += f", ≥ {cfg.min_vecinas} vecinas"
    if modo == "exploratorio":
        lineas += [
            "",
            "Criterios (ADR-0014): nodo con dato y",
            f"fuera de exclusión +{cfg.resguardo_km:g} km{extra}.",
            f"Numeración por distancia a la isóbata de {cfg.isobata_principal:g} m;",
            "la elección final es visual.",
        ]
    else:
        lineas += [
            "",
            "Nodo elegido (ADR-0007). Cumple con esta",
            "batimetría: tiene dato y su celda queda",
            f"fuera de exclusión +{cfg.resguardo_km:g} km (ADR-0014).",
            "Detalle de su celda y sus vecinas:",
            "detalle_nodo.png.",
        ]
    if aviso:
        lineas += ["", aviso]
    # Debajo de la leyenda, sea cual sea su alto (bbox_inches="tight" al guardar incluye todo).
    notas = ax.annotate(
        "\n".join(lineas),
        xy=(0, 0),
        xycoords=leyenda,
        xytext=(0, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        color=estilo.TINTA_SECUNDARIA,
        linespacing=1.35,
    )
    notas.set_in_layout(False)  # que el diseño no reserve espacio bajo el mapa por estas notas
    return fig


def guardar_figura(fig, destino: Path, intentos: int = 10, espera_s: float = 0.5) -> None:
    """Guarda en un temporal y lo pone en su lugar con reintentos.

    En Windows un PNG abierto en un visor queda bloqueado y ``savefig`` directo falla a medias.
    """

    tmp = destino.with_suffix(".tmp.png")
    fig.savefig(tmp, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15, metadata={"Software": None})
    for k in range(intentos):
        try:
            os.replace(tmp, destino)
            return
        except PermissionError as exc:
            if k == intentos - 1:
                tmp.unlink(missing_ok=True)
                raise PermissionError(
                    f"No se pudo escribir {destino}: está abierto en otro programa. Ciérralo y vuelve a correr."
                ) from exc
            time.sleep(espera_s)


def _texto_siguiente_paso(cfg: ConfigMapa, caja: tuple[float, float, float, float], tabla: pd.DataFrame) -> str:
    o, e, s, n = caja
    top = tabla[tabla["candidato"].notna()]
    filas = "\n".join(
        f"| {int(f.candidato)} | {f.lat:.4f} | {f.lon:.4f} | {f.zonas_colision or '—'} |" for f in top.itertuples()
    )
    return f"""# Siguiente paso: mapa definitivo de «{cfg.nombre}»

Este mapa es **exploratorio** (batimetría GMRT, ADR-0011). Para el mapa definitivo:

1. Elegir el nodo entre los candidatos:

| Candidato | Latitud | Longitud | Colisiones de uso declaradas |
|---|---|---|---|
{filas or "| — | — | — | ningún nodo cumple los criterios |"}

2. Descargar el recorte de GEBCO en {URL_GEBCO} con este recuadro, en formato **NetCDF** (malla de elevación):

| Norte | Sur | Oeste | Este |
|---|---|---|---|
| {n:.1f} | {s:.1f} | {o:.1f} | {e:.1f} |

3. Registrarlo (lo copia a `WAVE-DATA/raw/batimetria/gebco/` y anota su hash):

```
uv run registrar-gebco --archivo <ruta al .nc descargado> --version GEBCO_AAAA
```

4. Generar el mapa definitivo (no descarga nada; usa los insumos de esta corrida):

```
uv run mapa-sitio-final --exploratorio <ruta a este metadatos.json> --gebco <ruta en raw/batimetria/gebco> --nodo-lat <lat> --nodo-lon <lon>
```
"""  # noqa: E501 (línea de un comando en el texto publicado de siguiente_paso.md: partirla cambiaría la salida)


def _guardar(cfg: ConfigMapa, raiz: Path, res: ResultadoMapa) -> dict[str, Path]:
    destino = rutas.interim(raiz, FASE, "mapa-sitio", cfg.nombre)
    salidas = {
        "mapa": destino / "mapa.png",
        "tabla": destino / "candidatos.csv",
        "metadatos": destino / "metadatos.json",
        "siguiente_paso": destino / "siguiente_paso.md",
    }
    guardar_figura(res.figura, salidas["mapa"])
    res.tabla.to_csv(salidas["tabla"], index=False, float_format="%.4f", encoding="utf-8")
    meta = {
        "tipo": "mapa exploratorio",
        "generado_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version_codigo": __version__,
        "python": platform.python_version(),
        "configuracion": cfg.como_dict(),
        "insumos": {"batimetria": res.batimetria.origen, "waverys": res.malla.origen},
        "zonas": res.zonas.capas if res.zonas is not None else None,
        "n_nodos": int(len(res.tabla)),
        "n_nodos_validos": int(res.tabla["valido"].sum()),
        "n_cumplen_criterios": int(res.tabla["cumple_criterios"].sum()),
        "candidatos": res.tabla.loc[res.tabla["candidato"].notna(), ["candidato", "lat", "lon"]]
        .astype(float)
        .to_dict("records"),
        "recuadro_gebco_oeste_este_sur_norte": list(res.recuadro_gebco),
        "no_evaluado": (res.zonas.no_evaluadas if res.zonas is not None else ["todas las zonas"])
        + list(zonas_mod.NO_CUBIERTAS),
    }
    salidas["metadatos"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    salidas["siguiente_paso"].write_text(_texto_siguiente_paso(cfg, res.recuadro_gebco, res.tabla), encoding="utf-8")
    return salidas


def generar_mapa(cfg: ConfigMapa, guardar: bool = True) -> ResultadoMapa:
    """Descarga (o reutiliza) los insumos, arma la tabla de candidatos, dibuja el mapa exploratorio y guarda las
    salidas."""
    raiz = rutas.directorio_datos(cfg.directorio_datos)
    bat = batimetria.obtener(cfg, raiz)
    malla = waverys.obtener(cfg, raiz)
    zonas = zonas_mod.obtener(batimetria.recuadro_con_margen(cfg), raiz) if cfg.zonas else None
    tabla, _ = candidatos.tabla(cfg, bat.elevacion, malla, zonas)
    caja = recuadro_gebco(cfg)
    o, e, s, n = caja
    aviso = (
        "MAPA EXPLORATORIO (GMRT). Para el definitivo,\n"
        f"descargar GEBCO: N {n:.1f}, S {s:.1f}, O {o:.1f}, E {e:.1f}\n"
        "(ver siguiente_paso.md)."
    )
    fig = dibujar(cfg, bat, malla, tabla, zonas, modo="exploratorio", aviso=aviso)
    res = ResultadoMapa(figura=fig, tabla=tabla, batimetria=bat, malla=malla, zonas=zonas, recuadro_gebco=caja)
    if guardar:
        res.salidas = _guardar(cfg, raiz, res)
    return res
