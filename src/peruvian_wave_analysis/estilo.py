"""Estilo único de figuras.

Reglas para las figuras de análisis: etiquetas en español con unidades entre paréntesis; nunca recortar el máximo
observado (``asegurar_visible``); posición de trazado i/(N+1) (Makkonen, 2006); guardado determinista en PNG.
``revisar_figura`` sirve para que las pruebas y los gates detecten ejes sin unidades, palabras sin tilde o sin ñ y
textos que se salen del lienzo.

Colores: paleta de referencia validada para daltonismo (orden fijo, nunca cíclico).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RC = {
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "axes.formatter.use_locale": False,
}

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
TIERRA = "#e9dfc4"  # beige pardo

# Isóbatas: orden fijo naranja, violeta, magenta, ocre, rojo. Sin verdes ni pardos: esos son de las zonas.
COLORES_ISOBATAS = ["#eb6834", "#6250d6", "#d55181", "#c98500", "#e34948"]
ESTILOS_ISOBATAS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
# Profundidad: rampa secuencial azul, de claro a oscuro.
RAMPA_PROFUNDIDAD = [
    "#f2f7fb",
    "#dbe8f4",
    "#bfd6eb",
    "#9fc1df",
    "#7fa9d0",
    "#6290bd",
    "#4a77a3",
]  # azul suave: deja leer las zonas


def aplicar() -> None:
    matplotlib.rcParams.update(RC)


# Zonas: un color por familia (verde = SERNANP, pardo = hidrocarburos), estilo limpio:
# relleno tenue + borde, trama solo en áreas protegidas. Exclusión = relleno opaco con borde rojo.
COLOR_BORDE_EXCLUSION = "#d03b3b"
ESTILO_FAMILIA = {
    #                   borde       relleno     opacidad  trama  línea  etiquetar
    "areas_protegidas": {
        "borde": "#1f7a2e",
        "relleno": "#9fd3a8",
        "alfa": 0.45,
        "trama": "//",
        "linea": "-",
        "etiqueta": True,
    },
    "amortiguamiento": {
        "borde": "#5f8f2a",
        "relleno": "none",
        "alfa": 0.0,
        "trama": None,
        "linea": ":",
        "etiqueta": False,
    },
    "aavc": {"borde": "#138a7e", "relleno": "#9fd8cf", "alfa": 0.15, "trama": None, "linea": "--", "etiqueta": True},
    "propuesta": {"borde": "#3f8f3f", "relleno": "none", "alfa": 0.0, "trama": None, "linea": "-.", "etiqueta": True},
    "petroleo": {"borde": "#8b5a1a", "relleno": "#e3cfa6", "alfa": 0.45, "trama": None, "linea": "-", "etiqueta": True},
}


# --- Figuras de análisis -------------------------------------------------------------------------

RC_ANALISIS = {
    "figure.figsize": (7.0, 4.5),
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}

ETIQUETAS: dict[str, str] = {
    "Hs": "Hs (m)",
    "Tp": "Tp (s)",
    "Tz": "Tz (s)",
    "Dir": "Dirección (°)",
    "exceso": "Exceso sobre u (m)",
    "tiempo": "Fecha (UTC)",
    "T": "Período de retorno (años)",
    "RL": "Nivel de retorno de Hs (m)",
    "u": "Umbral u (m)",
    "IET": "IET (h)",
    "xi": "ξ (–)",
    "sigma": "σ (m)",
    "prob": "Probabilidad (–)",
    "n_picos": "Picos (n)",
    "tasa": "Tasa (eventos/año)",
    "cobertura": "Cobertura (%)",
}

# Palabras que suelen aparecer sin ñ o sin tilde.
_SIN_TILDE = {
    r"\banos\b": "años",
    r"\bano\b": "año",
    r"\bestacion\b": "estación",
    r"\bperiodo\b": "período",
    r"\bdireccion\b": "dirección",
    r"\bnumero\b": "número",
}
_UNIDAD = re.compile(r"\(.+\)")
_TOL_PX = 1.0


def nueva_figura(filas: int = 1, columnas: int = 1, **kw):

    matplotlib.rcParams.update(RC)
    matplotlib.rcParams.update(RC_ANALISIS)
    return plt.subplots(filas, columnas, **kw)


def posiciones_trazado(n: int) -> np.ndarray:
    """Posición de trazado i/(N+1), i = 1..N."""
    if n < 1:
        raise ValueError("n debe ser ≥ 1")
    return np.arange(1, n + 1) / (n + 1.0)


def asegurar_visible(ax, valores: Iterable[float], eje: str = "y", margen: float = 0.05) -> None:
    """Amplía el eje para que todos los valores (por ejemplo el máximo observado) queden dentro."""
    v = np.asarray(list(valores), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    leer, fijar = (ax.get_ylim, ax.set_ylim) if eje == "y" else (ax.get_xlim, ax.set_xlim)
    lo, hi = leer()
    tramo = max(hi - lo, float(v.max() - v.min()), 1e-12)
    fijar(
        min(lo, float(v.min()) - margen * tramo) if v.min() < lo else lo,
        max(hi, float(v.max()) + margen * tramo) if v.max() > hi else hi,
    )


def _marcas_dibujadas(eje, limites) -> list:
    lo, hi = sorted(limites)
    tol = 1e-9 * max(abs(hi - lo), 1e-300)
    out = []
    for m in eje.get_major_ticks() + eje.get_minor_ticks():
        if lo - tol <= m.get_loc() <= hi + tol:
            out += [x for x in (m.label1, m.label2) if x.get_visible()]
    return out


def _textos_fuera(fig) -> list[str]:
    """Textos visibles cuya caja se sale del lienzo (se guardan sin ``bbox_inches``: quedarían cortados)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ancho, alto = fig.bbox.width, fig.bbox.height
    # _suptitle, _left_title y _right_title son atributos privados de matplotlib (versión fijada en uv.lock);
    # la prueba de estilo los vigila.
    textos = [("supertítulo", fig._suptitle)] if fig._suptitle is not None else []
    textos += [("texto de la figura", x) for x in fig.texts] + [("leyenda de la figura", x) for x in fig.legends]
    for i, ax in enumerate(fig.get_axes()):
        textos += [(f"título del panel {i}", x) for x in (ax.title, ax._left_title, ax._right_title)]
        textos += [(f"etiqueta x del panel {i}", ax.xaxis.label), (f"etiqueta y del panel {i}", ax.yaxis.label)]
        textos += [(f"marca x del panel {i}", x) for x in _marcas_dibujadas(ax.xaxis, ax.get_xlim())]
        textos += [(f"marca y del panel {i}", x) for x in _marcas_dibujadas(ax.yaxis, ax.get_ylim())]
        if ax.get_legend() is not None:
            textos.append((f"leyenda del panel {i}", ax.get_legend()))
    problemas = []
    for nombre, art in textos:
        if not art.get_visible() or (hasattr(art, "get_text") and not art.get_text().strip()):
            continue
        bb = art.get_window_extent(rend)
        if bb.x0 < -_TOL_PX or bb.y0 < -_TOL_PX or bb.x1 > ancho + _TOL_PX or bb.y1 > alto + _TOL_PX:
            texto = art.get_text() if hasattr(art, "get_text") else "(leyenda)"
            problemas.append(f"{nombre} fuera de la figura (quedaría cortado): «{texto[:60]}»")
    return problemas


def revisar_figura(fig, ejes_categoricos: set[str] | None = None) -> list[str]:
    """Problemas de presentación: ejes sin etiqueta o sin unidad entre paréntesis, palabras sin tilde o sin ñ y
    textos que se salen del lienzo. Lista vacía = sin problemas."""
    cat = ejes_categoricos or set()
    problemas = []
    for i, ax in enumerate(fig.get_axes()):
        textos = {"x": ax.get_xlabel(), "y": ax.get_ylabel()}
        barra = getattr(ax, "_colorbar", None)
        if barra is not None:
            eje = "y" if barra.orientation == "vertical" else "x"
            textos = {eje: textos[eje]}
        for nombre, etiqueta in textos.items():
            if not etiqueta.strip():
                problemas.append(f"eje {nombre} del panel {i} sin etiqueta")
            elif nombre not in cat and not _UNIDAD.search(etiqueta):
                problemas.append(f"eje {nombre} del panel {i} sin unidad: «{etiqueta}»")
        for etiqueta in list(textos.values()) + [ax.get_title()]:
            for patron, bien in _SIN_TILDE.items():
                if re.search(patron, etiqueta, flags=re.IGNORECASE):
                    problemas.append(f"panel {i}: «{etiqueta}» debería decir «{bien}»")
    return problemas + _textos_fuera(fig)


def guardar_figura(fig, ruta: Path) -> Path:
    """Guarda en PNG sin metadatos variables (idéntico byte a byte entre corridas) y cierra la figura.

    Solo PNG: PDF y SVG incrustan la fecha de creación.
    """

    ruta = Path(ruta)
    if ruta.suffix.lower() != ".png":
        raise ValueError(f"Solo se guardan figuras PNG (deterministas); recibido {ruta.suffix!r}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, metadata={"Software": None})
    plt.close(fig)
    return ruta
