"""Estilo único de figuras (mínimo para el mapa de sitio; el completo llega en la F2).

Colores: paleta de referencia validada para daltonismo (orden fijo, nunca cíclico).
"""
from __future__ import annotations

import matplotlib

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
RAMPA_PROFUNDIDAD = ["#f2f7fb", "#dbe8f4", "#bfd6eb", "#9fc1df", "#7fa9d0", "#6290bd", "#4a77a3"]  # azul suave: deja leer las zonas


def aplicar() -> None:
    matplotlib.rcParams.update(RC)

# Zonas: un color por familia (verde = SERNANP, pardo = hidrocarburos), estilo limpio:
# relleno tenue + borde, trama solo en áreas protegidas. Exclusión = relleno opaco con borde rojo.
COLOR_BORDE_EXCLUSION = "#d03b3b"
ESTILO_FAMILIA = {
    #                   borde       relleno     opacidad  trama  línea  etiquetar
    "areas_protegidas": {"borde": "#1f7a2e", "relleno": "#9fd3a8", "alfa": 0.45, "trama": "//", "linea": "-", "etiqueta": True},
    "amortiguamiento": {"borde": "#5f8f2a", "relleno": "none", "alfa": 0.0, "trama": None, "linea": ":", "etiqueta": False},
    "aavc": {"borde": "#138a7e", "relleno": "#9fd8cf", "alfa": 0.15, "trama": None, "linea": "--", "etiqueta": True},
    "propuesta": {"borde": "#3f8f3f", "relleno": "none", "alfa": 0.0, "trama": None, "linea": "-.", "etiqueta": True},
    "petroleo": {"borde": "#8b5a1a", "relleno": "#e3cfa6", "alfa": 0.45, "trama": None, "linea": "-", "etiqueta": True},
}
