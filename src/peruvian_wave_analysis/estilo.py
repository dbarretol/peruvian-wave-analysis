"""Estilo único de figuras (mínimo para el mapa de sitio; el completo se porta de v2 en la F2, ADR-0008).

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
TIERRA = "#e4e2dc"

# Isóbatas: orden fijo naranja, violeta, verde, magenta, rojo (validado: CVD ΔE ≥ 8 entre vecinos).
COLORES_ISOBATAS = ["#eb6834", "#6250d6", "#008300", "#e87ba4", "#e34948"]
ESTILOS_ISOBATAS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
# Profundidad: rampa secuencial azul, de claro a oscuro.
RAMPA_PROFUNDIDAD = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def aplicar() -> None:
    matplotlib.rcParams.update(RC)
