"""Constantes de la F5 fijadas en el contrato v6 (aceptado el 2026-11-17). Cada una cita su sección.

Nada de aquí se ajusta después de leer valores observados.
"""

from __future__ import annotations

from dataclasses import dataclass

NODO = (-15.0, -75.8)
RADIO_KM = 200.0  # §3.2
COSTA_MIN_KM = 50.0  # §3.2; Ribal y Young (2019)
COSTA_MIN_SENS_KM = 25.0  # §3.2, sensibilidad
PROF_MIN_M = 100.0  # §3.2, como en la F1
SECTORES_VALIDOS = ("abierto", "parcial")  # §3.2, como en la F1
CORTE_PASADA_S = 60.0  # como en la F1 y la F4
MIN_PUNTOS_SUPEROBS = 3  # §3.2
MIN_REGISTROS_PASO_BOYA = 2  # §3.2
VENTANA_BOYA_H = 1.5  # [t − 1.5 h, t + 1.5 h), §3.2
CUANTILES = {"P50": 0.50, "P90": 0.90, "P99": 0.99, "P99.9": 0.999}  # §3.4
B_BOOT = 9_999  # §3.4 (prueba observada y escalón del modelo)
TOLERANCIA_TRASLADO = 1.0 / 3.0  # §3.7, fracción del efecto relevante
FRACCION_MIN_ESCALON = 1.0 / 3.0  # §4.4: |S_mod| < ⅓ del efecto del nodo → la fuente no atribuye

# Directorio de cada misión en raw/altimetria/ribal_young/ → nombre en el contrato
MISIONES = {"GEOSAT": "GEOSAT", "ERS-1": "ERS-1", "TOPEX": "TOPEX", "HY-2": "HY-2A", "HY-2B": "HY-2B"}

# Tramos de calibración (§3.5; Ribal y Young, 2019, tabla «Online-only» 2). Entre 2013-04-01 y 2013-04-15 HY-2A no
# tiene calibración: esas mediciones no entran (se cuentan).
TRAMOS_CALIBRACION = (
    ("GEOSAT", "GEOSAT", "1985-03-31", "1989-12-30"),
    ("ERS-1", "ERS-1", "1991-08-01", "1996-06-02"),
    ("TOPEX", "TOPEX", "1992-09-25", "1997-04-25"),
    ("HY-2A", "HY-2A_1", "2011-10-01", "2013-04-01"),
    ("HY-2A", "HY-2A_2", "2013-04-15", "2018-06-06"),
    ("HY-2B", "HY-2B", "2018-11-24", "2026-01-01"),
)


@dataclass(frozen=True)
class Lado:
    """Un lado de una comparación: qué misiones o boya y en qué período (fechas inclusivas, UTC)."""

    nombre: str
    fuente: str  # «altimetría», «32302» o «32012»
    partes: tuple[tuple[str, str, str], ...]  # (misión o boya, inicio, fin)


def _alt(nombre, *partes):
    return Lado(nombre, "altimetría", tuple(partes))


# §3.4: lados de las comparaciones (principal y sensibilidades). Las fechas son inclusivas, de día completo.
LADOS = {
    # Altimetría
    "T0": _alt("T0", ("GEOSAT", "1985-03-01", "1988-08-31")),
    "T0+ERS1+TOPEX": _alt(
        "T0+ERS1+TOPEX",
        ("GEOSAT", "1985-03-01", "1988-08-31"),
        ("ERS-1", "1991-01-01", "1992-12-31"),
        ("TOPEX", "1991-01-01", "1992-12-31"),
    ),
    "A": _alt("A", ("HY-2A", "2011-10-11", "2018-06-05"), ("HY-2B", "2018-11-24", "2023-03-31")),
    "A_hasta_2026": _alt("A_hasta_2026", ("HY-2A", "2011-10-11", "2018-06-05"), ("HY-2B", "2018-11-24", "2026-01-01")),
    "A_solo_HY2A": _alt("A_solo_HY2A", ("HY-2A", "2011-10-11", "2016-02-29")),
    "A1": _alt("A1", ("HY-2A", "2011-10-11", "2016-02-29")),
    "A2": _alt("A2", ("HY-2A", "2016-03-01", "2018-06-05")),
    "A2+HY2B_2023": _alt("A2+HY2B_2023", ("HY-2A", "2016-03-01", "2018-06-05"), ("HY-2B", "2018-11-24", "2023-03-31")),
    "A2+HY2B_2026": _alt("A2+HY2B_2026", ("HY-2A", "2016-03-01", "2018-06-05"), ("HY-2B", "2018-11-24", "2026-01-01")),
    # Boyas
    "T0b": Lado("T0b", "32302", (("32302", "1986-01-01", "1990-12-31"),)),
    "T0b_hasta_1992": Lado("T0b_hasta_1992", "32302", (("32302", "1986-01-01", "1992-12-31"),)),
    "T1b": Lado("T1b", "32302", (("32302", "1993-02-01", "1995-12-31"),)),
    "A1b": Lado("A1b", "32012", (("32012", "2007-01-01", "2016-02-29"),)),
    "A2b": Lado("A2b", "32012", (("32012", "2016-03-01", "2018-12-31"),)),
}


@dataclass(frozen=True)
class Comparacion:
    """Un contraste de atribución (§4.1, §4.2): fuente, lado 1, lado 2, efecto del nodo y si es la principal."""

    nombre: str
    familia: str  # V1 o V2
    fuente: str  # «a» (altimetría) o «b» (boya)
    lado1: str
    lado2: str
    principal: bool


COMPARACIONES = (
    Comparacion("V1a", "V1", "a", "T0", "A", True),
    Comparacion("V1a_ERS1_TOPEX", "V1", "a", "T0+ERS1+TOPEX", "A", False),
    Comparacion("V1a_A_hasta_2026", "V1", "a", "T0", "A_hasta_2026", False),
    Comparacion("V1a_A_solo_HY2A", "V1", "a", "T0", "A_solo_HY2A", False),
    Comparacion("V1b", "V1", "b", "T0b", "T1b", True),
    Comparacion("V1b_hasta_1992", "V1", "b", "T0b_hasta_1992", "T1b", False),
    Comparacion("V2a", "V2", "a", "A1", "A2", True),
    Comparacion("V2a_HY2B_2023", "V2", "a", "A1", "A2+HY2B_2023", False),
    Comparacion("V2a_HY2B_2026", "V2", "a", "A1", "A2+HY2B_2026", False),
    Comparacion("V2b", "V2", "b", "A1b", "A2b", True),
)

# Efecto del nodo por familia (§4.4): de pruebas_cambio.json de la F4, δ_rel corregido, familia A.
# V1: C1, lado 1 (T0, +9 %); V2: C2, lado 1 (T1, +3 %). El código los lee; estas son las rutas.
EFECTO_NODO_RUTA = {"V1": ("A", "C1", "lado1"), "V2": ("A", "C2", "lado1")}

# --- Potencia (§5) y errores de la observación (§3.5) ---------------------------------------------------------

# Calibración por tramo (§3.5): (σ_a, σ_b en m). Ribal y Young (2019), tabla «Online-only» 2, límites del 95 % / 3.92.
# HY-2B: supuesto de P8, 2 % de pendiente.
CALIBRACION = {
    "GEOSAT": (0.0186, 0.045),
    "ERS-1": (0.0143, 0.033),
    "TOPEX": (0.0084, 0.015),
    "HY-2A_1": (0.0084, 0.017),
    "HY-2A_2": (0.0043, 0.0087),
    "HY-2B": (0.02, 0.0),
}
SIGMA_ESTADO_MAR = 0.01  # mínimo por tramo de calibración (S-16)
SIGMA_RED_EPOCA = 0.01  # red de boyas de referencia por época (1985–1992 y 2011–2026)
SIGMA_BOYA_ANIO = 0.02  # factor por año de calendario de las boyas 32302 y 32012
SIGMA_BOYA_LADO = 0.02  # término por lado de las boyas (sensibilidad obligatoria, S-14)
RUIDO_1HZ = (0.10, 0.05)  # σ de 1 Hz del generador: 0.10 m + 5 % de Hs
RUIDO_BOYA_REL = 0.06  # σ relativa de muestreo por registro horario (Donelan y Pierson, 1983)

SIMEX_LAMBDAS = (0.5, 1.0, 1.5, 2.0)
SIMEX_REPLICAS = 200
B_POTENCIA = 999
REPLICAS_POTENCIA = 1_000
ALFA_FUENTE = 0.025  # Bonferroni por fuente (P12)
ALFA_FAMILIA = 0.05
MULTIPLOS_EFECTO = (0.5, 1.0, 1.5, 2.0, 3.0)  # y de signo contrario (§5)
EFECTO_V3 = 0.019802627296179712  # log(1.02), §4.4

# Rejilla de supuestos (§5, P9): SI del lado asimilado / de T0, ρ y factor de calibración y ruido.
SI_NOMINAL = {"asimilado": 0.12, "T0": 0.16}
SI_REJILLA = {"asimilado": (0.09, 0.12, 0.16), "T0": (0.12, 0.16, 0.20)}
RHO_NOMINAL, RHO_REJILLA = 0.8, (0.5, 0.8)
FACTOR_CAL_REJILLA = (0.5, 1.0, 2.0)
