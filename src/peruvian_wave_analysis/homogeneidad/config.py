"""Parámetros de la F4, cada uno con la sección del contrato que lo fija (contrato de la F4, v7, aceptado 2026-11-16).

Las funciones del subpaquete no conocen el sitio: reciben estos valores como argumentos. Solo ``run.py`` los junta.
"""

from __future__ import annotations

import pandas as pd

T = pd.Timestamp

# --- §3.2 Tramos -------------------------------------------------------------------------------
# (nombre, inicio, fin inclusive al día, qué lo define). Enero de 1993 (mes de transición) queda fuera.
TRAMOS = (
    ("T0", "1980-01-01", "1992-12-31", "Corrida libre, sin corrientes, cy42"),
    ("T1", "1993-02-01", "2016-02-29", "Altímetros y corrientes, cy38"),
    ("T2", "2016-03-01", "2019-12-31", "+ SAR de Sentinel-1, Saral y Sentinel-3A, cy38"),
    ("T3", "2020-01-01", "2023-03-31", "cy42; + CFOSAT desde 2020-02"),
    ("T4", "2023-04-01", "2026-08-31", "Tramo interino"),
)
# Cortes descriptivos, con IC y sin p-valor (§3.2): 2018-08, 2020-01 y 2023-04. 2020-01 y 2023-04 ya son bordes de
# tramo; 2018-08 parte T2 en dos subtramos descriptivos.
SUBTRAMOS_DESCRIPTIVOS = (
    ("T2a", "2016-03-01", "2018-07-31", "T2 antes del SAR de Sentinel-1B"),
    ("T2b", "2018-08-01", "2019-12-31", "T2 con el SAR de Sentinel-1B"),
)
MES_TRANSICION = ("1993-01-01", "1993-01-31")

# Contrastes confirmatorios (§3.2, §3.5): (lado 1, lado 2) como listas de tramos; mes de inicio del año de contraste.
CONTRASTES = {
    "C1": {"lado1": ("T0",), "lado2": ("T1", "T2", "T3", "T4"), "mes_inicio": 1},
    "C2": {"lado1": ("T1",), "lado2": ("T2", "T3", "T4"), "mes_inicio": 3},
}

# §3.2 Estaciones provisionales (no prejuzgan la F6)
ESTACIONES = {"invierno": (5, 6, 7, 8, 9, 10), "verano": (11, 12, 1, 2, 3, 4)}

# --- §3.3 Cuantiles e indicadores --------------------------------------------------------------
VARIABLES = ("VHM0", "VTPK", "VTM02", "VHM0_WW", "VHM0_SW1", "VHM0_SW2")
CUANTILES = (0.50, 0.90, 0.99, 0.999)
CUANTIL_TP_HS_ALTO = 0.99  # Tp condicionado a Hs ≥ P99 del registro completo
B_IC = 999  # réplicas del bootstrap de bloques anuales para los IC del 95 % (descriptivos)
EQUIVALENCIA_P50 = 0.02  # el IC del cociente de P50 debe caer dentro de ±2 % (v6, F04-S-10 c)
NODO_VECINAS = "-15.0_-75.8"  # columna del nodo en vecinas_hs.parquet; las vecinas son el primer anillo

# --- §3.4 Prueba de altímetros -----------------------------------------------------------------
NODO = (-15.0, -75.8)
RADIO_KM = 200.0
CORTE_PASADA_S = 60.0  # mediciones de una misión separadas más de 60 s son pasadas distintas (como en la F1)
PASO_H = 3.0
VENTANA_CENTRADA = (-1.5, 1.5)  # horas, [t − 1.5 h, t + 1.5 h): principal (v6, F04-S-07 a)
VENTANA_ATRAS = (-3.0, 0.0)  # horas, (t − 3 h, t]: sensibilidad

# Tabla 3 del QUID leída en la imagen (§3.1). None = en adelante. Nombres de carpeta de Ribal y Young.
ASIMILADAS = {
    "ERS-1": ("1993-01-01", "1996-06-02"),
    "TOPEX": ("1993-01-01", "2005-10-08"),
    "ERS-2": ("1995-05-15", "2011-07-04"),
    "GFO": ("2000-01-07", "2006-08-01"),
    "JASON-1": ("2002-01-15", "2013-06-21"),
    "ENVISAT": ("2002-05-14", "2012-04-08"),
    "JASON-2": ("2008-07-04", None),
    "JASON-3": ("2008-07-04", None),
    "CRYOSAT-2": ("2010-07-14", None),
    "SARAL": ("2016-03-01", None),
    "SENTINEL-3A": ("2016-03-01", None),
}
# Fuentes asimiladas sin datos locales (espectros SAR; §3.1 y §3.4 punto 2): exposición incompleta desde su inicio.
ASIMILADAS_SIN_DATOS_LOCALES = {
    "SAR Sentinel-1 (rotulado «SAR CFOSAT/SWIM» en la tabla 3)": "2016-03-01",
    "SAR CFOSAT/SWIM": "2020-02-01",
}
# No asimiladas, confirmado en el QUID (§IV.1.1 y §VI.2)
NO_ASIMILADAS = ("GEOSAT", "HY-2", "HY-2B")
# Independencia por verificar (v3): se tratan como asimiladas en la exposición; sensibilidad sin ellas (§3.4 punto 1)
NO_VERIFICABLES = ("SENTINEL-3B", "SENTINEL-6A", "SWOT", "CFOSAT")
# Huecos de exposición documentados (§3.4 punto 2, v5): ERS-2 con la estación de Cuiabá
HUECOS_EXPOSICION = (("2011-05-01", "2011-07-04", "ERS-2 asimilada con adquisición parcial (estación de Cuiabá)"),)
# Análisis principal: solo pasos con exposición completa (§3.4 punto 2, v5)
VENTANA_PRINCIPAL = (("1993-02-01", "2011-04-30"), ("2011-07-05", "2016-02-29"))
# Épocas de misiones para la estratificación (§3.4 punto 5)
EPOCAS_PRINCIPAL = (("1993-02-01", "2003-06-16"), ("2003-06-17", "2011-04-30"), ("2011-07-05", "2016-02-29"))
EPOCAS_DESPUES = (("2016-03-01", "2019-12-31"), ("2020-01-01", "2026-08-31"))  # T2 y T3–T4 (sensibilidad)
INICIO_SAR = "2016-03-01"

# Umbral de salto (§3.4, v6 y v7)
CUANTIL_C_T0 = 0.99  # c_T0 = P99 de |ΔHs(t)|/Hs(t−1) en T0, solo con pares dentro de T0
NIVELES_HS = (0.50, 0.90)  # estratos de nivel de Hs(t−1): < P50, P50–P90, ≥ P90 del registro completo
K_A = (3.0, 4.5, 6.5)  # sensibilidades absolutas |ΔHs| > k·s (s = 1.4826·MAD(ΔHs), contrato de la F3 §3.4)
UMBRAL_RELATIVO_POSTERIOR = 0.10  # sensibilidad rotulada como posterior a ver los datos (F04-S-07 b)

COCIENTE_ALTERNATIVO = 1.5  # §3.4 punto 6
FRACCION_OCULTA = 1.0 / 3.0  # dilución por pasadas entre 200 y 600 km (§3.6, v6, F04-S-08)

# Placebos (§3.4 punto 8): (nombre, misión, inicio, fin). El de T0 se trata aparte.
PLACEBOS = (
    ("HY-2A", "HY-2", "2011-10-11", "2018-06-05"),
    ("Saral antes de 2016-03", "SARAL", "2013-03-17", "2016-02-29"),
    ("GFO después de 2006-08", "GFO", "2006-08-02", "2008-09-03"),
)
PLACEBO_SENSIBILIDAD = ("HY-2B", "HY-2B", "2018-11-24", "2026-01-02")
PLACEBO_T0 = ("calendario de 1993-02 a 2003-06 desplazado a T0", "1993-02-01", "2003-06-16", 13)  # años de desfase

# --- §3.5 Pruebas de cambio --------------------------------------------------------------------
ALFA = 0.05
B = 9_999  # permutaciones de la prueba observada
RHO_AR1 = 0.3  # supuesto, no estimación (v7)
TIPO_I_MAX = 0.07  # regla de §3.5
POTENCIA_MIN = 0.80
REPLICAS = 1_000  # réplicas de potencia y de error de tipo I (§3.6)
HORAS_MAX_POTENCIA = 8.0  # si la potencia proyectada pasa de 8 h, se reduce B en las réplicas (§3.6) y se declara
B_REDUCIDO = 999

# Efecto relevante (ADR-0006, §3.5)
CAMBIO_RL = 0.02
PERIODOS_RETORNO = (10.0, 50.0, 100.0)
CUANTIL_UMBRAL_PROVISIONAL = 0.99
IET_H = 48.0
REJILLA_DELTA = (-0.30, 0.30, 0.005)

# Rejilla de efectos para la potencia (§3.6): múltiplos de δ_rel, más δ_rel
MULTIPLOS_EFECTO = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

# Familia B (§3.5)
ANIOS_T1 = (1993, 2015)
SIN_PRIMEROS = (1993, 1997)  # sensibilidad sin 1993–1997 (QUID §VI.1.1)

# Familia C (§3.5)
VENTANA_COINCIDE_ANIOS = 1

# Familia D (§3.5)
EFECTO_TP_S = 0.3

# Resultados (ADR-0005)
ESTABLECIDO, DESCARTADO, NO_CONCLUYENTE = "establecido", "descartado", "no concluyente"
SIN_RAIZ = "sin raíz"
SENAL = "señal"
