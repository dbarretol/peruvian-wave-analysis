"""Constantes de la etapa 3 de la F5. Cada una cita la sección del contrato que la fija.

Las marcadas **«precisión de implementación (§20.7)»** concretan algo que el contrato deja abierto. Se fijan antes de
leer valores observados y van a la propuesta de §20.7 del contrato, que acepta el usuario.
"""

from __future__ import annotations

import math

from ..validacion import config as CV

# --- Control de calidad (§3.3) ---------------------------------------------------------------------------------
HS_RANGO = (0.0, 25.0)  # (0, 25] m, altimetría y boyas
PERIODO_RANGO = (1.0, 30.0)  # (1, 30] s, DPD y APD
PICO_MEDIANA_H = 3.0  # |Hs − mediana de ±3 h|
PICO_MAD_H = 12.0  # > 4·(1.4826·MAD de ±12 h)
PICO_K = 4.0
FRACCION_QC_RECALCULO = 0.01  # si un filtro de la etapa 3 quita más del 1 % de un lado, la potencia se recalcula

# --- Ruido de 1 Hz (§3.5) -----------------------------------------------------------------------------------
NIVELES_HS = (2.0, 4.0)  # < 2, 2–4, ≥ 4 m
DT_CONSECUTIVO_S = 1.5  # precisión de implementación (§20.7): «consecutivos» = misma pasada y a ≤ 1.5 s
MIN_DIFERENCIAS_NIVEL = 30  # precisión (§20.7): con menos diferencias en un nivel, se usa el σ de toda la misión
SENS_SIGMA_ALT = (0.5, 2.0)  # σ × 0.5 y σ × 2 (§3.5)
SENS_SIGMA_BOYA = (0.04, 0.08)  # 4 % y 8 % por registro (§3.5)

# --- Representatividad (§3.5 y §20.3) -----------------------------------------------------------------------
R_ESTRUCTURA_KM = (20.0, 50.0)

# --- Traslado por el gradiente (§3.7) -----------------------------------------------------------------------
DIST_COSTA_NODO_KM = 23.0
TOL_GRADIENTE = math.log(1.02)
N_BANDAS_GRADIENTE = 4  # precisión (§20.7): bandas de distancia a la costa por cuartiles de las colocaciones de A

# --- V2 establecido: la corrección por lado absorbe el escalón del nodo (§6.1, condición iii) --------------
TOL_V2_ABSORBE = math.log(1.03)

# --- Períodos (§3.6) ----------------------------------------------------------------------------------------
TOL_PERIODO_S = 0.3
CUANTILES_PERIODO = {"P90": 0.90, "P99": 0.99}

# --- Corrección (§6.2) ---------------------------------------------------------------------------------------
INICIO_ASIMILADO = "1993-01-01"  # precisión (§20.7): el tramo asimilado empieza con la asimilación (QUID, tabla 3)
# Prueba de la serie corregida en los tres cambios (§6.2): (nombre, fin del lado 1, inicio del lado 2, fin del lado 2)
CAMBIOS_SERIE = (
    ("1993-02", ("1980-01-01", "1992-12-31"), ("1993-02-01", "2016-02-29")),
    ("2011-10", ("1993-02-01", "2011-09-30"), ("2011-10-01", "2016-02-29")),
    ("2016-03", ("1993-02-01", "2016-02-29"), ("2016-03-01", "2026-08-31")),
)
TOL_ESCALON_CORREGIDO = 0.001  # precisión (§20.7): |S| corregido no puede superar |S| sin corregir en más de esto

# --- Fila de la fuente (§6.3) -------------------------------------------------------------------------------
C06 = (-0.027, 0.0)  # rango de C-06, «fuera del rango de Hs del sitio»

# --- Sesgo del estimador con el σ medido (§20.5, punto 3; descriptivo) --------------------------------------
REPLICAS_SESGO = 200

# --- Coherencia dentro de T0 (§4.1): ERS-1 y TOPEX antes de 1993 frente a GEOSAT ----------------------------
LADO_ERS1_TOPEX = CV.Lado(
    "ERS1+TOPEX", "altimetría", (("ERS-1", "1991-01-01", "1992-12-31"), ("TOPEX", "1991-01-01", "1992-12-31"))
)
# §20.9 (S-29), posterior a los datos y descriptivo: la forma de V3b por misión en el lado A
LADOS_MISION_A = {
    "A_HY2A": CV.Lado("A_HY2A", "altimetría", (("HY-2A", "2011-10-11", "2018-06-05"),)),
    "A_HY2B": CV.Lado("A_HY2B", "altimetría", (("HY-2B", "2018-11-24", "2023-03-31"),)),
}
ESTADISTICOS_COHERENCIA = ("P50", "P90", "max")  # precisión (§20.7): los de T0 que se esperan estimables (§3.4)

# --- Inventario de observaciones (§3.1) ---------------------------------------------------------------------
OBSERVACIONES = (
    {
        "fuente": "altimetría",
        "observacion": "GEOSAT",
        "periodo": "1985-03-31 a 1988-08-02 (datos locales)",
        "tramo_f4": "T0",
        "independencia_directa": "independiente",
        "independencia_indirecta": "independiente (anterior a la asimilación de ERA5)",
        "clase": "principal de T0",
        "causa": "Anterior a toda asimilación del producto y de ERA5",
        "cita": "QUID v1.5 §VI.4; Hersbach et al. (2019), p. 4; Ribal y Young (2019)",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "altimetría",
        "observacion": "ERS-1 antes de 1993",
        "periodo": "1991-12-20 a 1992-12-31",
        "tramo_f4": "T0",
        "independencia_directa": "dependiente por el forzamiento: ERA5 asimila su Hs desde 1991 (criterio a)",
        "independencia_indirecta": "época con otro producto (criterio b)",
        "clase": "sensibilidad",
        "causa": "Criterios (a) y (b) de §3.1",
        "cita": "Hersbach et al. (2019), p. 4",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "altimetría",
        "observacion": "TOPEX antes de 1993",
        "periodo": "1992-10-14 a 1992-12-31",
        "tramo_f4": "T0",
        "independencia_directa": "no consta que ERA5 la asimile",
        "independencia_indirecta": "época con otro producto (criterio b)",
        "clase": "sensibilidad",
        "causa": "Criterio (b) de §3.1",
        "cita": "Hersbach et al. (2019), p. 4",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "altimetría",
        "observacion": "HY-2A",
        "periodo": "2011-10-11 a 2018-06-05",
        "tramo_f4": "T1 (hasta 2016-02) y T2",
        "independencia_directa": "independiente (no asimilada)",
        "independencia_indirecta": "no consta que ERA5 la asimile (lista no verificada de forma exhaustiva)",
        "clase": "principal",
        "causa": "El QUID la declara no asimilada y la usa para validar",
        "cita": "QUID v1.5 §I.2 y §IV.1.1",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "altimetría",
        "observacion": "HY-2B",
        "periodo": "2018-11-24 a 2023-03-31 (principal); a 2026-01-01 (sensibilidad)",
        "tramo_f4": "T2, T3 y T4",
        "independencia_directa": "independiente hasta 2023-03; desde 2023-04, sin documento del tramo interino",
        "independencia_indirecta": "no consta que ERA5 la asimile",
        "clase": "principal hasta 2023-03; sensibilidad hasta 2026-01 (v6)",
        "causa": "El QUID la usa como referencia independiente en 2020–2022",
        "cita": "QUID v1.5 §VI.2–§VI.4",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "boya",
        "observacion": "NDBC 32302",
        "periodo": "1986 a 1995",
        "tramo_f4": "T0 y T1",
        "independencia_directa": "independiente (las boyas no se asimilan en WAVERYS)",
        "independencia_indirecta": "ERA5 asimila presión y viento convencionales a los dos lados",
        "clase": "principal de C1 en boyas (sin 1991–1992, criterio b)",
        "causa": "Mismo instrumento a los dos lados del cambio",
        "cita": "Página de estación de NDBC guardada el 2026-11-17",
        "licencia": "Datos del Gobierno de EE. UU., dominio público (sin verificar)",
    },
    {
        "fuente": "boya",
        "observacion": "NDBC 32012",
        "periodo": "2007 a 2018",
        "tramo_f4": "T1 y T2",
        "independencia_directa": "independiente (las boyas no se asimilan en WAVERYS)",
        "independencia_indirecta": "ídem",
        "clase": "principal de C2 en boyas; no atribuye C2 por la regla del ⅓ (§19)",
        "causa": "Fondeo Stratus de WHOI, con recambios periódicos. Pudo haber sido vista antes del contrato (§0.1)",
        "cita": "Página de estación de NDBC guardada el 2026-11-17",
        "licencia": "Datos del Gobierno de EE. UU., dominio público (sin verificar)",
    },
    {
        "fuente": "boya",
        "observacion": "NDBC 32301",
        "periodo": "1984 a 1986",
        "tramo_f4": "T0",
        "independencia_directa": "—",
        "independencia_indirecta": "—",
        "clase": "sin uso",
        "causa": "A unos 3300 km, en otra cuenca de generación; solo T0",
        "cita": "Página de estación de NDBC guardada el 2026-11-17",
        "licencia": "—",
    },
    {
        "fuente": "altimetría",
        "observacion": "Saral y GFO",
        "periodo": "2013-03-17 a 2016-02-29 y 2006-08-02 a 2008-09-03",
        "tramo_f4": "T1",
        "independencia_directa": "se comportan como asimiladas (F4, S-34 c)",
        "independencia_indirecta": "—",
        "clase": "excluida",
        "causa": "F4, S-34 c (rotulado)",
        "cita": "Contrato de la F4",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "altimetría",
        "observacion": "Sentinel-3B, Sentinel-6A, SWOT y nadir de CFOSAT",
        "periodo": "varios",
        "tramo_f4": "T2–T4",
        "independencia_directa": "no disponible: ningún documento del productor dice que no se asimilan",
        "independencia_indirecta": "—",
        "clase": "excluida",
        "causa": "Búsqueda de P4 (§15)",
        "cita": "Contrato de la F5, §15",
        "licencia": "CC BY 4.0 (IMOS/AODN)",
    },
    {
        "fuente": "altimetría",
        "observacion": "Misiones asimiladas en su período",
        "periodo": "—",
        "tramo_f4": "T1–T4",
        "independencia_directa": "dependiente (asimilada)",
        "independencia_indirecta": "—",
        "clase": "excluida",
        "causa": "No independiente",
        "cita": "QUID v1.5, tabla 3",
        "licencia": "—",
    },
    {
        "fuente": "boya",
        "observacion": "DHN (Perú), Copernicus In Situ 013_045 y SAR de Envisat",
        "periodo": "—",
        "tramo_f4": "—",
        "independencia_directa": "no disponible",
        "independencia_indirecta": "—",
        "clase": "no disponible",
        "causa": "Sin series encontrables; PDF sin texto; fuera del alcance (§15)",
        "cita": "Contrato de la F5, §15",
        "licencia": "—",
    },
)
NOTA_32012 = (
    "Una comparación anterior de un modelo con la boya 32012 pudo haber sido vista por el usuario antes del contrato "
    "(§0.1; texto completo en el contrato §0.1 y §20.9). No cambia lo fijado: la regla, los efectos y la potencia ya "
    "estaban fijados, y la boya sola no retira escenarios."
)
