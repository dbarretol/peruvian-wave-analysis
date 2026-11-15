"""Parámetros de la F3, fijados en su contrato antes de abrir la serie. Cada uno cita su sección."""

from __future__ import annotations

# --- Nodo y celdas (§3.1) ----------------------------------------------------------------------
NODO = (-15.0, -75.8)  # (lat, lon), ADR-0007
TOL_COORD = 1e-6  # coincidencia exacta de coordenadas, en grados
PASO_MALLA = 0.2  # grados entre celdas de WAVERYS
VECINAS_MIN = 5  # mínimo de vecinas del primer anillo con dato para decidir (§3.1, F03-S-03)

# --- Ingesta (§3.1) -----------------------------------------------------------------------------
ESCALA = 0.01  # scale_factor de todas las variables (PUM v1.6)
RELLENO = -32767  # _FillValue (PUM v1.6)
CUANTO = 0.01  # cuanto de cuantización en la unidad física
ALTURAS = ("VHM0", "VHM0_WW", "VHM0_SW1", "VHM0_SW2")
PERIODOS = ("VTPK", "VTM02", "VTM10", "VTM01_WW", "VTM01_SW1", "VTM01_SW2")
DIRECCIONES = ("VMDR", "VMDR_WW", "VMDR_SW1", "VMDR_SW2", "VPED")
DERIVA = ("VSDX", "VSDY")  # deriva de Stokes: se lee y se publica; ninguna regla la usa (ADR-0001)
VARIABLES = ALTURAS + PERIODOS + DIRECCIONES + DERIVA
UNIDADES = {
    **dict.fromkeys(ALTURAS, "m"),
    **dict.fromkeys(PERIODOS, "s"),
    **dict.fromkeys(DIRECCIONES, "degree"),
    **dict.fromkeys(DERIVA, "m s-1"),
}
DESPLAZAMIENTO = {v: (180.0 if v in DIRECCIONES else 0.0) for v in VARIABLES}  # add_offset (PUM v1.6, FAQ 1)
PARTICIONES = {  # partición → (altura, período medio, dirección)
    "WW": ("VHM0_WW", "VTM01_WW", "VMDR_WW"),
    "SW1": ("VHM0_SW1", "VTM01_SW1", "VMDR_SW1"),
    "SW2": ("VHM0_SW2", "VTM01_SW2", "VMDR_SW2"),
}

# --- Centinelas, particiones ausentes y saturación (§3.2) --------------------------------------
FIRMA_FRACCION_MIN = 0.99  # un par (período, dirección) es la firma de ausencia si lo comparte ≥ 99 % (F03-S-05)
ACUM_FACTOR = 20.0  # frecuencia ≥ 20 × mediana de los cuantos vecinos (§8 P3)
ACUM_VECINOS = 10  # cuantos a cada lado
ACUM_MIN = 50  # registros mínimos para una acumulación
DISCRETO_MAX_DISTINTOS = 60  # VTPK es discreto si tiene ≤ 60 valores distintos (F03-S-06)
F_MIN_HZ = 0.035  # primera frecuencia de MFWAM (QUID v1.5, §I y §II)
T_SATURACION_DISCRETO = 27.3  # s; punto medio entre 25.97 y 28.57 s (F03-S-16)
TOL_SATURACION_CONTINUO = 0.05  # s; saturado si T ≥ 1/f_min − 0.05 s

# --- Límites físicos y peralte (§3.3) ----------------------------------------------------------
HS_MAX = 20.0  # m, NDBC (2023) tabla 6
T_MAX = 30.0  # s; períodos en (0, 30]
PERALTE_TP = (8.0, 15.0)  # s; DNVGL-RP-C205 (2017) §3.5.4
PERALTE_LIM = (1.0 / 15.0, 1.0 / 25.0)
G = 9.80665  # m/s²

# --- Coherencia (§3.6) --------------------------------------------------------------------------
TOL_PARTICION_M = 0.01  # Hs_i − Hs > un cuanto
TOL_ENERGIA = 0.10  # |Hs² − ΣHs_i²| / Hs²
HS_MIN_ENERGIA = 0.5  # m
TOL_TM10_TM02 = 0.01  # s
TOL_TP_TM02 = 0.5  # s

# --- Cobertura (§3.7) ---------------------------------------------------------------------------
COBERTURA_MIN = 0.90
ANIO_PARCIAL = 2026  # hasta el 2026-08-31: parcial por construcción
