"""Cronología ampliada del producto, tramos y cobertura de altimetría (contrato de la F4, §3.1 y §3.2).

La cronología parte de la de la F1 y añade las filas de §3.1. Cada fila lleva referencia y estado: «verificada»,
«inferida» o «no disponible» con su causa (G3). La F4 no modifica la salida de la F1.
"""

from __future__ import annotations

import pandas as pd

QUID = "CMEMS-GLO-QUID-001-032 v1.5 (31/05/2024)"
LAW = "Law-Chune et al. (2021), Ocean Dynamics"
PUM = "CMEMS-GLO-PUM-001-032 v1.6"

# (fecha, tipo, descripción, referencia, estado, tramo al que da origen)
FILAS_F4 = (
    (
        "1980-01-01",
        "configuración",
        "Corrida libre: sin asimilación y sin corrientes, configuración V1.2 (cy42)",
        f"{QUID} §VI.4",
        "verificada",
        "T0",
    ),
    (
        "1993-01-01",
        "transición",
        "Mes de transición: mezcla lineal de la corrida libre y la asimilada",
        f"{QUID} §VI.4",
        "verificada",
        "fuera de toda comparación",
    ),
    (
        "1993-01-01",
        "asimilación",
        "Empiezan la asimilación de Hs de altímetros y las corrientes de GLORYS12",
        f"{QUID} §II, §VI.4",
        "verificada",
        "T1",
    ),
    (
        "1993-01-01",
        "código",
        "El código pasa a IFS-38R2 (cy38) en el período asimilado original",
        f"{LAW} §2.1",
        "verificada",
        "T1",
    ),
    (
        "1993-01-01",
        "reprocesado",
        "Altimetría de las primeras misiones (1993–2005) reprocesada en la V1.1 para "
        "quitar un sesgo de unos −15 cm global, visible sobre todo en 1993–1997",
        f"{QUID} §VI.1.1",
        "verificada",
        "",
    ),
    (
        "2002-01-15",
        "corrección",
        "Error de asimilación de Jason-1, Jason-2 y Cryosat (2002–2016), corregido en la V1.1",
        f"{QUID} §VI.1.2",
        "verificada",
        "",
    ),
    (
        "2003-06-22",
        "cobertura",
        "ERS-2 pierde el registrador de a bordo; sin adquisición a ≤ 600 km del nodo hasta 2011-04-30",
        "ESA (informe cíclico de GOME, Earth Online, nota de 2005-04-21) y geometría",
        "inferida",
        "",
    ),
    (
        "2011-05-01",
        "cobertura",
        "Estación de Cuiabá: exposición a ERS-2 incompleta hasta 2011-07-04",
        "ESA (Earth Online) y geometría",
        "inferida",
        "",
    ),
    (
        "2016-03-01",
        "asimilación",
        "SAR de Sentinel-1A; entran Saral y Sentinel-3A",
        f"{QUID} §II y tabla 3",
        "verificada",
        "T2",
    ),
    ("2018-08-01", "asimilación", "SAR de Sentinel-1B", f"{LAW} §2.3.2", "verificada", "T2b (descriptivo)"),
    (
        "2020-01-01",
        "código",
        "Cambio de código cy38 → cy42 (V1.2, extensión de 2020)",
        f"{QUID} «Important notice» 2, §VI.2",
        "verificada",
        "T3",
    ),
    ("2020-02-01", "asimilación", "SAR de CFOSAT/SWIM", f"{QUID} §II, tabla 3", "verificada", ""),
    (
        "2020-01-01",
        "corrección",
        "Dos anomalías de 2020–2021 corregidas en la entrega 202411 (la F3 comprobó que la serie trae la corrección)",
        f"{QUID} §VI.4.1.2",
        "verificada",
        "",
    ),
    ("2023-04-01", "tramo", "Tramo interino", f"{QUID} §VI.3.2", "verificada", "T4"),
    (
        "1980-01-01",
        "forzamiento",
        "Cambios del sistema de observación de ERA5 (forzamiento de viento)",
        "no disponible: los documentos del producto no los describen",
        "no disponible",
        "",
    ),
    ("2025-05-30", "entrega", "Fusión de los datasets multianual e interino", f"{PUM}", "verificada", ""),
)
COLUMNAS = ["fecha", "tipo", "descripcion", "referencia", "estado", "tramo", "origen"]


def cronologia(f1: pd.DataFrame) -> pd.DataFrame:
    """Filas de la F1 (con estado «verificada» si traen referencia) más las de la F4, ordenadas por fecha."""
    a = f1.copy()
    a["estado"] = a["referencia"].fillna("").map(lambda r: "verificada" if r.strip() else "no disponible")
    a["tramo"] = ""
    a["origen"] = "F1"
    b = pd.DataFrame(FILAS_F4, columns=COLUMNAS[:-1])
    b["origen"] = "F4"
    out = pd.concat([a[COLUMNAS], b[COLUMNAS]], ignore_index=True)
    out["fecha"] = pd.to_datetime(out["fecha"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["fecha", "origen", "tipo"], kind="stable").reset_index(drop=True)


def tramos(tramos_cfg, subtramos, estaciones: dict) -> pd.DataFrame:
    filas = []
    for nombre, a, b, desc in list(tramos_cfg) + list(subtramos):
        ini, fin = pd.Timestamp(a), pd.Timestamp(b)
        fila = {"tramo": nombre, "inicio": a, "fin": b, "descripcion": desc, "dias": int((fin - ini).days + 1)}
        meses = pd.period_range(ini, fin, freq="M")
        completos = {
            m for m in meses if pd.Timestamp(m.start_time) >= ini and pd.Timestamp(m.end_time).normalize() <= fin
        }
        anios = sorted({m.year for m in meses})
        fila["anios_completos"] = sum(
            all(pd.Period(f"{y}-{k:02d}", "M") in completos for k in range(1, 13)) for y in anios
        )
        for est, ms in estaciones.items():
            # año estacional nombrado por el año de su primer mes, en el orden del año de bloque que empieza en él
            n = 0
            for y in range(ini.year - 1, fin.year + 1):
                periodos = [pd.Period(f"{y + (1 if (m < ms[0]) else 0)}-{m:02d}", "M") for m in ms]
                n += all(p in completos for p in periodos)
            fila[f"anios_completos_{est}"] = n
        filas.append(fila)
    return pd.DataFrame(filas)


def cobertura_altimetria(
    locales: pd.DataFrame, asimiladas: dict, sin_datos: dict, no_asimiladas, no_verificables, placebos, huecos
) -> pd.DataFrame:
    """Por misión: período asimilado (tabla 3), datos locales, huecos de exposición y clase."""
    loc = locales.set_index("mision") if len(locales) else pd.DataFrame(columns=["inicio", "fin", "pasadas"])
    misiones = sorted(set(loc.index) | set(asimiladas))
    pl = {m: n for n, m, _, _ in placebos}
    filas = []
    for m in misiones:
        a = asimiladas.get(m)
        if m in asimiladas:
            clase = "asimilada"
        elif m in no_verificables:
            clase = "no verificable (tratada como asimilada; sensibilidad sin ella)"
        elif m in no_asimiladas:
            clase = "no asimilada"
        else:
            clase = "sin uso"
        if m in pl:
            clase += f"; placebo «{pl[m]}»"
        hu = "; ".join(f"{x} a {y}: {mot}" for x, y, mot in huecos if m in mot)
        filas.append(
            {
                "mision": m,
                "asimilada_desde": a[0] if a else "",
                "asimilada_hasta": (a[1] or "en adelante") if a else "",
                "datos_locales_desde": str(pd.Timestamp(loc.at[m, "inicio"]).date()) if m in loc.index else "",
                "datos_locales_hasta": str(pd.Timestamp(loc.at[m, "fin"]).date()) if m in loc.index else "",
                "pasadas": int(loc.at[m, "pasadas"]) if m in loc.index else 0,
                "huecos_de_exposicion": hu,
                "clase": clase,
            }
        )
    if len(loc):
        ultimo = pd.Timestamp(pd.to_datetime(loc["fin"]).max())
        filas.append(
            {
                "mision": "todas",
                "asimilada_desde": "",
                "asimilada_hasta": "",
                "datos_locales_desde": "",
                "datos_locales_hasta": str(ultimo.date()),
                "pasadas": 0,
                "huecos_de_exposicion": f"exposición incompleta después del último dato local ({ultimo})",
                "clase": "hueco de exposición (§3.4 punto 2; §21, K-10)",
            }
        )
    for nombre, inicio in sin_datos.items():
        filas.append(
            {
                "mision": nombre,
                "asimilada_desde": inicio,
                "asimilada_hasta": "en adelante",
                "datos_locales_desde": "",
                "datos_locales_hasta": "",
                "pasadas": 0,
                "huecos_de_exposicion": f"exposición incompleta desde {inicio} (sin datos locales)",
                "clase": "asimilada sin datos locales (espectros)",
            }
        )
    return pd.DataFrame(filas)
