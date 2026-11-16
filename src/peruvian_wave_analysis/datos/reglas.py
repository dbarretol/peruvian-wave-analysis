"""Reglas de control de calidad de la F3 que no dependen del detector de picos (contrato §3.2, §3.3, §3.6 y §3.7).

Las funciones reciben la serie (``DataFrame`` con índice de tiempo y una columna por variable) y devuelven máscaras
booleanas o tablas; no conocen el sitio. Ninguna descarta registros: el descarte se aplica después, con la
protección de cola y las resoluciones (§3.4, §3.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

# --- §3.3: direcciones --------------------------------------------------------------------------


def normalizar_direcciones(df: pd.DataFrame) -> pd.DataFrame:
    """360.00 → 0, sin marca (F03-S-02): con add_offset 180 y factor 0.01, ≥ 359.995° se guarda como 360.00."""
    out = df.copy()
    for v in C.DIRECCIONES:
        if v in out:
            col = out[v]
            out[v] = col.where(~np.isclose(col, 360.0, rtol=0, atol=1e-9), 0.0)
    return out


# --- §3.2: particiones ausentes ----------------------------------------------------------------


@dataclass
class ResumenAusencia:
    """Conteos de §3.2 por partición (v7): se publican; ninguno detiene la etapa."""

    particion: str
    ausentes: int  # período 0, con cualquier dirección
    incoherencias: int  # período 0 con Hs_i > 0: exige resolución
    bajo_cuanto: int  # Hs_i = 0 con período > 0: marca informativa
    direcciones_distintas_ausentes: int  # valores distintos de dirección (a 0.01°) entre los ausentes
    fraccion_ausentes_180: float | None  # fracción de los ausentes con dirección 180° ± 0.01°


@dataclass
class ResultadoAusentes:
    serie: pd.DataFrame  # período y dirección de las particiones ausentes, a NaN
    marcas: pd.DataFrame  # particion_ausente_<p>, incoherencia_ausente_<p>, bajo_cuanto_<p>, hs_total_cero
    resumen: dict[str, ResumenAusencia] = field(default_factory=dict)


def particiones_ausentes(df: pd.DataFrame) -> ResultadoAusentes:
    """Firma de ausencia por partición (contrato v7, §3.2 y §15; regla posterior a ver las particiones).

    - Período 0, con cualquier dirección → partición ausente: período y dirección a NaN, sin descarte. La altura se
      conserva.
    - Período 0 con Hs_i > 0 → además, incoherencia que exige resolución.
    - Hs_i = 0 con período > 0 → partición por debajo de un cuanto: marca informativa; período y dirección se
      conservan, porque son salida real del modelo.
    """
    out = df.copy()
    marcas = pd.DataFrame(index=df.index)
    resumen: dict[str, ResumenAusencia] = {}
    for p, (h, t, d) in C.PARTICIONES.items():
        ausente = df[t] == 0.0
        incoherente = ausente & (df[h] > 0.0)
        bajo = (df[h] == 0.0) & (df[t] > 0.0)
        marcas[f"particion_ausente_{p}"] = ausente
        marcas[f"incoherencia_ausente_{p}"] = incoherente
        marcas[f"bajo_cuanto_{p}"] = bajo
        dirs = df.loc[ausente, d]
        resumen[p] = ResumenAusencia(
            p,
            int(ausente.sum()),
            int(incoherente.sum()),
            int(bajo.sum()),
            int(dirs.round(2).nunique()),
            float(np.isclose(dirs, 180.0, rtol=0, atol=0.01).mean()) if len(dirs) else None,
        )
        out.loc[ausente, [t, d]] = np.nan  # sin dato de período ni dirección
    marcas["hs_total_cero"] = df["VHM0"] == 0.0
    return ResultadoAusentes(out, marcas, resumen)


def periodos_bajo_resolucion(df: pd.DataFrame) -> dict[str, dict]:
    """Períodos medios de partición por debajo de 1/f_max (dato del producto, sin interpretar; v7, §15).

    Por partición: registros con 0 < T < 1/f_max y con 0 < T ≤ 1.5 s, cuantiles de T y de Hs_i en ese grupo, y los
    valores de T más frecuentes.
    """
    t_res = 1.0 / C.F_MAX_HZ
    out = {}
    for p, (h, t, _) in C.PARTICIONES.items():
        T, H = df[t], df[h]
        m = (T > 0) & (T < t_res)
        q = [0.0, 0.5, 0.95, 1.0]
        out[p] = {
            "t_resolucion_s": t_res,
            "n_bajo_resolucion": int(m.sum()),
            "n_hasta_1_5_s": int(((T > 0) & (T <= 1.5)).sum()),
            "cuantiles_t_s": {str(x): float(T[m].quantile(x)) for x in q} if m.any() else None,
            "cuantiles_hs_m": {str(x): float(H[m].quantile(x)) for x in q} if m.any() else None,
            "valores_t_mas_frecuentes": {f"{k:.2f}": int(v) for k, v in T[m].round(2).value_counts().head(10).items()},
        }
    return out


def marcas_sobre_tope(conteos: dict[str, int]) -> list[str]:
    """Motivos de «revisión requerida» de las marcas que exigen resolución con más de TOPE_MARCA registros (v7)."""
    return [
        f"{n} registros con la marca {regla} > {C.TOPE_MARCA}"
        for regla, n in sorted(conteos.items())
        if n > C.TOPE_MARCA
    ]


# --- §3.2: acumulaciones, discreción y saturación ----------------------------------------------


def acumulaciones(valores: pd.Series | np.ndarray, cuanto: float = C.CUANTO) -> list[float]:
    """Valores con frecuencia ≥ ACUM_FACTOR × mediana de los ACUM_VECINOS cuantos a cada lado y ≥ ACUM_MIN registros."""
    v = np.asarray(valores, dtype=float)
    q = np.round(v[np.isfinite(v)] / cuanto).astype(np.int64)
    if q.size == 0:
        return []
    cuantos, cuenta = np.unique(q, return_counts=True)
    frec = dict(zip(cuantos.tolist(), cuenta.tolist(), strict=True))
    out = []
    for c, n in zip(cuantos.tolist(), cuenta.tolist(), strict=True):
        if n < C.ACUM_MIN:
            continue
        vecinos = [frec.get(c + k, 0) for k in range(-C.ACUM_VECINOS, C.ACUM_VECINOS + 1) if k != 0]
        if n >= C.ACUM_FACTOR * float(np.median(vecinos)):
            out.append(round(c * cuanto, 10))
    return out


def acumulaciones_discretas(valores: pd.Series | np.ndarray) -> list[float]:
    """Para una variable discreta (§3.2): las «vecinas» de un valor son los valores distintos adyacentes."""
    v = np.asarray(valores, dtype=float)
    distintos, cuenta = np.unique(np.round(v[np.isfinite(v)], 2), return_counts=True)
    out = []
    for i, n in enumerate(cuenta):
        if n < C.ACUM_MIN:
            continue
        vecinos = np.r_[cuenta[max(0, i - C.ACUM_VECINOS) : i], cuenta[i + 1 : i + 1 + C.ACUM_VECINOS]]
        if vecinos.size and n >= C.ACUM_FACTOR * float(np.median(vecinos)):
            out.append(float(distintos[i]))
    return out


def es_discreto(valores: pd.Series | np.ndarray) -> bool:
    v = np.asarray(valores, dtype=float)
    return int(np.unique(np.round(v[np.isfinite(v)], 2)).size) <= C.DISCRETO_MAX_DISTINTOS


def razones_consecutivas(valores: pd.Series | np.ndarray) -> np.ndarray:
    """Razón entre valores distintos consecutivos (para comparar con el QUID; §3.2)."""
    v = np.asarray(valores, dtype=float)
    d = np.unique(np.round(v[np.isfinite(v)], 2))
    return d[1:] / d[:-1] if d.size > 1 else np.array([])


def saturados(valores: pd.Series, discreto: bool) -> pd.Series:
    """Período saturado (censurado), marca sin descarte (§3.2, F03-S-06 y F03-S-16)."""
    t_max = 1.0 / C.F_MIN_HZ
    if discreto:
        finitos = valores.dropna()
        if finitos.empty:
            return pd.Series(False, index=valores.index)
        mayor = float(np.round(finitos, 2).max())
        if mayor < C.T_SATURACION_DISCRETO:
            return pd.Series(False, index=valores.index)
        return pd.Series(np.isclose(valores, mayor, atol=0.005), index=valores.index)
    return (valores >= t_max - C.TOL_SATURACION_CONTINUO).fillna(False)


# --- §3.3: límites físicos y peralte ------------------------------------------------------------


def fuera_de_limites(df: pd.DataFrame) -> pd.DataFrame:
    """Una columna por regla de §3.3; True = fuera de límite (candidato a descarte automático).

    Los períodos y direcciones ausentes (particiones ausentes, §3.2) no se evalúan.
    """
    out = pd.DataFrame(index=df.index)
    for h in C.ALTURAS:
        x = df[h]
        out[f"limite_{h}"] = x.notna() & ((x < 0.0) | (x > C.HS_MAX))
    for t in C.PERIODOS:
        x = df[t]
        out[f"limite_{t}"] = x.notna() & ((x <= 0.0) | (x > C.T_MAX))
    for d in C.DIRECCIONES:
        x = df[d]
        out[f"limite_{d}"] = x.notna() & ((x < 0.0) | (x > 360.0))
    return out


def limite_peralte(tp: pd.Series | np.ndarray) -> np.ndarray:
    """S_p límite: 1/15 para Tp ≤ 8 s, 1/25 para Tp ≥ 15 s, lineal entre ambos (DNVGL-RP-C205 2017, §3.5.4)."""
    return np.interp(np.asarray(tp, dtype=float), C.PERALTE_TP, C.PERALTE_LIM)


def peralte(hs: pd.Series, tp: pd.Series) -> pd.Series:
    return 2.0 * np.pi * hs / (C.G * tp**2)


def peralte_excedido(df: pd.DataFrame) -> pd.Series:
    """Marca que exige resolución, no descarte (§3.3)."""
    sp = peralte(df["VHM0"], df["VTPK"])
    return (sp > limite_peralte(df["VTPK"])) & sp.notna()


# --- §3.6: coherencia ---------------------------------------------------------------------------


def coherencia(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas: particion_mayor (resolución), energia (informativa), tm10_menor_tm02 (resolución), tp_menor_tm02
    (informativa). Más ``discrepancia_energia`` (valor continuo, para publicar su distribución)."""
    out = pd.DataFrame(index=df.index)
    hs = df["VHM0"]
    parts = [df[h] for h, _, _ in C.PARTICIONES.values()]
    out["particion_mayor"] = pd.concat([(p - hs) > C.TOL_PARTICION_M + 1e-9 for p in parts], axis=1).any(axis=1)
    suma = sum(p.fillna(0.0) ** 2 for p in parts)
    disc = (hs**2 - suma).abs() / hs**2
    out["discrepancia_energia"] = disc.where(hs >= C.HS_MIN_ENERGIA)
    out["energia"] = (hs >= C.HS_MIN_ENERGIA) & (disc > C.TOL_ENERGIA)
    out["tm10_menor_tm02"] = (df["VTM02"] - df["VTM10"]) > C.TOL_TM10_TM02 + 1e-9
    out["tp_menor_tm02"] = df["VTPK"] < df["VTM02"] - C.TOL_TP_TM02
    return out


# --- §3.7: cobertura ----------------------------------------------------------------------------


def cobertura(valido: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(por año, por mes): fracción de registros válidos sobre los esperados en la rejilla, con la marca de bloque
    incompleto (< 90 %) y la de año parcial por construcción."""
    v = valido.astype(bool)
    anual = v.groupby(v.index.year).agg(["sum", "count"]).rename(columns={"sum": "validos", "count": "registros"})
    anual["fraccion"] = anual["validos"] / anual["registros"]
    anual["incompleto"] = anual["fraccion"] < C.COBERTURA_MIN
    anual["parcial"] = anual.index == C.ANIO_PARCIAL
    anual.index.name = "anio"
    mensual = v.groupby([v.index.year, v.index.month]).agg(["sum", "count"])
    mensual.columns = ["validos", "registros"]
    mensual["fraccion"] = mensual["validos"] / mensual["registros"]
    mensual["incompleto"] = mensual["fraccion"] < C.COBERTURA_MIN
    mensual.index.names = ["anio", "mes"]
    return anual, mensual


def cobertura_por_variable(serie: pd.DataFrame, valido: pd.Series | np.ndarray, variables) -> pd.DataFrame:
    """`cobertura.csv` (§2, §3.7; F03-K-03): por variable, año y mes, fracción de registros válidos y con dato, con
    la bandera de bloque incompleto y la de año parcial."""
    v_ok = np.asarray(valido, dtype=bool)
    filas = []
    for var in variables:
        ok = pd.Series(v_ok & serie[var].notna().to_numpy(), index=serie.index)
        anual, mensual = cobertura(ok)
        a = anual.reset_index().assign(variable=var, bloque="anio", mes=np.nan)
        m = mensual.reset_index().assign(variable=var, bloque="mes", parcial=lambda d: d["anio"] == C.ANIO_PARCIAL)
        filas += [a, m]
    cols = ["variable", "bloque", "anio", "mes", "validos", "registros", "fraccion", "incompleto", "parcial"]
    return pd.concat(filas, ignore_index=True)[cols]
