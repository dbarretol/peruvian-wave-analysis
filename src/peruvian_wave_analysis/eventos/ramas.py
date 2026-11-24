"""Ramas y picos de la F7 (contrato §0 «Ramas», §3.1, §3.2 puntos 5 y 6, §3.4 «Borde inferior», §3.5 «Protección de
cola» y §3.6).

Funciones puras: no leen archivos ni publican. La etapa con datos las llama sobre los eventos de §3.2.

- **Eventos de cada rama** (§3.1): el instante del pico en E0 sin corregir asigna el evento a la rama y al período
  (hasta el 2025-12-31, ADR-0031; E1 desde su inicio). Las ramas por cuantiles toman los mismos eventos que su rama
  sin corregir.
- **Pico de la rama** (§3.2, punto 5; E-16): el máximo de la serie de Hs de la rama dentro de [t_inicio, t_fin] del
  evento (primera aparición; un NaN no puede ser pico). En E1 la ventana se recorta al inicio de E1 (``recortado``).
  ``pico_movido`` marca los picos cuyo instante difiere del de E0 sin corregir.
- **Etiquetas** (§3.2, punto 6): Tp, dirección y partición dominante en el instante del pico de la rama, de la serie
  de E0 (las variantes solo cambian Hs).
- **u_id llevado a la variante** (§3.2, punto 5; §3.4, E-29): por su propio cuantil. Con q0 = fracción de los estados
  de E0 sin corregir ≤ u_id, sobre los estados válidos de la rama, u_id en la variante es su cuantil q0
  (``inverted_cdf``, un valor de la serie). Si el mapa es una función monótona fija, «Hs > u_id» y «Hs_variante > u_id
  llevado» marcan los mismos estados. **Estados discordantes:** los que cumplen una y no la otra. **u_id^max:** el
  máximo de la variante entre los estados con Hs de E0 ≤ u_id; si el mapa cambia en el tiempo, es el máximo en el
  tiempo de u_id llevado.
- **Borde inferior** (E-29): con estados discordantes, solo cuentan los picos de la variante > u_id^max, y N0 es su
  número.
- **Protección de cola** (G8): en las variantes por cuantiles, ningún pico de rama puede caer en un instante con Hs
  de E0 sin corregir < P95 de la F3. Si cae, se lista y la etapa se detiene con «revisión requerida».
- **Exposición** (§3.1): años con dato del período (``extremos.tasa``, un estrato); λ = picos / E con IC de Garwood.
- **Banderas** (§3.6): ``dudoso_f3`` (la ventana contiene un registro dudoso que la F3 mantuvo) y
  ``revision_visual_f6`` (clasificación de la F6 de un pico de ``declustering_dudoso.csv`` que cae en la ventana).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..clima import clase_fisica as CF
from ..extremos import tasa
from . import conteo_previo as CP
from . import identificacion as ID

RAMAS = ("E0_sin_corregir", "E0_cuantiles", "E1_sin_corregir", "E1_cuantiles")
SIN_CORREGIR = {"E0_sin_corregir": "E0_sin_corregir", "E0_cuantiles": "E0_sin_corregir",
                "E1_sin_corregir": "E1_sin_corregir", "E1_cuantiles": "E1_sin_corregir"}  # fmt: skip
FIN_MODELO = CP.FIN_MODELO
ETIQUETAS = {"tp": "VTPK", "vmdr": "VMDR", "vped": "VPED"}
REVISION_REQUERIDA = "revisión requerida"
TOL_EMPATE = 1e-4  # precisión n.º 24 del contrato (propuesta; posterior a los datos, por artefacto)
NOTA_GARWOOD_E0 = (
    "Supone conteos de Poisson. En E0 ese supuesto no vale: la tasa sobre u_id salta en 1993 (6.9 frente a 14.6 "
    "eventos por año, conteo de §5.0) y la prueba de Poisson de §3.5 (punto 1) lo rechaza o se espera que lo rechace. "
    "El intervalo es optimista: más estrecho de lo que debería."
)


def es_variante(rama: str) -> bool:
    return rama.endswith("_cuantiles")


def inicio_rama(rama: str, inicio_serie: pd.Timestamp, inicio_e1: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(inicio_e1) if rama.startswith("E1") else pd.Timestamp(inicio_serie)


def eventos_de_rama(eventos: pd.DataFrame, inicio, fin=FIN_MODELO) -> pd.DataFrame:
    """Eventos con el pico de E0 sin corregir en [inicio, fin] (§3.1)."""
    return ID.en_periodo(eventos, inicio, fin)


def _alinear(hs_rama: pd.Series, indice: pd.DatetimeIndex) -> np.ndarray:
    """La serie de la rama en la rejilla de E0 (NaN fuera de la rama: así la ventana queda recortada)."""
    return hs_rama.reindex(indice).to_numpy(dtype=float)


def particion_dominante(fila_serie: pd.DataFrame) -> np.ndarray:
    """La partición de mayor energía (WW, SW1 o SW2) en cada fila; «sin particiones» si ninguna tiene energía."""
    if not all(c in fila_serie.columns for c in CF.PARTICIONES.values()):
        return np.full(len(fila_serie), CF.SIN_CLASE, dtype=object)
    aus = {q: fila_serie[f"marca_particion_ausente_{q}"] for q in CF.PARTICIONES
           if f"marca_particion_ausente_{q}" in fila_serie}  # fmt: skip
    fr = CF.fracciones(fila_serie, aus)
    r = fr[[f"r_{p}" for p in CF.PARTICIONES]].to_numpy(dtype=float)
    out = np.full(len(fila_serie), CF.SIN_CLASE, dtype=object)
    ok = np.isfinite(r).all(axis=1)
    nombres = np.array(list(CF.PARTICIONES), dtype=object)
    out[ok] = nombres[np.argmax(r[ok], axis=1)]
    return out


def picos_de_rama(
    eventos_rama: pd.DataFrame, hs_rama: pd.Series, serie_e0: pd.DataFrame, tol_empate: float | None = None
) -> pd.DataFrame:
    """Pico de la rama en la ventana de cada evento, con etiquetas de E0 en su instante y ``pico_movido``.

    ``serie_e0``: la serie del nodo en la rejilla completa (índice de tiempo), con ``VHM0`` y las etiquetas.

    ``tol_empate`` (precisión n.º 24, propuesta): si la Hs de la rama en el instante del pico de E0 queda a menos de
    ``tol_empate`` relativo del máximo de la ventana, el pico se toma en el instante del pico de E0
    (``empate_resuelto``). Corrige el desempate artificial de la corrección por cuantiles de la F5 entre estados con
    la misma Hs en E0. ``None``: sin la regla (lo publicado el 2026-11-23).
    """
    idx = pd.DatetimeIndex(serie_e0.index)
    v = _alinear(hs_rama, idx)
    inicio = pd.Timestamp(hs_rama.index[0])
    hs_e0 = serie_e0["VHM0"].to_numpy(dtype=float)
    filas = []
    for ev in eventos_rama.itertuples(index=False):
        s, e = int(ev.i_inicio), int(ev.i_fin)
        seg = v[s : e + 1]
        if not np.isfinite(seg).any():
            raise ValueError(f"El evento {ev.id_evento} no tiene ningún dato de la rama en su ventana")
        j = s + int(np.argmax(np.where(np.isfinite(seg), seg, -np.inf)))
        i0 = int(ev.i_pico)
        empate = tol_empate is not None and j != i0 and bool(np.isfinite(v[i0])) and v[i0] >= v[j] * (1 - tol_empate)
        if empate:
            j = i0
        filas.append({
            "id_evento": int(ev.id_evento), "id_racha": int(ev.id_racha),
            "t_inicio": ev.t_inicio, "t_fin": ev.t_fin, "t_pico": idx[j], "t_pico_e0": ev.t_pico,
            "i_pico": j, "n_excedencias": int(ev.n_excedencias), "hs": float(v[j]), "hs_e0_en_pico": float(hs_e0[j]),
            "toca_hueco": bool(ev.toca_hueco), "dividido_por_bajada": bool(ev.dividido_por_bajada),
            "pico_movido": bool(j != i0), "empate_resuelto": bool(empate), "recortado": bool(idx[s] < inicio),
        })  # fmt: skip
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    sub = serie_e0.iloc[df["i_pico"].to_numpy()]
    for nombre, col in ETIQUETAS.items():
        df[nombre] = sub[col].to_numpy(dtype=float) if col in sub.columns else np.nan
    df["particion_dominante"] = particion_dominante(sub)
    df["estrato"] = "unico"
    return df


def u_id_en_variante(hs_e0: pd.Series, hs_var: pd.Series, u_id: float) -> dict:
    """u_id llevado a la variante por su propio cuantil, estados discordantes y u_id^max (§3.2 p. 5; E-29).

    Las dos series se comparan en los instantes en que ambas tienen dato.
    """
    a, b = hs_e0.align(hs_var, join="inner")
    ok = np.isfinite(a.to_numpy(float)) & np.isfinite(b.to_numpy(float))
    x, y = a.to_numpy(float)[ok], b.to_numpy(float)[ok]
    if x.size == 0:
        raise ValueError("Sin estados comunes entre la rama y su variante")
    q0 = float(np.mean(x <= u_id))
    u_var = float(np.quantile(y, q0, method="inverted_cdf")) if q0 > 0 else float(np.min(y))
    sobre_e0, sobre_var = x > u_id, y > u_var
    bajo = x <= u_id
    return {
        "u_id_m": float(u_id),
        "cuantil_de_u_id": q0,
        "u_id_variante_m": u_var,
        "u_id_max_m": float(np.max(y[bajo])) if bajo.any() else u_var,
        "estados_comparados": int(x.size),
        "estados_discordantes": int(np.sum(sobre_e0 != sobre_var)),
        "sobre_u_id_solo_en_e0": int(np.sum(sobre_e0 & ~sobre_var)),
        "sobre_u_id_solo_en_variante": int(np.sum(~sobre_e0 & sobre_var)),
    }


def borde_inferior(picos: pd.DataFrame, info: Mapping) -> tuple[pd.DataFrame, dict]:
    """E-29: con estados discordantes, solo los picos de la variante > u_id^max. Devuelve los picos y el resumen."""
    activo = int(info["estados_discordantes"]) > 0
    quedan = picos[picos["hs"] > float(info["u_id_max_m"])] if activo else picos
    return quedan, {
        "activo": activo,
        "u_id_max_m": float(info["u_id_max_m"]),
        "picos_antes": int(len(picos)),
        "picos_despues": int(len(quedan)),
        "N0": int(len(quedan)),
    }


def proteccion_cola(picos: pd.DataFrame, p95_f3: float) -> dict:
    """G8: picos de la rama en instantes con Hs de E0 sin corregir < P95 de la F3. Alguno → «revisión requerida»."""
    malos = picos[~(picos["hs_e0_en_pico"] >= float(p95_f3))]
    return {
        "p95_f3_m": float(p95_f3),
        "picos_bajo_p95": int(len(malos)),
        "lista": [
            {"id_evento": int(r.id_evento), "t_pico": str(r.t_pico), "hs_e0_en_pico": float(r.hs_e0_en_pico)}
            for r in malos.itertuples(index=False)
        ],  # fmt: skip
        "estado": REVISION_REQUERIDA if len(malos) else "cumple",
    }


def exposicion(hs_rama: pd.Series, inicio, fin=FIN_MODELO, n_picos: int | None = None, rama: str = "") -> dict:
    """Años con dato del período (un estrato) y, si se da ``n_picos``, λ con su IC de Garwood del 95 %.

    En las ramas de E0 el intervalo lleva ``NOTA_GARWOOD_E0`` (pedido del usuario, 2026-11-23); en E1 va sin nota.
    """
    anios = CP.anios_exposicion(hs_rama, inicio, fin)
    out = {"inicio": str(pd.Timestamp(inicio)), "fin": str(pd.Timestamp(fin)), "anios": anios}
    if n_picos is not None:
        lo, hi = tasa.ic_garwood(int(n_picos), anios)
        out |= {"picos": int(n_picos), "lambda_por_anio": n_picos / anios, "ic_garwood_95": [lo, hi],
                "nota_ic": NOTA_GARWOOD_E0 if rama.startswith("E0") else None}  # fmt: skip
    return out


def marcar_dudosos_f3(picos: pd.DataFrame, candidatos: pd.DataFrame) -> np.ndarray:
    """True si la ventana contiene un registro que la F3 mantuvo con motivo dudoso (§1)."""
    c = candidatos[(candidatos["decision"] == "mantener") & candidatos["motivo"].astype(str).str.contains("dudoso")]
    if picos.empty or c.empty:
        return np.zeros(len(picos), bool)
    d = np.sort(pd.DatetimeIndex(pd.to_datetime(c["tiempo"])).to_numpy())
    ini, fin = pd.DatetimeIndex(picos["t_inicio"]).to_numpy(), pd.DatetimeIndex(picos["t_fin"]).to_numpy()
    return np.searchsorted(d, fin, side="right") > np.searchsorted(d, ini, side="left")


def marcar_revision_f6(picos: pd.DataFrame, revision: pd.DataFrame) -> np.ndarray:
    """La clasificación visual de la F6 de los picos de ``declustering_dudoso.csv`` que caen en la ventana."""
    out = np.full(len(picos), "", dtype=object)
    if picos.empty or revision.empty:
        return out
    ini, fin = pd.DatetimeIndex(picos["t_inicio"]), pd.DatetimeIndex(picos["t_fin"])
    for r in revision.itertuples(index=False):
        for col in ("t_pico", "t_pico_siguiente"):
            t = getattr(r, col, None)
            if t is None or pd.isna(t):
                continue
            t = pd.Timestamp(t)
            en = np.flatnonzero((ini <= t) & (t <= fin))
            etiqueta = f"{r.tipo}: {r.clasificacion_visual}"
            for i in en:
                previas = out[i].split("; ") if out[i] else []
                if etiqueta not in previas:
                    out[i] = "; ".join([*previas, etiqueta])
    return out
