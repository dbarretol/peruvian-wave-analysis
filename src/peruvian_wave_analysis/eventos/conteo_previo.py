"""Conteo previo de eventos, ``eventos-conteo-previo`` (contrato de la F7, §5.0; rotulado: se mira el número de
tormentas, no su tamaño).

Cuenta los eventos identificados con §3.2 (u_id = P95 de la F3, bajada intermedia) en E0 y en E1, con IET = 48 h y
con IET = 120 h, en el período del modelo (ADR-0031: picos hasta el 2025-12-31; E1 desde el inicio que publica la
F4). Son 4 números, con su exposición en años (``extremos.tasa``, un solo estrato) y su tasa por año.

**No mira** ningún valor de Hs de los picos, ningún ajuste, ξ ni RL: la salida tiene solo conteos, exposiciones y
tasas, además de u_id (un número ya publicado por la F3).

**Regla de revisión de u_id (P9), fijada en el contrato antes de contar:** si E0 con 48 h da más de 20 eventos por
año, u_id pasa al menor cuantil de {P97, P98, P99} de Hs sin corregir (con peso mensual, sobre toda la serie válida)
que dé ≤ 20 eventos por año con 48 h, y el conteo se repite. Se rotula como posterior al conteo y se avisa. Si
ninguno cumple, el gate falla y decide el usuario.

**N0 de la simulación (§5.1):** N_mín y N_máx de los 4 conteos finales. Si N_máx / N_mín < 1.5, se usan
{0.8·N_mín, 1.2·N_máx} (redondeados). Los años simulados son los de la exposición de la rama de cada extremo,
redondeados (46 en E0; 33 en E1).

Salida en ``processed/F07-eventos/simulacion-previa/conteo-previo/``: la simulación publica su propia salida al lado,
en ``simulacion-previa/``, sin reemplazar esta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..clima import insumos as CI
from ..comun import etapa, json_estricto, orquestador
from ..comun.errores import GateFallido
from ..datos import carga
from ..datos.run import ruta_candidatos, ruta_serie
from ..extremos import tasa
from ..extremos.estratos import Estratificacion
from ..homogeneidad.cuantiles import cuantiles_mensuales
from . import identificacion as ID

ETAPA = orquestador.Etapa("eventos-conteo-previo", Path(__file__).resolve(), Path("."))
NOMBRE = CI.NOMBRE
IETS_H = (48.0, 120.0)
FIN_MODELO = pd.Timestamp("2025-12-31 23:59:59")  # ADR-0031
TASA_MAX = 20.0  # eventos por año (P9)
CUANTILES_REVISION = (0.97, 0.98, 0.99)
RAZON_MIN_N0 = 1.5
FACTORES_N0 = (0.8, 1.2)


def ruta_salida(raiz: Path) -> Path:
    return Path(raiz) / "processed" / "F07-eventos" / "simulacion-previa" / "conteo-previo"


def rutas_insumos(raiz: Path, nombre: str = NOMBRE) -> dict[str, Path]:
    """Insumos, tras exigir que cada salida esté publicada y sin cambios (G1)."""
    ds, dc, d4 = ruta_serie(raiz, nombre), ruta_candidatos(raiz, nombre), CI.ruta_f04(raiz, nombre)
    for d in (ds, dc, d4):
        etapa.exigir_publicada(d)
    return {
        "serie_nodo": ds / "serie_nodo.parquet",
        "resumen_candidatos": dc / "resumen_candidatos.json",
        "resumen_f04": d4 / "resumen_f04.json",
    }


def anios_exposicion(hs: pd.Series, inicio, fin) -> float:
    """Años con dato del período [inicio, fin] (``tasa.anios_con_dato`` con un solo estrato)."""
    s = hs[(hs.index >= pd.Timestamp(inicio)) & (hs.index <= pd.Timestamp(fin))]
    bloques = tasa.anios_con_dato(s.index, np.isfinite(s.to_numpy(float)), ID.PASO_H, Estratificacion.anual())
    return float(bloques["fraccion"].sum())


def contar(hs: pd.Series, u_id: float, inicio_e1, fin=FIN_MODELO) -> list[dict]:
    """Los 4 conteos (E0 y E1 × 48 y 120 h): solo números de eventos, rachas, exposición y tasa."""
    periodos = {"E0": (hs.index[0], fin), "E1": (pd.Timestamp(inicio_e1), fin)}
    expo = {r: anios_exposicion(hs, a, b) for r, (a, b) in periodos.items()}
    out = []
    for iet in IETS_H:
        ev = ID.identificar(hs, u_id, iet)
        for rama, (a, b) in periodos.items():
            sub = ID.en_periodo(ev, a, b)
            n = int(len(sub))
            out.append(
                {
                    "rama": rama,
                    "iet_h": iet,
                    "eventos": n,
                    "rachas": int(sub["id_racha"].nunique()),
                    "eventos_de_rachas_divididas": int(sub["dividido_por_bajada"].sum()),
                    "anios_exposicion": expo[rama],
                    "eventos_por_anio": n / expo[rama],
                }
            )
    return out


def _tasa_e0_48(conteos: list[dict]) -> float:
    return next(c["eventos_por_anio"] for c in conteos if c["rama"] == "E0" and c["iet_h"] == 48.0)


def n0_simulacion(conteos: list[dict]) -> dict:
    """N0 ∈ {N_mín, N_máx} de §5.1, con los años simulados de la rama de cada extremo."""
    lo = min(conteos, key=lambda c: (c["eventos"], c["rama"] != "E1"))
    hi = max(conteos, key=lambda c: (c["eventos"], c["rama"] == "E0"))
    n_min, n_max = lo["eventos"], hi["eventos"]
    ampliado = n_min == 0 or n_max / n_min < RAZON_MIN_N0
    n0 = [round(FACTORES_N0[0] * n_min), round(FACTORES_N0[1] * n_max)] if ampliado else [n_min, n_max]
    return {
        "n_min": n_min,
        "n_max": n_max,
        "origen_min": {"rama": lo["rama"], "iet_h": lo["iet_h"]},
        "origen_max": {"rama": hi["rama"], "iet_h": hi["iet_h"]},
        "razon": n_max / n_min if n_min else float("inf"),
        "ampliado_por_razon_menor_a_1_5": bool(ampliado),
        "n0": [int(x) for x in n0],
        "anios_simulados": [int(round(lo["anios_exposicion"])), int(round(hi["anios_exposicion"]))],
    }


def conteo_previo(hs: pd.Series, u_id_f3: float, inicio_e1) -> dict:
    """Conteo de §5.0 con la regla de revisión de u_id (P9). No devuelve magnitudes de los picos."""
    conteos = contar(hs, u_id_f3, inicio_e1)
    resumen: dict = {
        "u_id_f3_m": u_id_f3,
        "conteos_con_u_id_f3": conteos,
        "tasa_e0_48h": _tasa_e0_48(conteos),
        "u_id_revisado": False,
        "revision": None,
    }
    u_final, finales = u_id_f3, conteos
    if resumen["tasa_e0_48h"] > TASA_MAX:
        intentos = []
        u_q = cuantiles_mensuales(hs, list(CUANTILES_REVISION))
        elegido = None
        for q, u in zip(CUANTILES_REVISION, u_q, strict=True):
            c = contar(hs, float(u), inicio_e1)
            intentos.append({"cuantil": q, "u_m": float(u), "tasa_e0_48h": _tasa_e0_48(c)})
            if _tasa_e0_48(c) <= TASA_MAX:
                elegido = (q, float(u), c)
                break
        resumen["revision"] = {
            "rotulo": "Posterior al conteo previo (§5.0, P9): u_id se revisa por la regla fijada antes de contar.",
            "intentos": intentos,
            "cuantil_elegido": None if elegido is None else elegido[0],
        }
        if elegido is not None:
            resumen["u_id_revisado"] = True
            u_final, finales = elegido[1], elegido[2]
        else:
            u_final, finales = float("nan"), []
    resumen["u_id_m"] = u_final
    resumen["conteos"] = finales
    resumen["n0_simulacion"] = n0_simulacion(finales) if finales else None
    return resumen


def ejecutar(raiz_datos=None, nombre: str = NOMBRE) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    ins = rutas_insumos(raiz, nombre)
    etapa.exigir_insumos(ins)
    salida = ruta_salida(raiz)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        serie = carga.cargar_serie(raiz, nombre)
        if "tiempo" in serie.columns:
            serie = serie.set_index("tiempo")
        hs = serie["VHM0"].astype(float)
        u_id = float(json_estricto.leer(ins["resumen_candidatos"])["p95_hs_m"])
        ini_e1 = CI.inicio_e1(json_estricto.leer(ins["resumen_f04"]))
        res = conteo_previo(hs, u_id, ini_e1)
        res |= {
            "valores_de_picos_leidos": False,
            "criterio": (
                f"Eventos de §3.2 (u_id, bajada intermedia de {ID.BAJADA_M} m) en E0 (desde {hs.index[0].date()}) y E1 "
                f"(desde {ini_e1.date()}), picos hasta {FIN_MODELO.date()} (ADR-0031), IET {list(IETS_H)} h. Revisión "
                f"de u_id si E0 con 48 h da más de {TASA_MAX:g} eventos por año (P9)."
            ),
            "periodo_fin": str(FIN_MODELO.date()),
            "inicio_e1": str(ini_e1.date()),
        }
        json_estricto.escribir(st / "conteo_previo.json", res)
        for c in res["conteos"]:
            print(f"  {c['rama']} IET {c['iet_h']:g} h: {c['eventos']} eventos, {c['eventos_por_anio']:.2f} por año")
        if res["u_id_revisado"]:
            print(
                f"AVISO: E0 con 48 h dio {res['tasa_e0_48h']:.2f} eventos por año (> {TASA_MAX:g}); u_id pasa al "
                f"cuantil {res['revision']['cuantil_elegido']} ({res['u_id_m']:.2f} m). Rotulado; avisar al usuario.",
                flush=True,
            )
        motivos = []
        if not res["conteos"]:
            motivos.append(
                f"Ningún cuantil de {list(CUANTILES_REVISION)} da ≤ {TASA_MAX:g} eventos por año en E0 con 48 h: "
                "decide el usuario (§5.0)"
            )
        criterios = {"tasa_max_por_anio": TASA_MAX, "iets_h": list(IETS_H), "u_id_revisado": res["u_id_revisado"]}
        return etapa.publicar(st, salida, etapa.Gate(not motivos, criterios, motivos), raices, ins, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eventos-conteo-previo", description="Conteo previo de eventos de la F7 (§5.0)")
    ap.add_argument("--datos")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = ejecutar(a.datos)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
