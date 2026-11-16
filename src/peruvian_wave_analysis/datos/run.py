"""Etapas publicadas de la F3: ``datos-candidatos`` y ``datos-serie`` (contrato de la F3, §2).

1. ``datos-candidatos``: ingesta, reglas, calibración de k, detector, dipolo, máximos y figuras. No exige
   resoluciones; su gate comprueba la ingesta y que no se superan los topes de §3.4 ni el de marcas con resolución (v7).
2. ``datos-serie``: lee la salida de la primera y el YAML de resoluciones, aplica los descartes y publica la serie.
   Su gate exige todas las resoluciones (G4, G5). Si falla, no toca la salida de la primera.

Uso::

    uv run datos-candidatos --datos C:/WORKSPACES/TESIS-M5-WAVE/WAVE-DATA
    uv run datos-serie --datos C:/WORKSPACES/TESIS-M5-WAVE/WAVE-DATA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import estilo, rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..comun.huella import DIR_PROYECTO
from ..sitio import serie as serie_f1
from . import calibracion as K
from . import config as C
from . import detector as D
from . import ingesta, periodos, reglas
from . import resoluciones as R

NOMBRE = "marcona-200m"
RESOLUCIONES = DIR_PROYECTO / "config" / "f03_resoluciones.yaml"
DIR_ETAPA = Path(__file__).resolve().parent
ETAPA_CANDIDATOS = orquestador.Etapa("datos-candidatos", DIR_ETAPA, Path("."))
ETAPA_SERIE = orquestador.Etapa("datos-serie", DIR_ETAPA, Path("."), codigo_extra=(RESOLUCIONES,))
CONTEXTO = pd.Timedelta(hours=48)
REGLAS_CON_RESOLUCION = ("hs_total_cero", "peralte", "particion_mayor", "tm10_menor_tm02")


def ruta_candidatos(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F03-datos" / "candidatos" / nombre


def ruta_serie(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F03-datos" / "serie" / nombre


# --- datos-candidatos ---------------------------------------------------------------------------


def _insumos_serie(raiz: Path, nombre: str) -> tuple[dict[str, Path], dict[str, str], dict]:
    leido = serie_f1.leer_gate(raiz, nombre)
    if leido is None:
        raise ingesta.ErrorIngesta("No hay gate publicado de la serie de la F1: correr `serie-waverys-gate`")
    ruta_gate, gate = leido
    raw = rutas.raw(raiz)
    insumos = {f"serie:{a}": raw / "waverys" / "serie" / a for a in sorted(gate["archivos_sha256"])}
    esperados = {f"serie:{a}": h for a, h in gate["archivos_sha256"].items()}
    insumos["serie_gate"] = ruta_gate
    return insumos, esperados, gate


def _vecinas(serie: ingesta.Serie) -> tuple[np.ndarray, list[str]]:
    v = ingesta.vecinas_con_dato(serie, anillo=1)
    nombres = [c.nombre for c in v]
    return serie.hs_celdas[nombres].to_numpy(), nombres


def _vecinas_sensibilidad(serie: ingesta.Serie) -> dict[str, np.ndarray]:
    """Variantes del criterio de vecinas que se publican como sensibilidad y no deciden nada (§3.1, F03-K-03):
    sin la vecina con tierra y con los dos anillos."""
    a1 = [c.nombre for c in ingesta.vecinas_con_dato(serie, anillo=1)]
    a2 = [c.nombre for c in ingesta.vecinas_con_dato(serie, anillo=2)]
    tierra = ingesta.Celda(*C.VECINA_TIERRA, 1).nombre
    return {
        "sin_tierra": serie.hs_celdas[[n for n in a1 if n != tierra]].to_numpy(),
        "dos_anillos": serie.hs_celdas[a1 + a2].to_numpy(),
    }


def _particion_del_salto(df: pd.DataFrame, e: D.Excursion) -> str:
    """Partición con mayor |ΔHs_i| entre el registro previo y el pico (F03-S-20)."""
    saltos = {p: abs(df[h].iloc[e.pico] - df[h].iloc[e.inicio - 1]) for p, (h, _, _) in C.PARTICIONES.items()}
    saltos = {p: v for p, v in saltos.items() if np.isfinite(v)}
    return max(saltos, key=saltos.get) if saltos else ""


def _resumen_numerico(x: pd.Series) -> dict:
    x = x.dropna()
    q = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    return {"n": int(x.size), "cuantiles": {str(c): float(x.quantile(c)) for c in q} if x.size else None}


def _figura_maximo(
    df: pd.DataFrame,
    vec: pd.DataFrame,
    t: pd.Timestamp,
    ruta: Path,
    titulo: str | None = None,
    tramo: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    """Contexto de ±48 h. ``tramo``: (registro previo, registro de vuelta) de la excursión, sombreado en todos los
    paneles para que se vea el paso del salto, no solo el pico."""
    ventana = df.loc[t - CONTEXTO : t + CONTEXTO]
    vv = vec.loc[t - CONTEXTO : t + CONTEXTO]
    fig, ax = estilo.nueva_figura(3, 1, figsize=(8.0, 8.0), sharex=True, layout="constrained")
    for c in vv.columns:
        ax[0].plot(vv.index, vv[c], color="0.75", lw=0.8)
    ax[0].plot(ventana.index, ventana["VHM0"], color="k", lw=1.6, label="Hs nodo")
    for h, et in (("VHM0_WW", "Hs mar de viento"), ("VHM0_SW1", "Hs swell 1"), ("VHM0_SW2", "Hs swell 2")):
        ax[0].plot(ventana.index, ventana[h], lw=1.0, label=et)
    ax[0].axvline(t, color="r", lw=0.8)
    if tramo is not None:
        for a in ax:
            a.axvspan(tramo[0], tramo[1], color="r", alpha=0.12, lw=0)
    ax[0].set_ylabel("Hs (m)")
    ax[0].legend(fontsize=7, ncol=2)
    for v, et in (("VTPK", "Tp"), ("VTM02", "Tm02"), ("VTM10", "Tm−1,0")):
        ax[1].plot(ventana.index, ventana[v], lw=1.0, label=et)
    ax[1].set_ylabel("Período (s)")
    ax[1].legend(fontsize=7)
    ax[2].plot(ventana.index, ventana["VMDR"], ".", ms=3, color="k")
    ax[2].set_ylabel("Dirección de origen (°)")
    ax[2].set_ylim(0, 360)
    ax[2].set_xlabel("Tiempo (UTC)")
    ax[0].set_title(titulo or f"Máximo del {t:%Y-%m-%d %H:%M} UTC: Hs = {df.at[t, 'VHM0']:.2f} m (gris: vecinas)")
    estilo.guardar_figura(fig, ruta)


def _figura_histograma(x: pd.Series, valores: list[float], ruta: Path, unidad: str) -> None:
    fig, ax = estilo.nueva_figura(figsize=(7.0, 3.5), layout="constrained")
    ax.hist(x.dropna(), bins=200, color="0.4")
    for v in valores:
        ax.axvline(v, color="r", lw=0.8)
    ax.set_xlabel(f"{x.name} ({unidad})")
    ax.set_ylabel("Registros")
    ax.set_title(f"{x.name}: acumulaciones en rojo")
    estilo.guardar_figura(fig, ruta)


def candidatos(raiz_datos=None, nombre: str = NOMBRE, anios_calibracion: int = K.ANIOS) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    insumos, esperados, gate = _insumos_serie(raiz, nombre)
    etapa.exigir_insumos(insumos, esperados)
    salida = ruta_candidatos(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_CANDIDATOS)
    with etapa.staging_protegido(salida, raices) as st:
        s_nodo = ingesta.leer_serie([insumos[k] for k in insumos if k.startswith("serie:")])
        ingesta.comprobar_contra_gate(s_nodo, gate)
        df0 = reglas.normalizar_direcciones(s_nodo.nodo)
        aus = reglas.particiones_ausentes(df0)
        df = aus.serie
        motivos: list[str] = []
        hs = df["VHM0"].to_numpy()
        tm02 = df["VTM02"].to_numpy()
        s = D.escala_s(hs)
        cal = None
        try:
            cuantos = K.exigir_escala(s)
            cal = K.calibrar(cuanto_en_s=C.CUANTO / s, anios=anios_calibracion)
        except K.CalibracionImposible as exc:
            motivos.append(f"revisión requerida: {exc}")
            cuantos = s / C.CUANTO
        p95 = D.percentil_cola(df0["VHM0"])
        vec, nombres_vec = _vecinas(s_nodo)
        vec_sens = _vecinas_sensibilidad(s_nodo)
        idx = df.index
        filas = []  # candidatos: tiempo, regla, decision, motivo, hs, inicio, fin, salto, vecinas
        # máximos, sobre la serie antes de todo descarte (§3.10)
        maximos = D.maximos_independientes(df0["VHM0"])

        def agregar(i: int, regla: str, decision: str, motivo: str, exc: D.Excursion | None = None) -> None:
            e = exc if exc is not None else D.Excursion(i, i, i, float(hs[i] - hs[i - 1]) if i > 0 else 0.0)
            razon, nv = D.razon_vecinas(hs, vec, e) if (exc is not None or regla == "dipolo") else (np.nan, 0)
            filas.append(
                {
                    "tiempo": idx[i],
                    "regla": regla,
                    "decision": decision,
                    "motivo": motivo,
                    "hs": round(float(hs[i]), 2),
                    "inicio": idx[e.inicio],
                    "fin": idx[e.fin],
                    "salto_m": round(float(e.salto), 2) if exc is not None else np.nan,
                    "razon_vecinas": round(float(razon), 3) if np.isfinite(razon) else np.nan,
                    "n_vecinas": int(nv),
                }
            )

        def por_clase(e: D.Excursion, regla: str) -> None:
            """Candidato del detector en la cola: regla por clase (v8, §16). La clase «vuelve y no la comparten»
            queda manual con figura: nada se descarta automáticamente en la cola (v9, §17, F03-S-21)."""
            r = D.resolver_por_clase(e, hs, tm02, vec)
            if r.decision == D.DESCARTAR:
                agregar(e.pico, regla, D.MANUAL, f"clase:{r.motivo}:cola", e)
            else:
                agregar(e.pico, regla, r.decision, f"clase:{r.motivo}", e)

        if cal is not None:
            picos_desc = set()
            for e in D.detectar(hs, cal.k_desc, s):
                r = D.resolver(e, hs, tm02, vec, p95)
                if r.motivo == "cola_p95":
                    por_clase(e, "detector")
                else:
                    agregar(e.pico, "detector", r.decision, r.motivo, e)
                picos_desc.add(e.pico)
            n_rev = 0
            for e in D.detectar(hs, cal.k_rev, s):
                if hs[e.pico] >= p95:
                    n_rev += 1
                    if e.pico not in picos_desc:
                        por_clase(e, "detector_revision")
            for i in D.dipolo(hs, vec, cal.k_desc, s).tolist():
                if i not in picos_desc:
                    agregar(i, "dipolo", D.MANUAL, "dipolo")
        else:
            n_rev = 0
        lim = reglas.fuera_de_limites(df)
        for col in lim.columns:
            for i in np.flatnonzero(lim[col].to_numpy()).tolist():
                cola = df0["VHM0"].iloc[i] >= p95
                agregar(i, col, D.MANUAL if cola else D.DESCARTAR, "cola_p95" if cola else "fuera_de_limite")
        marcas = pd.concat([aus.marcas, reglas.coherencia(df), lim], axis=1)
        marcas["peralte"] = reglas.peralte_excedido(df)
        for p in C.PARTICIONES:
            for i in np.flatnonzero(marcas[f"incoherencia_ausente_{p}"].to_numpy()).tolist():
                agregar(i, f"incoherencia_ausente_{p}", D.MANUAL, "periodo_cero_con_altura")
        for regla in REGLAS_CON_RESOLUCION:
            for i in np.flatnonzero(marcas[regla].fillna(False).to_numpy()).tolist():
                agregar(i, regla, D.MANUAL, regla)
        # tope por marca que exige resolución (v7, §15)
        marcas_resolucion = [f"incoherencia_ausente_{p}" for p in C.PARTICIONES] + list(REGLAS_CON_RESOLUCION)
        conteo_marcas = {r: int(marcas[r].fillna(False).sum()) for r in marcas_resolucion}
        motivos += [f"revisión requerida: {m}" for m in reglas.marcas_sobre_tope(conteo_marcas)]
        columnas = [
            "tiempo",
            "regla",
            "decision",
            "motivo",
            "hs",
            "inicio",
            "fin",
            "salto_m",
            "razon_vecinas",
            "n_vecinas",
        ]
        cand = pd.DataFrame(filas, columns=columnas)
        cand = cand.sort_values(["tiempo", "regla"]).reset_index(drop=True)
        auto = pd.DatetimeIndex(cand.loc[cand["decision"] == D.DESCARTAR, "tiempo"].unique())
        top = D.topes(auto, n_rev)
        motivos += [f"revisión requerida: {m}" for m in top.revision_requerida]
        # período saturado, discreción y acumulaciones (marcas sin descarte)
        discreto = reglas.es_discreto(df["VTPK"])
        marcas["tp_saturado"] = reglas.saturados(df["VTPK"], discreto)
        for _, t, _ in C.PARTICIONES.values():
            marcas[f"{t}_saturado"] = reglas.saturados(df[t], discreto=False)
        acum = {}
        for v in C.ALTURAS + C.PERIODOS + C.DIRECCIONES:
            x = df[v]
            if v in C.ALTURAS[1:]:
                x = x.where(x > 0)  # las particiones ausentes no son acumulaciones
            vals = reglas.acumulaciones_discretas(x) if (v == "VTPK" and discreto) else reglas.acumulaciones(x)
            if vals:
                acum[v] = vals
                _figura_histograma(x.rename(v), vals, st / "figuras" / f"histograma_{v}.png", C.UNIDADES[v])
        # todas las excursiones de k_rev, en toda la serie, para la F4 (v9, §17, F03-S-20), con las sensibilidades
        # del criterio de vecinas de §3.1 (F03-K-03)
        p99 = float(np.nanpercentile(df0["VHM0"], 99))
        en_cand = set(cand.loc[cand["regla"].isin(["detector", "detector_revision"]), "tiempo"])
        exc_rev = []
        for e in D.detectar(hs, cal.k_rev, s) if cal is not None else []:
            fila = {
                "tiempo": idx[e.pico],
                "inicio": idx[e.inicio],
                "fin": idx[e.fin],
                "anio": int(idx[e.pico].year),
                "signo": int(np.sign(e.salto)),
                "salto_m": round(float(e.salto), 2),
                "hs": round(float(hs[e.pico]), 2),
                "cola_p95": bool(hs[e.pico] >= p95),
                "candidato": idx[e.pico] in en_cand,
                "particion": _particion_del_salto(df, e),
                "d_tm02_s": round(float(tm02[e.pico] - tm02[e.inicio - 1]), 2),
                "d_tp_s": round(float(df["VTPK"].iloc[e.pico] - df["VTPK"].iloc[e.inicio - 1]), 2),
            }
            for nombre_v, arr in (
                ("", vec),
                ("_sin_tierra", vec_sens["sin_tierra"]),
                ("_dos_anillos", vec_sens["dos_anillos"]),
            ):
                r_, n_v = D.razon_vecinas(hs, arr, e)
                fila[f"razon_vecinas{nombre_v}"] = round(float(r_), 3) if np.isfinite(r_) else np.nan
                fila[f"n_vecinas{nombre_v}"] = int(n_v)
            exc_rev.append(fila)
        exc_rev = pd.DataFrame(exc_rev)
        # una columna booleana por regla (§3.8, F03-K-01)
        for regla in ("detector", "detector_revision", "dipolo"):
            marcas[regla] = idx.isin(cand.loc[cand["regla"] == regla, "tiempo"])
        for v, vals in acum.items():
            marcas[f"acumulacion_{v}"] = df[v].round(2).isin(np.round(vals, 2))
        vec_df = pd.DataFrame(vec, index=idx, columns=nombres_vec)
        for n_, t in enumerate(maximos.index, start=1):
            _figura_maximo(df, vec_df, t, st / "figuras" / f"maximo_{n_:02d}.png")
        # una figura por candidato del detector o del dipolo (v8, §16: salida añadida)
        paso = pd.Timedelta(hours=3)
        figs = []
        for n_, f in enumerate(cand.itertuples(), start=1):
            if f.regla not in ("detector", "detector_revision", "dipolo"):
                figs.append("")
                continue
            nombre_fig = f"candidato_{n_:02d}.png"
            titulo = f"{f.regla} del {f.tiempo:%Y-%m-%d %H:%M} UTC: Hs = {f.hs:.2f} m · {f.decision} ({f.motivo})"
            tramo = (f.inicio - paso, f.fin + paso)
            _figura_maximo(df, vec_df, f.tiempo, st / "figuras" / nombre_fig, titulo, tramo)
            figs.append(nombre_fig)
        cand["figura"] = figs
        # salidas
        df.to_parquet(st / "serie_decodificada.parquet")
        s_nodo.hs_celdas.to_parquet(st / "hs_celdas.parquet")
        marcas.to_parquet(st / "marcas.parquet")
        cand.to_csv(st / "candidatos.csv", index=False, date_format="%Y-%m-%d %H:%M", lineterminator="\n")
        exc_rev.to_csv(st / "excursiones_k_rev.csv", index=False, date_format="%Y-%m-%d %H:%M", lineterminator="\n")
        pd.DataFrame({"tiempo": maximos.index, "hs": maximos.to_numpy().round(2)}).to_csv(
            st / "maximos.csv", index=False, date_format="%Y-%m-%d %H:%M", lineterminator="\n"
        )
        resumen = {
            "s_m": s,
            "s_cuantos": cuantos,
            "p95_hs_m": p95,
            "k_desc": cal.k_desc if cal else None,
            "k_rev": cal.k_rev if cal else None,
            "k_por_generador": {f"phi={p},nu={n}": k for (p, n), k in cal.k_por_generador.items()} if cal else None,
            "anios_calibracion": anios_calibracion,
            "conteos_fondo": {f"phi={p},nu={n}": c for (p, n), c in cal.conteos.items()} if cal else None,
            "carga_tormentas": {f"phi={p},nu={n}": c for (p, n), c in cal.carga_tormentas.items()} if cal else None,
            "rejilla_k": K.REJILLA_K.tolist(),
            "potencia_m": {"k_desc": cal.k_desc * s, "k_rev": cal.k_rev * s} if cal else None,
            # §3.4: por generador, sobre la serie sintética (no la real); F03-K-03
            "falsas_alarmas_sinteticas": {
                "k_desc": K.tasa_falsas_alarmas(cal, cal.k_desc),
                "k_rev": K.tasa_falsas_alarmas(cal, cal.k_rev),
            }
            if cal
            else None,
            "potencia_picos_un_paso": {
                "amplitudes_en_k_s": list(K.AMPLITUDES_POTENCIA),
                "anios": K.ANIOS_POTENCIA,
                "k_desc": K.potencia(C.CUANTO / s, cal.k_desc),
                "k_rev": K.potencia(C.CUANTO / s, cal.k_rev),
            }
            if cal
            else None,
            "excursiones_k_rev": {
                "n": int(len(exc_rev)),
                "en_la_cola": int(exc_rev["cola_p95"].sum()) if len(exc_rev) else 0,
                "comparten_vecinas": {
                    v: int((exc_rev[f"razon_vecinas{v}"] >= D.FRACCION_VECINAS).sum()) if len(exc_rev) else 0
                    for v in ("", "_sin_tierra", "_dos_anillos")
                },
                "razon_vecinas_minima": float(exc_rev["razon_vecinas"].min()) if len(exc_rev) else None,
                "por_anio": {str(a): int(n) for a, n in exc_rev.groupby("anio").size().items()} if len(exc_rev) else {},
            },
            "discrepancia_energia": _resumen_numerico(marcas["discrepancia_energia"]),
            "saturados_en_extremos": {
                c: {
                    "n": int(marcas[c].sum()),
                    "con_hs_ge_p99": int((marcas[c] & (df0["VHM0"] >= p99)).sum()),
                    "entre_los_maximos": int(marcas[c].reindex(maximos.index).sum()),
                }
                for c in marcas.columns
                if c.endswith("_saturado")
            },
            "p99_hs_m": p99,
            "ausencia": {
                p: {
                    "ausentes": r.ausentes,
                    "incoherencias": r.incoherencias,
                    "bajo_cuanto": r.bajo_cuanto,
                    "direcciones_distintas_ausentes": r.direcciones_distintas_ausentes,
                    "fraccion_ausentes_180": r.fraccion_ausentes_180,
                }
                for p, r in aus.resumen.items()
            },
            "periodos_bajo_resolucion": reglas.periodos_bajo_resolucion(df0),
            "marcas_con_resolucion": {"conteos": conteo_marcas, "tope": C.TOPE_MARCA},
            "dudosos_por_clase": [str(t) for t in cand.loc[cand["motivo"] == f"clase:{D.DUDOSO}", "tiempo"]],
            "fraccion_particion_ausente": {
                p: float(aus.marcas[f"particion_ausente_{p}"].mean()) for p in C.PARTICIONES
            },
            "vtpk_discreto": discreto,
            "vtpk_razones_consecutivas": reglas.razones_consecutivas(df["VTPK"]).round(4).tolist(),
            "acumulaciones": acum,
            "vecinas_primer_anillo": nombres_vec,
            "candidatos_por_regla_y_decision": cand.groupby(["regla", "decision"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records"),
            "topes": {
                "descartes_automaticos": top.total,
                "max_por_anio": top.max_anio,
                "revision_cola_k_rev": top.revision_cola,
            },
            "semilla_calibracion": semillas.semilla("f03-calibracion"),
        }
        json_estricto.escribir(st / "resumen_candidatos.json", resumen)
        gate_c = etapa.Gate(not motivos, {"topes": [D.TOPE_TOTAL, D.TOPE_ANIO, D.TOPE_REVISION]}, motivos)
        return etapa.publicar(st, salida, gate_c, raices, insumos, raiz, semillas.semilla("f03-calibracion"))


# --- datos-serie --------------------------------------------------------------------------------


def _profundidades(raiz: Path, nombre: str) -> tuple[float, float]:
    ficha = json.loads(
        (raiz / "processed" / "F01-sitio-y-fuentes" / "mapa-sitio-final" / nombre / "ficha_nodo.json").read_text(
            encoding="utf-8"
        )
    )
    h_sitio = float(ficha["nodo_elegido"]["prof_nodo_m"])
    est = sorted((rutas.raw(raiz) / "waverys" / "estatico").glob("deptho*.nc"))
    with xr.open_dataset(est[0]) as ds:
        lat = "latitude" if "latitude" in ds.coords else "lat"
        lon = "longitude" if "longitude" in ds.coords else "lon"
        i = ingesta.indice_exacto(ds[lat].values, C.NODO[0])
        j = ingesta.indice_exacto(ds[lon].values, C.NODO[1])
        h_modelo = float(ds["deptho"].isel({lat: i, lon: j}).values)
    return h_sitio, h_modelo


def _marcas_por_regla(
    booleanas: pd.DataFrame, df: pd.DataFrame, marcas: pd.DataFrame, cand: pd.DataFrame, idx_res: dict
) -> pd.DataFrame:
    """`marcas_por_regla.csv` (§2, F03-K-03): cada registro marcado, con regla, valor de Hs, magnitud del criterio y
    resolución. Magnitud: el salto en m para el detector y el dipolo; la discrepancia relativa para la energía."""
    por_cand = {(f.tiempo, f.regla): f for f in cand.itertuples()}
    filas = []
    for regla in booleanas.columns:
        for t in booleanas.index[booleanas[regla].to_numpy()]:
            c = por_cand.get((t, regla))
            r = idx_res.get((t, regla))
            if regla == "energia":
                magnitud = float(marcas.at[t, "discrepancia_energia"])
            elif c is not None:
                magnitud = float(c.salto_m)
            else:
                magnitud = np.nan
            if r is not None:
                resolucion = f"{r.decision}: {r.motivo}"
            elif c is not None:
                resolucion = f"{c.decision}: {c.motivo}"
            else:
                resolucion = "descartado" if regla == "descartado" else "marca"
            filas.append(
                {"tiempo": t, "regla": regla, "hs": df.at[t, "VHM0"], "magnitud": magnitud, "resolucion": resolucion}
            )
    out = pd.DataFrame(filas, columns=["tiempo", "regla", "hs", "magnitud", "resolucion"])
    return out.sort_values(["tiempo", "regla"]).reset_index(drop=True)


def serie(raiz_datos=None, nombre: str = NOMBRE, ruta_resoluciones: Path = RESOLUCIONES) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    dir_c = ruta_candidatos(raiz, nombre)
    etapa.exigir_publicada(dir_c)
    salida = ruta_serie(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_SERIE)
    insumos = {
        "candidatos": dir_c / "candidatos.csv",
        "maximos": dir_c / "maximos.csv",
        "serie_decodificada": dir_c / "serie_decodificada.parquet",
        "marcas": dir_c / "marcas.parquet",
        "hs_celdas": dir_c / "hs_celdas.parquet",
        "ficha_nodo": raiz / "processed" / "F01-sitio-y-fuentes" / "mapa-sitio-final" / nombre / "ficha_nodo.json",
    }
    with etapa.staging_protegido(salida, raices) as st:
        cand = pd.read_csv(insumos["candidatos"], parse_dates=["tiempo", "inicio", "fin"])
        maximos = pd.read_csv(insumos["maximos"], parse_dates=["tiempo"])["tiempo"]
        df = pd.read_parquet(insumos["serie_decodificada"])
        marcas = pd.read_parquet(insumos["marcas"])
        res = R.leer(ruta_resoluciones)
        manuales = cand[cand["decision"] == D.MANUAL]
        clase_dudoso = set(cand.loc[cand["motivo"] == f"clase:{D.DUDOSO}", "tiempo"])
        faltas = R.faltas(manuales, maximos, res, maximos_dudosos=sorted(clase_dudoso & set(maximos)))
        idx_res = R.indice(res)
        # excursión de cada candidato del detector: un descarte afecta a toda la excursión (v9, §17, F03-S-22)
        exc_de = {
            f.tiempo: (f.inicio, f.fin)
            for f in cand.itertuples()
            if f.regla in ("detector", "detector_revision", "dipolo")
        }

        def tramo(t: pd.Timestamp) -> pd.DatetimeIndex:
            ini, fin = exc_de.get(t, (t, t))
            return df.index[(df.index >= ini) & (df.index <= fin)]

        descartar: set[pd.Timestamp] = set()
        for t in pd.to_datetime(cand.loc[cand["decision"] == D.DESCARTAR, "tiempo"]):
            descartar |= set(tramo(t))
        for (t, _), r in idx_res.items():
            if r.decision == "descartar":
                descartar |= set(tramo(t))
        valido = ~df.index.isin(sorted(descartar))
        out = df.copy()
        out.loc[~valido, list(C.VARIABLES)] = np.nan  # un descarte afecta a todas las variables del registro
        out = pd.concat([out, marcas.add_prefix("marca_")], axis=1)
        out["valido"] = valido
        anual, mensual = reglas.cobertura(pd.Series(valido, index=df.index))
        cobertura_var = reglas.cobertura_por_variable(out, valido, ("VHM0", "VTPK"))
        # solo columnas booleanas: una magnitud continua (discrepancia de energía) no se suma (F03-K-01)
        booleanas = marcas.select_dtypes(include="bool").copy()
        booleanas["descartado"] = ~valido
        por_anio = booleanas.groupby(booleanas.index.year).sum()
        por_anio.index.name = "anio"
        por_regla = _marcas_por_regla(booleanas, df, marcas, cand, idx_res)
        h_sitio, h_modelo = _profundidades(raiz, nombre)
        completos = [a for a in sorted(set(df.index.year)) if a != C.ANIO_PARCIAL]
        largos = periodos.resumen(out[out["valido"]], completos, h_sitio, h_modelo)
        dudosos = [
            {"tiempo": str(t), "decision": r.decision, "motivo": r.motivo}
            for (t, regla), r in idx_res.items()
            if regla == R.REGLA_MAXIMO and (r.decision == "descartar" or "dudoso" in r.motivo.lower())
        ]
        # saltos compartidos por las vecinas que se mantienen por clase: van a la F4 y a la sensibilidad de la F9 (v8)
        dudosos += [
            {"tiempo": str(t), "decision": "mantener", "motivo": f"clase:{D.DUDOSO}"}
            for t in cand.loc[cand["motivo"] == f"clase:{D.DUDOSO}", "tiempo"]
        ]
        # un registro aparece una sola vez aunque lo listen su resolución y su clase: se unen los motivos
        unidos: dict[str, dict] = {}
        for x in dudosos:
            if x["tiempo"] in unidos:
                unidos[x["tiempo"]]["motivo"] += " | " + x["motivo"]
            else:
                unidos[x["tiempo"]] = dict(x)
        dudosos = [unidos[t] for t in sorted(unidos)]
        # qué significa «sin él» para la F9; se publican las dos variantes y elige el contrato de la F9 (v9, §17):
        # - sin el registro: la tormenta conserva su siguiente máximo;
        # - sin la excursión: se quitan su excursión y las de todo candidato del detector a ±48 h (F03-S-22)
        for x in dudosos:
            t = pd.Timestamp(x["tiempo"])
            ventana = df["VHM0"].loc[t - CONTEXTO : t + CONTEXTO]
            w = ventana.drop(t, errors="ignore")
            quitar = set(tramo(t))
            for t2 in exc_de:
                if abs(t2 - t) <= CONTEXTO:
                    quitar |= set(tramo(t2))
            w2 = ventana.drop(sorted(quitar), errors="ignore")
            x["hs_m"] = round(float(df.at[t, "VHM0"]), 2) if t in df.index else None
            x["hs_max_48h_sin_registro_m"] = round(float(w.max()), 2) if w.notna().any() else None
            x["hs_max_48h_sin_excursion_m"] = round(float(w2.max()), 2) if w2.notna().any() else None
            # solo pesa en la F9 si es el máximo de su ventana: si no, quitarlo no cambia el pico de la tormenta
            x["es_maximo_48h"] = bool(
                x["hs_m"] is not None
                and x["hs_max_48h_sin_registro_m"] is not None
                and x["hs_m"] > x["hs_max_48h_sin_registro_m"]
            )
        out.to_parquet(st / "serie_nodo.parquet")
        pd.read_parquet(insumos["hs_celdas"]).to_parquet(st / "vecinas_hs.parquet")
        anual.to_csv(st / "cobertura_anual.csv", lineterminator="\n")
        cobertura_var.to_csv(st / "cobertura.csv", index=False, lineterminator="\n")
        por_regla.to_csv(st / "marcas_por_regla.csv", index=False, date_format="%Y-%m-%d %H:%M", lineterminator="\n")
        mensual.to_csv(st / "cobertura_mensual.csv", lineterminator="\n")
        por_anio.to_csv(st / "marcas_por_anio.csv", lineterminator="\n")
        json_estricto.escribir(st / "periodos_largos.json", largos)
        json_estricto.escribir(
            st / "resumen_f03.json",
            {
                "registros": int(len(df)),
                "validos": int(valido.sum()),
                "descartados": int((~valido).sum()),
                "resoluciones": len(res),
                "sensibilidad_f9": dudosos,
                "sensibilidad_f9_definicion": (
                    "Dos variantes; cuál usa la F9 lo decide su contrato. hs_max_48h_sin_registro_m: se quita solo "
                    "el registro listado y la tormenta conserva su siguiente máximo. hs_max_48h_sin_excursion_m: se "
                    "quitan la excursión completa del registro y las de todo candidato del detector a ±48 h. Si "
                    "es_maximo_48h es falso, el registro no es el pico de su ventana y quitarlo no cambia el máximo."
                ),
                "faltas": faltas,
            },
        )
        gate_s = etapa.Gate(not faltas, {"resoluciones": len(res)}, faltas)
        return etapa.publicar(st, salida, gate_s, raices, insumos, raiz, None)


# --- CLI ----------------------------------------------------------------------------------------


def _cli(funcion, prog: str, argv=None) -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--datos", help="raíz de WAVE-DATA (si no, WAVE_DATA_DIR)")
    ap.add_argument("--nombre", default=NOMBRE)
    a = ap.parse_args(argv)
    try:
        print(f"Publicado en {funcion(a.datos, a.nombre)}")
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    return 0


def main_candidatos(argv=None) -> int:
    return _cli(candidatos, "datos-candidatos", argv)


def main_serie(argv=None) -> int:
    return _cli(serie, "datos-serie", argv)
