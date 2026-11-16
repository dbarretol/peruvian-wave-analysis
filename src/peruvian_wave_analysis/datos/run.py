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
        idx = df.index
        filas = []  # candidatos: tiempo, regla, decision, motivo, hs, inicio, fin
        # máximos, sobre la serie antes de todo descarte (§3.10); se necesitan aquí para la regla por clase
        maximos = D.maximos_independientes(df0["VHM0"])
        en_maximos = set(maximos.index)

        def agregar(i: int, regla: str, decision: str, motivo: str, exc: D.Excursion | None = None) -> None:
            ini, fin = (exc.inicio, exc.fin) if exc is not None else (i, i)
            filas.append(
                {
                    "tiempo": idx[i],
                    "regla": regla,
                    "decision": decision,
                    "motivo": motivo,
                    "hs": round(float(hs[i]), 2),
                    "inicio": idx[ini],
                    "fin": idx[fin],
                }
            )

        def por_clase(e: D.Excursion, regla: str) -> None:
            """Candidato del detector en la cola: regla por clase (v8, §16). Descartar un máximo sigue exigiendo la
            confirmación del usuario (§3.10), así que en ese caso queda manual."""
            r = D.resolver_por_clase(e, hs, tm02, vec)
            if r.decision == D.DESCARTAR and idx[e.pico] in en_maximos:
                agregar(e.pico, regla, D.MANUAL, f"clase:{r.motivo}:maximo", e)
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
        cand = pd.DataFrame(filas, columns=["tiempo", "regla", "decision", "motivo", "hs", "inicio", "fin"])
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
        cand = pd.read_csv(insumos["candidatos"], parse_dates=["tiempo"])
        maximos = pd.read_csv(insumos["maximos"], parse_dates=["tiempo"])["tiempo"]
        df = pd.read_parquet(insumos["serie_decodificada"])
        marcas = pd.read_parquet(insumos["marcas"])
        res = R.leer(ruta_resoluciones)
        manuales = cand[cand["decision"] == D.MANUAL]
        faltas = R.faltas(manuales, maximos, res)
        idx_res = R.indice(res)
        descartar = set(pd.to_datetime(cand.loc[cand["decision"] == D.DESCARTAR, "tiempo"]))
        descartar |= {t for (t, _), r in idx_res.items() if r.decision == "descartar"}
        valido = ~df.index.isin(sorted(descartar))
        out = df.copy()
        out.loc[~valido, list(C.VARIABLES)] = np.nan  # un descarte afecta a todas las variables del registro
        out = pd.concat([out, marcas.add_prefix("marca_")], axis=1)
        out["valido"] = valido
        anual, mensual = reglas.cobertura(pd.Series(valido, index=df.index))
        por_anio = marcas.groupby(marcas.index.year).sum(numeric_only=True)
        por_anio.index.name = "anio"
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
        # qué significa «sin él» para la F9: se quita solo ese registro y la tormenta conserva su siguiente máximo
        for x in dudosos:
            t = pd.Timestamp(x["tiempo"])
            w = df["VHM0"].loc[t - CONTEXTO : t + CONTEXTO].drop(t, errors="ignore")
            x["hs_m"] = round(float(df.at[t, "VHM0"]), 2) if t in df.index else None
            x["hs_max_48h_sin_registro_m"] = round(float(w.max()), 2) if w.notna().any() else None
        out.to_parquet(st / "serie_nodo.parquet")
        pd.read_parquet(insumos["hs_celdas"]).to_parquet(st / "vecinas_hs.parquet")
        anual.to_csv(st / "cobertura_anual.csv", lineterminator="\n")
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
                    "Quitar solo el registro listado; la tormenta conserva su siguiente máximo "
                    "(hs_max_48h_sin_registro_m), no se elimina entera."
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
