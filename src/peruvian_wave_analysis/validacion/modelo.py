"""Etapa 1 de la F5, ``validacion-modelo``: series del modelo en la región y en las boyas (contrato v6, §1 y §2).

Solo datos del modelo; ningún valor observado.

**Control automático (subconjunto de la F3, con su misma configuración; contrato §1):**
1. Excluyen el registro:
   - ``fuera_de_limites`` (``datos/reglas.py``): Hs en [0, 20] m; períodos en (0, 30] s;
   - ``acumulaciones``: los valores con frecuencia anómala de cada celda y variable (``VTPK`` con la regla discreta
     si la serie es discreta, como en la F3);
   - ``saturados``: solo para los períodos (``VTPK``, ``VTM02``) de las celdas de las boyas. Para Hs no aplica.
2. Marca sin excluir: el **detector de saltos** de la F3 con su ``k_desc`` publicado y la escala ``s`` de cada celda.
   La marca cubre todos los registros de la excursión. La sensibilidad de la etapa 3 quita las colocaciones marcadas.

No se aplican, y se declara en ``resumen_modelo.json``: peralte, coherencia entre particiones y resoluciones manuales
(contrato §1). El detector no se recalibra por celda: usa el ``k_desc`` del nodo.

**Región:** celdas con dato cuyo centro está a ≤ 200 km del nodo más media diagonal de celda (una celda así puede
contener puntos del círculo de 200 km). **Boyas:** la celda con dato más cercana a la posición de la boya.

**Gate (contrato §8, «control del modelo regional»):** la celda del nodo reproduce la Hs de la serie de la F3 y las
reglas automáticas dan en ella lo mismo que en la F3; rejilla de 3 h completa; cada boya con su celda.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import manifiesto, rutas
from ..comun import etapa, json_estricto, orquestador
from ..comun.errores import GateFallido
from ..datos import config as CF
from ..datos import detector as D
from ..datos import ingesta, reglas
from ..datos.run import ruta_candidatos, ruta_serie
from ..sitio.exposicion import distancia_km
from . import ndbc, waverys

NOMBRE = "marcona-200m"
DIR_ETAPA = Path(__file__).resolve().parent
ETAPA_MODELO = orquestador.Etapa("validacion-modelo", DIR_ETAPA, Path("."))
SEMIDIAGONAL_KM = np.hypot(CF.PASO_MALLA / 2, CF.PASO_MALLA / 2) * waverys.KM_POR_GRADO
DIST_MAX_BOYA_KM = 30.0  # la celda con dato más cercana a una boya en mar abierto está a < 15 km
NO_APLICADAS = [
    "peralte (usa VTPK y VTM02 en la celda; contrato §1)",
    "coherencia entre particiones (usa las particiones, no descargadas)",
    "resoluciones manuales del detector (inviables en cientos de celdas)",
    "recalibración de k por celda: se usa el k_desc del nodo con la s de cada celda",
]


def ruta_modelo(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F05-validacion-externa" / "modelo" / nombre


def nombre_celda(lat: float, lon: float) -> str:
    return f"{lat:+.1f}_{lon:+.1f}"


# --- Lectura ------------------------------------------------------------------------------------------


def leer_bloque(archivos: list[Path], variables: tuple[str, ...]) -> dict[str, xr.DataArray]:
    """(tiempo, lat, lon) por variable, decodificada y comprobada (unidad, escala, relleno) como en la F3."""
    partes: dict[str, list[xr.DataArray]] = {v: [] for v in variables}
    for r in archivos:
        with xr.open_dataset(r, mask_and_scale=True) as ds0:
            ds = ingesta._renombrar(ds0)
            for v in variables:
                if v not in ds:
                    raise ingesta.ErrorIngesta(f"{Path(r).name}: falta {v}")
                ingesta.comprobar_variable(ds[v], v)
                partes[v].append(ds[v].load())  # mismo tipo decodificado que la F3
    out = {}
    for v, ps in partes.items():
        da = xr.concat(ps, dim="time")
        da = da.assign_coords(
            lat=np.round(da["lat"].values.astype(float), 4), lon=np.round(da["lon"].values.astype(float), 4)
        )
        ingesta.comprobar_tiempo(pd.DatetimeIndex(da["time"].values))
        out[v] = da.sortby("lat").sortby("lon")
    return out


# --- Control por celda --------------------------------------------------------------------------------


def control_celda(x: pd.Series, variable: str, k_desc: float | None) -> tuple[pd.Series, dict, pd.Series]:
    """(máscara de excluidos, recuento por regla, marca del detector) de una serie de una celda.

    ``k_desc`` None: sin detector (los períodos no lo llevan, como en la F3).
    """
    df = pd.DataFrame({variable: x})
    excl = pd.Series(False, index=x.index)
    cuenta = {}
    if variable in CF.ALTURAS:
        lim = x.notna() & ((x < 0.0) | (x > CF.HS_MAX))
    else:
        lim = x.notna() & ((x <= 0.0) | (x > CF.T_MAX))
    cuenta["fuera_de_limites"] = int(lim.sum())
    excl |= lim
    discreto = variable == "VTPK" and reglas.es_discreto(x)
    vals = reglas.acumulaciones_discretas(x) if discreto else reglas.acumulaciones(x)
    acum = x.round(2).isin([round(v, 2) for v in vals]) & x.notna()
    cuenta["acumulaciones"] = int(acum.sum())
    cuenta["valores_acumulados"] = vals
    excl |= acum
    if variable in CF.PERIODOS:
        sat = reglas.saturados(df[variable], discreto)
        cuenta["saturados"] = int(sat.sum())
        excl |= sat
    marca = pd.Series(False, index=x.index)
    if k_desc is not None:
        hs = x.to_numpy(dtype=float)
        s = D.escala_s(hs)
        cuenta["s_m"] = s
        if np.isfinite(s) and s > 0:
            m = np.zeros(hs.size, bool)
            for e in D.detectar(hs, k_desc, s):
                m[e.inicio : e.fin + 1] = True
            marca = pd.Series(m, index=x.index)
    cuenta["marcados_detector"] = int(marca.sum())
    cuenta["excluidos"] = int(excl.sum())
    return excl, cuenta, marca


def celdas_region(da: xr.DataArray, nodo=waverys.NODO, radio_km: float = waverys.RADIO_KM) -> pd.DataFrame:
    """Celdas con dato a ≤ radio + media diagonal del nodo, con su distancia, en orden (lat, lon)."""
    con_dato = da.notnull().any("time")
    filas = []
    for la in da["lat"].values:
        for lo in da["lon"].values:
            d = float(distancia_km(nodo[0], nodo[1], la, lo))
            if d <= radio_km + SEMIDIAGONAL_KM and bool(con_dato.sel(lat=la, lon=lo)):
                filas.append({"celda": nombre_celda(la, lo), "lat": float(la), "lon": float(lo), "dist_nodo_km": d})
    return pd.DataFrame(filas)


def celda_boya(da: xr.DataArray, boya: ndbc.Boya) -> dict:
    con_dato = da.notnull().any("time")
    mejor = None
    for la in da["lat"].values:
        for lo in da["lon"].values:
            if not bool(con_dato.sel(lat=la, lon=lo)):
                continue
            d = float(distancia_km(boya.lat, boya.lon, la, lo))
            clave = (round(d, 6), float(la), float(lo))
            if mejor is None or clave < mejor:
                mejor = clave
    if mejor is None:
        raise ingesta.ErrorIngesta(f"Boya {boya.estacion}: ninguna celda con dato en el recuadro")
    d, la, lo = mejor
    return {"boya": boya.estacion, "celda": nombre_celda(la, lo), "lat": la, "lon": lo, "dist_boya_km": d}


# --- Etapa --------------------------------------------------------------------------------------------


def _archivos(entradas: dict, prefijo: str, marca: str) -> list[str]:
    return sorted(r for r in entradas if r.startswith(prefijo) and marca in r and r.endswith(".nc"))


def _csv(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, index=False, lineterminator="\n", float_format="%.6g")


def modelo(raiz_datos=None, nombre: str = NOMBRE) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    raw = rutas.raw(raiz)
    entradas = manifiesto.leer(raw)
    rel_reg = _archivos(entradas, "waverys/regional/", "waverys_regional_")
    rel_boya = {b.estacion: _archivos(entradas, "waverys/boyas/", f"waverys_boya_{b.estacion}_") for b in ndbc.BOYAS}
    insumos = {f"raw:{r}": raw / r for r in rel_reg + sum(rel_boya.values(), [])}
    esperados = {f"raw:{r}": entradas[r]["sha256"] for r in rel_reg + sum(rel_boya.values(), [])}
    dir_c, dir_s = ruta_candidatos(raiz, nombre), ruta_serie(raiz, nombre)
    etapa.exigir_publicada(dir_c)
    etapa.exigir_publicada(dir_s)
    insumos |= {
        "resumen_candidatos": dir_c / "resumen_candidatos.json",
        "candidatos": dir_c / "candidatos.csv",
        "serie_nodo": dir_s / "serie_nodo.parquet",
    }
    etapa.exigir_insumos(insumos, esperados)  # G2: hash distinto del manifiesto de raw → error
    salida = ruta_modelo(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_MODELO)
    with etapa.staging_protegido(salida, raices) as st:
        motivos: list[str] = []
        if not rel_reg:
            raise ingesta.ErrorIngesta(
                "No hay WAVERYS regional en el manifiesto: `validacion-descarga waverys-regional`"
            )
        f3 = json_estricto.leer(insumos["resumen_candidatos"])
        k_desc = float(f3["k_desc"])
        nodo_f3 = pd.read_parquet(insumos["serie_nodo"], columns=["VHM0", "valido"])

        # Región
        hs = leer_bloque([raw / r for r in rel_reg], waverys.VARIABLES_REGION)["VHM0"]
        t = pd.DatetimeIndex(hs["time"].values, name="tiempo")
        if not t.equals(pd.DatetimeIndex(nodo_f3.index)):
            motivos.append("G2: la rejilla de tiempo regional no es la de la serie del nodo de la F3")
        celdas = celdas_region(hs)
        valores, marcas, filas = {}, {}, []
        for c in celdas.itertuples():
            x = pd.Series(hs.sel(lat=c.lat, lon=c.lon).values.astype(float), index=t)
            excl, cuenta, marca = control_celda(x, "VHM0", k_desc)
            valores[c.celda] = x.where(~excl).astype("float32")
            marcas[c.celda] = marca
            filas.append(
                {
                    **c._asdict(),
                    **{k: v for k, v in cuenta.items() if k != "valores_acumulados"},
                    "valores_acumulados": ";".join(f"{v:g}" for v in cuenta["valores_acumulados"]),
                    "sin_dato": int(x.isna().sum()),
                }
            )
        tabla = pd.DataFrame(filas).drop(columns=["Index"], errors="ignore")
        pd.DataFrame(valores, index=t).to_parquet(st / "modelo_region.parquet")
        pd.DataFrame(marcas, index=t).to_parquet(st / "marcas_region.parquet")
        _csv(tabla, st / "celdas_region.csv")

        # Coherencia con la F3 en el nodo (§8)
        c_nodo = nombre_celda(*waverys.NODO)
        en_nodo = tabla[tabla["celda"] == c_nodo]
        coherencia = {"celda_nodo_presente": bool(len(en_nodo))}
        if len(en_nodo):
            x_nodo = pd.Series(hs.sel(lat=waverys.NODO[0], lon=waverys.NODO[1]).values.astype(float), index=t)
            dif = float(np.nanmax(np.abs(x_nodo.to_numpy() - nodo_f3["VHM0"].to_numpy())))
            cand = pd.read_csv(insumos["candidatos"])
            f3_lim = int((cand["regla"] == "limite_VHM0").sum())
            f3_acum = sorted(f3.get("acumulaciones", {}).get("VHM0", []))
            fila = en_nodo.iloc[0]
            coherencia |= {
                "max_dif_hs_m": dif,
                "fuera_de_limites": {"f5": int(fila["fuera_de_limites"]), "f3": f3_lim},
                "acumulaciones": {"f5": fila["valores_acumulados"], "f3": ";".join(f"{v:g}" for v in f3_acum)},
                "s_m": {"f5": float(fila["s_m"]), "f3": float(f3["s_m"])},
            }
            if dif > 1e-6:
                motivos.append(f"§8: la celda del nodo difiere de la serie de la F3 en {dif:.3g} m")
            if int(fila["fuera_de_limites"]) != f3_lim:
                motivos.append("§8: fuera_de_límites en el nodo no coincide con la F3")
            if fila["valores_acumulados"] != coherencia["acumulaciones"]["f3"]:
                motivos.append("§8: acumulaciones en el nodo no coinciden con la F3")
            if not np.isclose(float(fila["s_m"]), float(f3["s_m"]), rtol=1e-9, atol=1e-12):
                motivos.append("§8: la escala s del detector en el nodo no coincide con la F3")
        else:
            motivos.append("§8: la celda del nodo no está entre las celdas regionales")

        # Boyas
        filas_b, series_b = [], []
        for b in ndbc.BOYAS:
            if not rel_boya[b.estacion]:
                motivos.append(f"G2: no hay WAVERYS en la boya {b.estacion}")
                continue
            bloque = leer_bloque([raw / r for r in rel_boya[b.estacion]], waverys.VARIABLES_BOYA)
            cel = celda_boya(bloque["VHM0"], b)
            if cel["dist_boya_km"] > DIST_MAX_BOYA_KM:
                motivos.append(f"boya {b.estacion}: la celda con dato está a {cel['dist_boya_km']:.1f} km")
            tb = pd.DatetimeIndex(bloque["VHM0"]["time"].values, name="tiempo")
            esperado = pd.date_range(f"{b.anios[0]}-01-01", f"{b.anios[-1]}-12-31 21:00", freq="3h")
            if not tb.equals(esperado):
                motivos.append(f"G2: rejilla de tiempo incompleta en la boya {b.estacion}")
            df = pd.DataFrame(index=tb)
            for v in waverys.VARIABLES_BOYA:
                x = pd.Series(bloque[v].sel(lat=cel["lat"], lon=cel["lon"]).values.astype(float), index=tb)
                excl, cuenta, marca = control_celda(x, v, k_desc if v == "VHM0" else None)
                df[v] = x.where(~excl).astype("float32")
                if v == "VHM0":
                    df["marca_detector"] = marca
                cel |= {
                    f"{v}:{k}": (";".join(f"{z:g}" for z in w) if k == "valores_acumulados" else w)
                    for k, w in cuenta.items()
                }
                cel[f"{v}:sin_dato"] = int(x.isna().sum())
            df.insert(0, "boya", b.estacion)
            series_b.append(df.reset_index())
            filas_b.append(cel)
        pd.concat(series_b, ignore_index=True).to_parquet(st / "modelo_boyas.parquet")
        _csv(pd.DataFrame(filas_b), st / "celdas_boyas.csv")

        resumen = {
            "celdas_region": int(len(tabla)),
            "radio_km": waverys.RADIO_KM,
            "margen_celda_km": round(float(SEMIDIAGONAL_KM), 3),
            "pasos": int(len(t)),
            "inicio": str(t[0]),
            "fin": str(t[-1]),
            "k_desc_f3": k_desc,
            "excluidos_region": {k: int(tabla[k].sum()) for k in ("fuera_de_limites", "acumulaciones", "excluidos")},
            "marcados_detector_region": int(tabla["marcados_detector"].sum()),
            "celdas_con_exclusiones": int((tabla["excluidos"] > 0).sum()),
            "coherencia_nodo_f3": coherencia,
            "boyas": filas_b,
            "reglas_no_aplicadas": NO_APLICADAS,
            "valores_observados_leidos": False,
        }
        json_estricto.escribir(st / "resumen_modelo.json", resumen)
        criterios = {
            "celdas_region": resumen["celdas_region"],
            "max_dif_hs_nodo_m": coherencia.get("max_dif_hs_m"),
            "boyas": [f["boya"] for f in filas_b],
        }
        return etapa.publicar(st, salida, etapa.Gate(not motivos, criterios, motivos), raices, insumos, raiz, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validacion-modelo", description="Etapa 1 de la F5: series del modelo.")
    ap.add_argument("--datos")
    ap.add_argument("--nombre", default=NOMBRE)
    a = ap.parse_args(argv)
    try:
        salida = modelo(a.datos, a.nombre)
    except (GateFallido, ingesta.ErrorIngesta) as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
