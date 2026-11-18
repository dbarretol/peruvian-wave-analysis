"""Etapa 3 de la F5, ``validacion`` (contrato v6, §2–§6, §19 y §20). **Lee los valores observados.**

Orden:
0. **G4, antes de abrir ningún valor:** la etapa 2 está publicada con el gate aprobado, cada archivo coincide con
   su manifiesto (también ``potencia.csv`` y ``potencia.json``), su huella de código es la de hoy y declara
   ``valores_observados_leidos = False``. Si algo falla, la etapa se detiene sin leer valores.
1. Valores observados y control de calidad (§3.2, §3.3); comprobación de que la geometría es la de la etapa 2.
2. σ de 1 Hz medido y función de estructura (§3.5, §20.3): elige la configuración con o sin máximos.
3. Por lado, bootstrap de B = 9 999 con SIMEX y calibración; V1, V2 y V3 con sus sensibilidades (§4).
4. SI y ρ observados y control de calidad: si alguno es peor, potencia recalculada (réplicas reanudables).
5. Categorías, marcas, escenarios, corrección y fila de la fuente (§6, §19, §20.6).
6. Sesgo del estimador con el σ medido (§20.5, descriptivo), períodos, saltos y figuras.

``--previo``: lo mismo con valores **sintéticos** (de la geometría y del modelo; no abre ninguna variable observada),
en ``interim/``, para cronometrar. Las réplicas se cronometran con una sola réplica de cada clase.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..sitio import observaciones
from ..validacion import colocacion as K
from ..validacion import config as CV
from ..validacion import etapa_potencia as EP
from ..validacion import geometria as G
from ..validacion import potencia as P
from ..validacion import reanudable
from . import complementos, figuras, lectura, ruido
from . import config as C
from . import decision as D
from . import prueba as PR
from . import simulacion_obs as SO

NOMBRE = "marcona-200m"
DIR_ETAPA = Path(__file__).resolve().parent
ETAPA_VALIDACION = orquestador.Etapa("validacion", DIR_ETAPA, Path("."))
ARCHIVOS_POTENCIA = (
    "potencia.csv",
    "potencia.json",
    "efecto_minimo.csv",
    "escalon_modelo.csv",
    "replicas_potencia.parquet",
    "colocaciones_geometria.parquet",
)
LADOS_ALT = ("T0", "T0+ERS1+TOPEX", "A", "A_hasta_2026", "A_solo_HY2A", "A1", "A2", "A2+HY2B_2023", "A2+HY2B_2026")
LADOS_ALT25 = ("T0", "A", "A1", "A2")  # sensibilidad ≥ 25 km de la costa (§3.2)
LADOS_BOYA = ("T0b", "T0b_hasta_1992", "T1b", "A1b", "A2b")
VARIANTES = ("principal", "ruido_bajo", "ruido_alto", "representatividad")
EXTRA_25 = (("V1a_costa25", "V1", "a", "T0@25", "A@25"), ("V2a_costa25", "V2", "a", "A1@25", "A2@25"))
FACTOR_DE = {"V1": {"a": "V1a", "b": "V1b"}, "V2": {"a": "V2a", "b": "V2b"}}


def ruta_validacion(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F05-validacion-externa" / "validacion" / nombre


def ruta_previo(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "interim" / "F05-validacion-externa" / "validacion-previo" / nombre


def ruta_avance(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "interim" / "F05-validacion-externa" / "validacion-avance" / nombre


# --- G4 ----------------------------------------------------------------------------------------------------


def verificar_g4(raiz: Path, nombre: str = NOMBRE) -> dict:
    """Comprobaciones de G4 antes de leer valores. Lanza ``GateFallido`` si alguna falla (no se lee nada)."""
    d = EP.ruta_potencia(raiz, nombre)
    if not (d / "potencia.csv").is_file():
        raise GateFallido(f"G4: falta potencia.csv en {d}; la etapa 3 no lee valores sin la potencia publicada.")
    man = etapa.exigir_publicada(d)  # hash de cada archivo contra el manifiesto; falla si cambió alguno
    if not man["gate"]["aprobado"]:
        raise GateFallido("G4: la etapa 2 tiene el gate fallido.")
    for f in ARCHIVOS_POTENCIA:
        if f not in man["salidas"]:
            raise GateFallido(f"G4: {f} no está en el manifiesto de la etapa 2.")
    huella = orquestador.huella_etapa(EP.ETAPA_POTENCIA)
    if huella != man["huella_codigo"]:
        raise GateFallido(
            "G4: el código de la etapa 2 cambió después de publicar la potencia "
            f"(huella publicada {man['huella_codigo'][:12]}, actual {huella[:12]})."
        )
    pj = json_estricto.leer(d / "potencia.json")
    if pj.get("valores_observados_leidos") is not False:
        raise GateFallido("G4: potencia.json no declara valores_observados_leidos = False.")
    return {
        "potencia_csv_sha256": man["salidas"]["potencia.csv"],
        "potencia_json_sha256": man["salidas"]["potencia.json"],
        "huella_codigo_etapa_2": man["huella_codigo"],
        "comprobado_antes_de_leer_valores": True,
    }


def insumos(raiz: Path, nombre: str) -> tuple[dict[str, Path], dict[str, str]]:
    ins, esp = EP.insumos(raiz, nombre)
    dp = EP.ruta_potencia(raiz, nombre)
    man = etapa.exigir_publicada(dp)
    for f in ARCHIVOS_POTENCIA:
        ins[f"potencia:{f}"], esp[f"potencia:{f}"] = dp / f, man["salidas"][f]
    from ..datos.run import ruta_candidatos  # noqa: PLC0415

    dc = ruta_candidatos(raiz, nombre)
    etapa.exigir_publicada(dc)
    ins["serie_sin_descartar"] = dc / "serie_decodificada.parquet"
    d4 = Path(raiz) / "processed" / "F04-homogeneidad" / "homogeneidad" / nombre
    ins["prueba_altimetros"] = d4 / "prueba_altimetros.json"
    return ins, esp


# --- Lados ---------------------------------------------------------------------------------------------------


def _lado_df(so: pd.DataFrame, boy: pd.DataFrame, nombre: str) -> pd.DataFrame:
    if nombre.endswith("@25"):
        base = CV.LADOS[nombre[:-3]]
        x = so[so["costa_min_km"] == CV.COSTA_MIN_SENS_KM]
        return x[K.en_lado(x, base, "mision")]
    if nombre == C.LADO_ERS1_TOPEX.nombre:
        x = so[so["costa_min_km"] == CV.COSTA_MIN_KM]
        return x[K.en_lado(x, C.LADO_ERS1_TOPEX, "mision")]
    lado = CV.LADOS[nombre]
    if lado.fuente == "altimetría":
        x = so[so["costa_min_km"] == CV.COSTA_MIN_KM]
        return x[K.en_lado(x, lado, "mision")]
    return boy[K.en_lado(boy, lado, "boya")]


def _sigmas(df: pd.DataFrame, alt: bool, tabla: dict, v_rep: float | None) -> dict[str, np.ndarray]:
    if alt:
        s = ruido.sigma_superobs(df, tabla)
        out = {"principal": s, "ruido_bajo": C.SENS_SIGMA_ALT[0] * s, "ruido_alto": C.SENS_SIGMA_ALT[1] * s}
        out["representatividad"] = np.sqrt(s**2 + v_rep) if v_rep is not None else s
        return out
    s = ruido.sigma_boya(df)
    return {
        "principal": s,
        "ruido_bajo": ruido.sigma_boya(df, C.SENS_SIGMA_BOYA[0]),
        "ruido_alto": ruido.sigma_boya(df, C.SENS_SIGMA_BOYA[1]),
        "representatividad": s,
    }


def medir_lados(so, boy, tabla, v_rep, B) -> dict[str, PR.LadoObs]:
    nombres = [*LADOS_ALT, C.LADO_ERS1_TOPEX.nombre, *(f"{n}@25" for n in LADOS_ALT25), *LADOS_BOYA]
    out = {}
    for n in nombres:
        df = PR.ordenar(_lado_df(so, boy, n))
        if len(df) < 2:
            continue
        out[n] = PR.medir_lado(n, df, _sigmas(df, n not in LADOS_BOYA, tabla, v_rep), B)
    return out


# --- Pruebas -------------------------------------------------------------------------------------------------


def comparaciones(lados, errores, factores) -> dict:
    """Cada comparación (principal y sensibilidades) en cada configuración de errores y de ruido."""
    e2 = PR.escalar(errores, 2.0)
    lista = [(c.nombre, c.familia, c.fuente, c.lado1, c.lado2) for c in CV.COMPARACIONES] + list(EXTRA_25)
    out = {}
    for nombre, fam, fuente, l1, l2 in lista:
        if l1 not in lados or l2 not in lados:
            out[nombre] = {"familia": fam, "fuente": fuente, "estado": "sin colocaciones"}
            continue
        alfa = CV.ALFA_FUENTE
        f = factores.get(FACTOR_DE[fam][fuente], 1.0)
        a, b = lados[l1], lados[l2]
        r = {"familia": fam, "fuente": fuente, "lado1": l1, "lado2": l2}
        r["principal"] = PR.diferencia(a, b, "principal", errores, False, alfa, f)
        r["calibracion_x2"] = PR.diferencia(a, b, "principal", e2, False, alfa, f)
        for var in ("ruido_bajo", "ruido_alto", "representatividad"):
            r[var] = PR.diferencia(a, b, var, errores, False, alfa, f)
        if fuente == "b":
            fl = factores.get("V1b_lado", 1.0) if fam == "V1" else f
            r["termino_por_lado"] = PR.diferencia(a, b, "principal", errores, True, alfa, fl)
        out[nombre] = r
    return out


def _s_mod(esc: pd.DataFrame, comp: str) -> dict[str, float]:
    x = esc[esc["comparacion"] == comp]
    return dict(zip(x["estadistico"], x["s_mod"].astype(float), strict=True))


def decidir(comps, v3conf, var, pj: dict, esc, coherencia, absorbe, pot_final) -> dict:
    """§4.1, §6.1 y §6.2 con la variante de ruido ``var`` (principal o representatividad)."""
    atrib, trasl = pj["atribuye"], pj["traslado_p90"]
    fams = {}
    for fam, (ca, cb) in {"V1": ("V1a", "V1b"), "V2": ("V2a", "V2b")}.items():
        fuentes, marcas_f, cats = {}, {}, {}
        for fuente, comp in (("a", ca), ("b", cb)):
            r = comps.get(comp, {})
            if "principal" not in r:
                fuentes[fuente] = {"atribuye": False, "rechaza": False, "signo": 0, "descartada": False}
                cats[comp] = "sin colocaciones"
                continue
            res = r[var]
            p_ind = pot_final.get("V1b_lado" if comp == "V1b" else comp, np.nan)
            desc = (not res["rechaza"]) and np.isfinite(p_ind) and p_ind >= 0.8
            fuentes[fuente] = {
                "atribuye": bool(atrib.get(comp, False)),
                "rechaza": res["rechaza"],
                "signo": res["signo"],
                "descartada": bool(desc),
            }
            cats[comp] = D.SIN_ESCALON if not atrib.get(comp, False) else D.categoria(res["rechaza"], p_ind)
            marcas_f[fuente] = D.marcas_fuente(
                res,
                r["calibracion_x2"],
                r.get("termino_por_lado"),
                _s_mod(esc, comp),
                trasl.get(comp, "dudoso"),
                incoherente=(fam == "V1" and fuente == "a" and coherencia.get("incoherencia", False)),
            )
        if fam == "V1":
            activas = [k for k, f in fuentes.items() if f["atribuye"]]
            if len(activas) == 2:
                pc = min(pot_final.get("V1 conjunta", np.nan), pot_final.get("V1 conjunta con lado", np.nan))
            else:
                pc = pot_final.get({"a": "V1a", "b": "V1b_lado"}[activas[0]], np.nan) if activas else np.nan
        else:
            pc = pot_final.get("V2a", np.nan) if fuentes["a"]["atribuye"] else np.nan
        f = D.familia(fuentes, pc)
        f["marcas_fuente"] = marcas_f
        f["categorias_por_fuente"] = cats
        fams[fam] = f
    v3 = v3conf[var]
    suf = "" if v3["con_maximos"] else "_sinmax"
    cat_a = D.categoria(v3["V3a"]["rechaza"], pot_final.get(f"V3a{suf}", np.nan))
    cat_b = D.categoria(
        v3["V3b"]["rechaza"],
        pot_final.get(f"V3b{suf}", np.nan),
        puede_descartar=bool(pj["validacion_por_prueba"]["V3b"]),
    )
    esc_ = D.escenarios(fams["V1"], fams["V2"], cat_a, absorbe)
    corr = D.variantes(cat_a, cat_b, v3)
    return {"familias": fams, "V3a": cat_a, "V3b": cat_b, "escenarios": esc_, "correccion": corr}


def _resumen_decision(d: dict) -> tuple:
    return (
        d["familias"]["V1"]["categoria"],
        d["familias"]["V2"]["categoria"],
        d["V3a"],
        d["V3b"],
        d["escenarios"]["E0"],
        d["escenarios"]["E2"],
        tuple(v["nombre"] for v in d["correccion"]["variantes"]),
    )


def potencias_finales(pot_nom: D.Potencias, pot_rec: D.Potencias | None) -> tuple[dict, dict, dict]:
    pruebas = [
        ("V1a", CV.ALFA_FUENTE),
        ("V1b", CV.ALFA_FUENTE),
        ("V1b_lado", CV.ALFA_FUENTE),
        ("V2a", CV.ALFA_FUENTE),
        ("V3a", CV.ALFA_FAMILIA),
        ("V3b", CV.ALFA_FAMILIA),
        ("V3a_sinmax", CV.ALFA_FAMILIA),
        ("V3b_sinmax", CV.ALFA_FAMILIA),
    ]

    def tabla(p: D.Potencias) -> dict:
        t = {k: p.individual(k, a) for k, a in pruebas}
        t["V1 conjunta"] = p.conjunta_v1(False)
        t["V1 conjunta con lado"] = p.conjunta_v1(True)
        return t

    nom = tabla(pot_nom)
    rec = tabla(pot_rec) if pot_rec is not None else {}
    fin = {k: min(v, rec[k]) if k in rec else v for k, v in nom.items()}
    return nom, rec, fin


# --- Tablas --------------------------------------------------------------------------------------------------


def tabla_sesgo(lados: dict[str, PR.LadoObs], errores: dict, con_max: bool) -> pd.DataFrame:
    filas = []
    for n, lo in lados.items():
        med = lo.med["principal"]
        rc = PR.con_cal(lo, "principal", errores)
        fuente = "altimetría" if lo.L.fuente == "altimetría" else f"boya {lo.L.fuente}"
        base = {"fuente": fuente, "lado": n, "colocaciones": lo.L.n, "unidades": int(np.unique(lo.L.unidad).size)}
        for k, e in enumerate(PR.ESTS):
            lo95, hi95 = PR.ic(rc[:, k])
            filas.append(
                base
                | {
                    "estadistico": e,
                    "estimable": bool(med.estimable[k]),
                    "B_bruto": float(lo.bruto[k]),
                    "B_descontado": float(med.B[k]),
                    "R_bruto": float(np.exp(lo.bruto[k])),
                    "R_descontado": float(np.exp(med.B[k])),
                    "R_ic95_lo": float(np.exp(lo95)),
                    "R_ic95_hi": float(np.exp(hi95)),
                    "B_ruido_bajo": float(lo.med["ruido_bajo"].B[k]),
                    "B_ruido_alto": float(lo.med["ruido_alto"].B[k]),
                    "B_representatividad": float(lo.med["representatividad"].B[k]),
                }
            )
        v = PR.v3(lo, "principal", errores, con_max, {})
        for nom, blk in (("cola", v["B_cola"]), ("alta", v["B_alta"])):
            b = blk["estimado"]
            icb = blk["ic95"] or [np.nan, np.nan]
            filas.append(
                base
                | {
                    "estadistico": nom,
                    "estimable": b is not None,
                    "B_descontado": b if b is not None else np.nan,
                    "R_descontado": float(np.exp(b)) if b is not None else np.nan,
                    "R_ic95_lo": float(np.exp(icb[0])),
                    "R_ic95_hi": float(np.exp(icb[1])),
                }
            )
    return pd.DataFrame(filas)


def dentro_de_ic(tabla: pd.DataFrame, comps: dict) -> list[str]:
    """G6: cada estimado puntual dentro de su IC (los estimables)."""
    malos = []
    t = tabla[tabla["estimable"].astype(bool)]
    for r in t.itertuples():
        if np.isfinite(r.R_ic95_lo) and not (r.R_ic95_lo - 1e-12 <= r.R_descontado <= r.R_ic95_hi + 1e-12):
            malos.append(f"{r.lado} {r.estadistico}")
    for nombre, c in comps.items():
        p = c.get("principal")
        if not p:
            continue
        for e, v in p["estimado"].items():
            lo_hi = p["ic95"][e]
            if v is not None and lo_hi and not (lo_hi[0] - 1e-12 <= v <= lo_hi[1] + 1e-12):
                malos.append(f"{nombre} {e}")
    return malos


def _csv(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, index=False, float_format="%.6g", lineterminator="\n")


# --- Etapa -----------------------------------------------------------------------------------------------------


def validacion_etapa(
    raiz_datos=None, nombre: str = NOMBRE, procesos_: int = 4, previo: bool = False, B: int = CV.B_BOOT
) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    g4 = verificar_g4(raiz, nombre)  # antes de nada: sin esto no se abre ningún valor
    ins, esp = insumos(raiz, nombre)
    hashes = etapa.exigir_insumos(ins, esp)  # G2
    salida = ruta_previo(raiz, nombre) if previo else ruta_validacion(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_VALIDACION)
    clave = reanudable.clave(orquestador.huella_etapa(ETAPA_VALIDACION), hashes, B, previo)
    avance = ruta_avance(raiz, nombre) / clave
    tiempos: dict[str, float] = {}
    print(f"G4 comprobado: potencia.csv {g4['potencia_csv_sha256'][:12]}…; avance en {avance}", flush=True)

    def marca(nombre_paso, t0):
        tiempos[nombre_paso] = round(time.time() - t0, 1)
        print(f"  {nombre_paso}: {tiempos[nombre_paso]:.0f} s", flush=True)
        return time.time()

    with etapa.staging_protegido(salida, raices) as st:
        t = time.time()
        raw = rutas.raw(raiz)
        region = pd.read_parquet(ins["modelo_region"])
        marcas_r = pd.read_parquet(ins["marcas_region"])
        nodo = pd.read_parquet(ins["serie_nodo"], columns=["VHM0", "valido"])
        sect = pd.read_csv(ins["exposicion"])
        modelo_boyas = pd.read_parquet(ins["modelo_boyas"])
        pj = json_estricto.leer(ins["potencia:potencia.json"])
        esc = pd.read_csv(ins["potencia:escalon_modelo.csv"])
        mde = pd.read_csv(ins["potencia:efecto_minimo.csv"])
        geo = pd.read_parquet(ins["potencia:colocaciones_geometria.parquet"])
        alt_r = [p for k, p in sorted(ins.items()) if k.startswith("raw:altimetria/")]
        boy_r = {k.removeprefix("raw:"): p for k, p in sorted(ins.items()) if k.startswith("raw:boyas/")}

        # 1. Valores observados (o sintéticos en el previo)
        if previo:
            rng = semillas.generador("f05", "previo-sintetico")
            med = lectura.sintetico_altimetria(observaciones.leer(alt_r), region, rng)
            regs = lectura.sintetico_boyas(boy_r, modelo_boyas, rng)
        else:
            med = lectura.leer_altimetria(alt_r)
            regs = lectura.leer_boyas(boy_r)
        so, puntos, qc_alt, claves = lectura.colocaciones_altimetria(
            med, raw / G.GEBCO, G.tierra(raw), sect, region, marcas_r, nodo["VHM0"]
        )
        boy, qc_boy = lectura.colocaciones_boyas(regs, modelo_boyas, nodo["VHM0"])
        geo_alt = geo[geo["fuente"] == "altimetría"]
        k2 = set(map(tuple, geo_alt[["pasada", "celda", "costa_min_km"]].astype(str).to_numpy()))
        k3 = set(map(tuple, claves[["pasada", "celda", "costa_min_km"]].astype(str).to_numpy()))
        geometria_igual = k2 == k3
        t = marca("lectura y control de calidad", t)

        # 2. Ruido de 1 Hz y función de estructura
        sig = ruido.sigma_1hz(puntos)
        tabla_s = ruido.tabla_sigma(sig)
        pA = puntos[(puntos["dist_costa_km"] >= CV.COSTA_MIN_KM) & K.en_lado(puntos, CV.LADOS["A"], "mision")]
        estr = ruido.estructura(pA, tabla_s)
        regla_activa = estr["supera_umbral"]
        con_max = not regla_activa
        v_rep = estr["varianza_pequena_escala"] if regla_activa else None
        t = marca("ruido de 1 Hz y función de estructura", t)

        # 3. Lados y pruebas
        lados = medir_lados(so, boy, tabla_s, v_rep, B)
        errores = PR.errores_calibracion(lados, B)
        factores = pj["factores_recalibrados"]
        comps = comparaciones(lados, errores, factores)
        v3conf = {}
        for var in ("principal", "representatividad"):
            v3conf[var] = PR.v3(lados["A"], var, errores, con_max, factores) | {"con_maximos": con_max}
        v3_otra = PR.v3(lados["A"], "principal", errores, not con_max, factores) | {"con_maximos": not con_max}
        v3_cal2 = PR.v3(lados["A"], "principal", PR.escalar(errores, 2.0), con_max, factores)
        coher = (
            PR.coherencia_t0(lados["T0"], lados["ERS1+TOPEX"], errores, C.ESTADISTICOS_COHERENCIA)
            if "ERS1+TOPEX" in lados
            else {"incoherencia": False, "estado": "sin colocaciones de ERS-1 ni TOPEX antes de 1993"}
        )
        absorbe = PR.absorcion_v2(lados["A1"], lados["A2"], errores, C.TOL_V2_ABSORBE)
        grad = PR.gradiente(
            lados["A"], lados["A"].sigmas["principal"], C.N_BANDAS_GRADIENTE, B, C.DIST_COSTA_NODO_KM, C.TOL_GRADIENTE
        )
        t = marca("bootstrap, SIMEX y pruebas", t)

        # 4. SI y ρ observados, control de calidad y potencia recalculada
        si_obs = {n: PR.si_observado(lados[n], lados[n].sigmas["principal"]) for n in P.LADOS_SIM if n in lados}
        rho_obs = {n: PR.rho_observado(lados[n]) for n in P.LADOS_SIM if n in lados}
        nominal = {n: CV.SI_NOMINAL[P.CLASE_SI[n]] for n in si_obs}
        peor = {n: bool(si_obs[n] > nominal[n] or rho_obs[n] > CV.RHO_NOMINAL) for n in si_obs}
        geo_boy = geo[geo["fuente"] == "boya"]
        geo_alt50 = geo_alt[geo_alt["costa_min_km"] == CV.COSTA_MIN_KM]
        so50 = so[so["costa_min_km"] == CV.COSTA_MIN_KM]
        perdida = lectura.fraccion_perdida(geo_alt50, so50, ["T0", "A", "A1", "A2"], "mision")
        perdida |= lectura.fraccion_perdida(geo_boy, boy, ["T0b", "T1b", "A1b", "A2b"], "boya")
        qc_recalculo = {n: v > C.FRACCION_QC_RECALCULO for n, v in perdida.items()}
        recalcular = any(peor.values()) or any(qc_recalculo.values())
        punto_obs, punto_desc = SO.punto_observado(si_obs, rho_obs)
        efectos = pj["efectos_por_fuente"]
        nominal_reps = D.replicas_desde_tabla(
            pd.read_parquet(ins["potencia:replicas_potencia.parquet"]).query("punto == @P.NOMINAL")
        )
        pot_nom = D.Potencias(nominal_reps, efectos, factores)
        ruta_alt, ruta_boy = st / "_alt.parquet", st / "_boy.parquet"
        cols_alt = ["mision", "pasada", "celda", "tiempo", "n", "tramo_cal", "costa_min_km", "hs_mod", "unidad"]
        so[cols_alt].to_parquet(ruta_alt)
        boy[["boya", "tiempo", "k", "hs_mod", "unidad"]].to_parquet(ruta_boy)
        lados_qc = P.preparar_lados(pd.read_parquet(ruta_alt), pd.read_parquet(ruta_boy))
        pot_rec, recalculo = None, {"necesario": recalcular, "punto": punto_desc, "por_lado_peor": peor,
                                    "perdida_qc": perdida, "qc_supera_1_por_ciento": qc_recalculo}  # fmt: skip
        if recalcular and not previo:
            r_obs = P.calibrar(lados_qc, punto_obs)
            tar = SO.tareas(ruta_alt, ruta_boy, punto_obs, r_obs, CV.REPLICAS_POTENCIA, "potencia-observada", None)
            reps_rec = [
                x
                for parte in reanudable.ejecutar(SO.trabajo, tar, avance, procesos_, _aviso("potencia recalculada"))
                for x in parte
            ]  # noqa: E501
            pot_rec = D.Potencias(reps_rec, efectos, factores)
            recalculo["replicas"] = len(reps_rec)
        elif previo:  # cronometra una réplica aunque no haga falta recalcular
            t1 = time.time()
            SO.replica(lados_qc, P.calibrar(lados_qc, punto_obs), punto_obs, 0, "previo-potencia", None)
            seg = time.time() - t1
            tiempos["una_replica_potencia_s"] = round(seg, 1)
            recalculo["estimacion_minutos"] = seg * CV.REPLICAS_POTENCIA / max(procesos_, 1) / 60
        p_nom, p_rec, p_fin = potencias_finales(pot_nom, pot_rec)
        t = marca("SI y ρ observados y potencia recalculada", t)

        # 5. Decisión (principal y, si la regla de representatividad está activa, con su término)
        dec = decidir(comps, v3conf, "principal", pj, esc, coher, absorbe["absorbe"], p_fin)
        sens_rep = None
        if regla_activa:
            dec_r = decidir(comps, v3conf, "representatividad", pj, esc, coher, absorbe["absorbe"], p_fin)
            cambia = _resumen_decision(dec) != _resumen_decision(dec_r)
            sens_rep = {"cambia": cambia, "con_termino": _resumen_decision(dec_r)}
            if cambia:
                for fam in ("V1", "V2"):
                    dec["familias"][fam]["marcas"].append(D.M_REPRESENTATIVIDAD)
                dec["escenarios"] = D.escenarios(
                    dec["familias"]["V1"], dec["familias"]["V2"], dec["V3a"], absorbe["absorbe"]
                )
                dec["correccion"]["marcas"] = [D.M_REPRESENTATIVIDAD]
        rango = D.rango_subida(pj)
        conf = "con_maximos" if con_max else "sin_maximos"
        mde_v3a = mde[(mde["punto"] == P.NOMINAL) & (mde["prueba"] == ("V3a" if con_max else "V3a_sinmax"))]
        fila = D.fila_fuente(
            v3conf["principal"], grad, rango[conf], float(mde_v3a["efecto_minimo_80"].iloc[0]) if len(mde_v3a) else None
        )  # noqa: E501
        t = marca("decisión", t)

        # 6. Serie corregida
        serie = pd.read_parquet(ins["serie_nodo"], columns=["VHM0", "valido"])
        ts = pd.DatetimeIndex(serie.index)
        hs = serie["VHM0"].where(serie["valido"]).to_numpy(float)
        sel_asim = ts >= pd.Timestamp(C.INICIO_ASIMILADO)
        corregida = pd.DataFrame({"VHM0": hs}, index=ts)
        prueba_serie = []
        e0_diseno = dec["escenarios"]["E0"] != "sensibilidad"
        for v in dec["correccion"]["variantes"]:
            if v["tipo"] == "sin corrección":
                continue
            ca = D.corregir(ts, hs, v, sel_asim)
            corregida[f"VHM0_{v['nombre']}_asimilado"] = ca
            for f in complementos.escalones_serie(ts, hs, ca, C.CAMBIOS_SERIE[1:]):
                prueba_serie.append({"variante": v["nombre"], "alcance": "tramo asimilado (E1, E2)"} | f)
            if e0_diseno:
                ct = D.corregir(ts, hs, v, np.isfinite(hs))
                corregida[f"VHM0_{v['nombre']}_toda"] = ct
                for f in complementos.escalones_serie(ts, hs, ct):
                    prueba_serie.append({"variante": v["nombre"], "alcance": "toda la serie (E0)"} | f)
        t = marca("serie corregida", t)

        # 7. Sesgo del estimador con el σ medido (descriptivo, §20.5) y ubicación en la rejilla (§20.6, punto 4)
        punto_nom = next(p for p in P.rejilla() if p.nombre == P.NOMINAL)
        sesgo_medido = None
        if not previo:
            r_nom = P.calibrar(lados_qc, punto_nom)
            tar = SO.tareas(ruta_alt, ruta_boy, punto_nom, r_nom, C.REPLICAS_SESGO, "sesgo-sigma-medido", tabla_s)
            reps_s = [
                x
                for parte in reanudable.ejecutar(SO.trabajo, tar, avance, procesos_, _aviso("sesgo del estimador"))
                for x in parte
            ]  # noqa: E501
            P.FACTORES.clear()
            P.FACTORES.update(factores)
            sesgo_medido = {
                "replicas": len(reps_s),
                "sesgo": P.sesgo_estimador(reps_s, efectos),
                "subida_esperada": {
                    ("con_maximos" if s == "" else "sin_maximos"): P.subida_esperada(
                        reps_s,
                        p_nom[f"V3a{s}"],
                        p_nom[f"V3b{s}"],
                        suf=s,
                        v3b_validado=bool(pj["validacion_por_prueba"]["V3b"]),
                    )
                    for s in ("", "_sinmax")
                },  # fmt: skip
                "rotulo": "Descriptivo, con el σ de 1 Hz medido (§20.5, punto 3; §20.6, punto 4). No cambia el rango.",
            }
            P.FACTORES.clear()
        else:
            t1 = time.time()
            SO.replica(lados_qc, P.calibrar(lados_qc, punto_nom), punto_nom, 0, "previo-sesgo", tabla_s)
            tiempos["una_replica_sigma_medido_s"] = round(time.time() - t1, 1)
        t = marca("sesgo del estimador con el σ medido", t)

        # 8. Períodos, saltos y figuras
        per = complementos.periodos(boy, B)
        c_t0 = float(json_estricto.leer(ins["prueba_altimetros"])["umbral_c_T0"])
        sd = pd.read_parquet(ins["serie_sin_descartar"], columns=["VHM0"])["VHM0"]
        saltos = complementos.saltos_validados(sd, c_t0, so)
        sesgo = tabla_sesgo(lados, errores, con_max)
        figs = st / "figuras"
        principales = [n for n in ("T0", "A", "A1", "A2", "T0b", "T1b", "A1b", "A2b") if n in lados]
        figuras.qq({n: (lados[n].L.t, lados[n].L.m, lados[n].o) for n in principales}, figs / "qq_por_lado.png")
        figuras.cocientes(sesgo[sesgo["lado"].isin(principales)], figs / "cocientes_por_cuantil.png")
        figuras.mapa(so, figs / "mapa_colocaciones.png")
        figuras.distancia(grad, figs / "cociente_distancia_costa.png")
        t = marca("períodos, saltos y figuras", t)

        # 9. Salidas
        pd.DataFrame(C.OBSERVACIONES).to_csv(st / "observaciones.csv", index=False, lineterminator="\n")
        cols = [c for c in so.columns if c != "marca_detector"] + ["marca_detector"]
        so[cols].assign(fuente="altimetría").pipe(
            lambda x: pd.concat([x, boy.assign(fuente="boya")], ignore_index=True)
        ).to_parquet(st / "colocaciones.parquet")
        qc = pd.DataFrame([{"fuente": "altimetría"} | q for q in qc_alt] + qc_boy)
        _csv(qc, st / "qc_observaciones.csv")
        _csv(sesgo, st / "sesgo_por_cuantil.csv")
        _csv(per, st / "periodos.csv")
        _csv(saltos, st / "saltos_validados.csv")
        _csv(sig, st / "sigma_1hz.csv")
        pruebas = {
            "comparaciones": comps,
            "familias": dec["familias"],
            "V3": {
                "configuracion": conf,
                "elegida": v3conf["principal"],
                "otra_configuracion": v3_otra,
                "calibracion_x2": v3_cal2,
                "representatividad": v3conf["representatividad"] if regla_activa else None,
                "categorias": {"V3a": dec["V3a"], "V3b": dec["V3b"]},
                "V3b_puede_descartarse": bool(pj["validacion_por_prueba"]["V3b"]),
            },
            "coherencia_T0": coher,
            "absorcion_V2": absorbe,
            "potencia": {"nominal": p_nom, "recalculada": p_rec, "final": p_fin, "recalculo": recalculo},
            "escalon_modelo_P90": {c: _s_mod(esc, c).get("P90") for c in pj["atribuye"]},
            "atribuye": pj["atribuye"],
            "traslado_p90": pj["traslado_p90"],
        }
        json_estricto.escribir(st / "pruebas_validacion.json", pruebas)
        json_estricto.escribir(
            st / "representatividad.json",
            {
                "sigma_1hz": sig.to_dict(orient="records"),
                "correlacion_diferencias_sucesivas": ruido.correlacion_rezago1(puntos),
                "funcion_de_estructura": estr,
                "regla_20_3_activa": regla_activa,
                "configuracion": conf,
                "sensibilidad_representatividad": sens_rep,
                "gradiente": grad,
                "si_observado": si_obs,
                "rho_observado": rho_obs,
                "si_nominal": nominal,
                "traslado_p90_escalon_modelo": pj["traslado_p90"],
            },
        )
        json_estricto.escribir(st / "correccion.json", dec["correccion"] | {"prueba_serie_corregida": prueba_serie})
        if corregida.shape[1] > 1:
            corregida.to_parquet(st / "serie_corregida.parquet")
        agranda = [f for f in prueba_serie if f["agranda"]]
        hallazgos = list(pj["hallazgos"]) + [D.hallazgo_20_6(rango)]
        if agranda:
            hallazgos.append(f"La serie corregida agranda el escalón en {len(agranda)} estadísticos (§6.2).")
        resumen = {
            "previo_sintetico": previo,
            "valores_observados_leidos": not previo,
            "g4": g4,
            "geometria_igual_a_la_etapa_2": geometria_igual,
            "escenarios": dec["escenarios"],
            "V1": dec["familias"]["V1"]["categoria"],
            "V2": dec["familias"]["V2"]["categoria"],
            "V3a": dec["V3a"],
            "V3b": dec["V3b"],
            "correccion": {
                "variantes": dec["correccion"]["variantes"],
                "mas_desfavorable": dec["correccion"]["mas_desfavorable"],
            },
            "configuracion_maximos": conf,
            "fila_fuente": fila,
            "rango_subida_esperada_20_6": rango,
            "ubicacion_descriptiva_20_6": {
                "si_observado": si_obs,
                "rho_observado": rho_obs,
                "sigma_1hz_medido": sig.to_dict(orient="records"),
                "subida_con_sigma_medido": (sesgo_medido or {}).get("subida_esperada"),
                "rotulo": "Rotulada y descriptiva (§20.6, punto 4): no cambia el rango ni el hallazgo.",
            },
            "sesgo_estimador_sigma_medido": sesgo_medido,
            "hallazgos": hallazgos,
            "alcance": {
                "altimetria": "La cola del producto en la región (≤ 200 km, ≥ 50 km de la costa), no el nodo.",
                "boyas": "El producto en la cuenca, a unos 1050–1100 km; su sesgo absoluto es descriptivo.",
                "traslado_al_nodo": "Supuesto declarado (ADR-0004); gradiente: " + str(grad.get("estado")),
                "periodo_cubierto": "El lado asimilado mide 2011–2023: llevar su sesgo a 1993–2011 es un supuesto.",
            },
            "que_atribuye_cada_fuente": {
                "altimetria": "GEOSAT (1985–1988) frente a HY-2 (2011–2023): cambio entre épocas, no en 1993-02.",
                "boya_32302": "1986–1990 frente a 1993–1995: localiza el cambio entre 1991 y 1993-02 (cuenca).",
            },
            "nota_32012": C.NOTA_32012,
            "entregas": {
                "F8": "V1 y V2 con sus marcas; descriptivos de §19 (escalón de C2 en la cuenca y cola de C1 en 32302).",
                "F9": "Sesgo de T0 como dato (P7); saltos_validados.csv; sensibilidad sin corregir.",
                "F10": "periodos.csv; la serie corregida no cumple la coherencia ni el peralte de la F3 (§6.2).",
                "F12": "Fila de la fuente (§6.3) y rango de la subida (§20.6), separada del sesgo de profundidad.",
            },
            "tiempos_s": tiempos,
        }
        json_estricto.escribir(st / "resumen_f05.json", resumen)
        for f in (ruta_alt, ruta_boy):
            f.unlink()
        motivos = []
        if not geometria_igual:
            motivos.append("Las superobservaciones antes del filtro de Hs no coinciden con la geometría de la etapa 2.")
        fuera = dentro_de_ic(sesgo, comps)
        if fuera:
            motivos.append(f"G6: estimados fuera de su IC: {fuera[:5]}")
        criterios = {"g4": True, "geometria_igual": geometria_igual, "B": B, "previo": previo}
        publicada = etapa.publicar(
            st,
            salida,
            etapa.Gate(not motivos, criterios, motivos),
            raices,
            ins,
            raiz,
            semillas.semilla("f05", "validacion"),
        )
        if not previo:
            reanudable.limpiar(avance)
        return publicada


def _aviso(etiqueta: str):
    return lambda t: print(f"  {etiqueta}: {t}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validacion", description="Etapa 3 de la F5: valores observados y decisión.")
    ap.add_argument("--datos")
    ap.add_argument("--nombre", default=NOMBRE)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--previo", action="store_true", help="Valores sintéticos, en interim/, para cronometrar.")
    a = ap.parse_args(argv)
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        salida = validacion_etapa(a.datos, a.nombre, a.procesos, a.previo)
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    print(f"Publicado en {salida}")
    return 0
