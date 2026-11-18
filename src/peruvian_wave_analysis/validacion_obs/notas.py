# ruff: noqa: E501  (textos largos en español; se formatean al leer el JSON)
"""Notas descriptivas **posteriores a los datos** (contrato §20.8 y §20.9, aceptados por el usuario el 2026-11-18).

No cambian ninguna categoría, escenario, corrección ni cifra de decisión: añaden texto y cifras descriptivas a
``resumen_f05.json`` y a las entregas. Las cifras salen de los resultados de la corrida, no se escriben a mano, y
un valor ausente o no finito se escribe «n/d» (§20.9, C-04).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ROTULO = "Posterior a los datos (§20.8–§20.9): nota descriptiva; no cambia ninguna categoría ni decisión."
UMBRAL_RELEVANCIA = math.log(1.02)  # efecto relevante del 2 % (ADR-0006)
ESTS_V1 = ("P50", "P90", "P99", "max")
NOMBRE_EST = {"P50": "P50", "P90": "P90", "P99": "P99", "max": "máximos"}


def _ok(x) -> bool:
    return x is not None and isinstance(x, int | float) and math.isfinite(x)


def f(x, fmt: str = ".3f") -> str:
    """Formato seguro: «n/d» si el valor falta o no es finito."""
    return format(x, fmt) if _ok(x) else "n/d"


def _fraccion(d, s):
    return d / s if _ok(d) and _ok(s) and s != 0 else None


# --- V1 (§20.8.1, reescrita en §20.9, S-28) -----------------------------------------------------------------


def discrepancia_v1(comps: dict, potencias: dict, s_mod: dict, familia_v1: dict) -> dict | None:
    """Si una fuente de V1 rechaza y la otra no: los cuatro estadísticos de cada fuente frente al escalón del modelo
    en sus colocaciones, la potencia (la boya con y sin el término por lado) y la robustez. ``s_mod``: por comparación,
    dict estadístico → S_mod. Lectura de evidencia mixta: ni «clima» ni «artefacto»."""
    rech = set(familia_v1.get("fuentes_que_rechazan", []))
    if len(rech) != 1:
        return None
    fuentes = {"a": ("V1a", "altimetría"), "b": ("V1b", "boya 32302")}
    datos = {}
    for k, (comp, nombre) in fuentes.items():
        c = comps.get(comp, {})
        if "principal" not in c:
            return None
        pr = c["principal"]
        sm = s_mod.get(comp, {}) or {}
        datos[k] = {
            "fuente": nombre,
            "rechaza": pr["rechaza"],
            "p": pr.get("p"),
            "decide": pr.get("decide"),
            "p_calibracion_x2": c["calibracion_x2"].get("p"),
            "rechaza_calibracion_x2": c["calibracion_x2"]["rechaza"],
            "por_estadistico": {
                e: {
                    "delta": pr["estimado"].get(e),
                    "ic95": (pr.get("ic95") or {}).get(e),
                    "escalon_modelo": sm.get(e),
                    "fraccion_del_escalon": _fraccion(pr["estimado"].get(e), sm.get(e)),
                }
                for e in ESTS_V1
            },
        }
    datos["a"]["potencia"] = potencias.get("V1a")
    datos["b"]["potencia_con_termino_por_lado"] = potencias.get("V1b_lado")
    datos["b"]["potencia_sin_termino_por_lado"] = potencias.get("V1b")
    sens = comps.get("V1b_hasta_1992", {}).get("principal", {})
    datos["b"]["sensibilidad_1991_1992"] = {"p": sens.get("p"), "rechaza": sens.get("rechaza")}

    def tramo(k, ests):
        fr = [datos[k]["por_estadistico"][e]["fraccion_del_escalon"] for e in ests]
        fr = [x for x in fr if _ok(x)]
        if not fr:
            return "n/d"
        return f"{min(fr):.0%}–{max(fr):.0%}" if len(fr) > 1 else f"{fr[0]:.0%}"

    def lista(k):
        return "; ".join(
            f"{NOMBRE_EST[e]} {f(v['delta'], '+.3f')} frente a {f(v['escalon_modelo'])}"
            for e, v in datos[k]["por_estadistico"].items()
        )

    a, b = datos["a"], datos["b"]
    texto = (
        "La evidencia de V1 es mixta. «V1 establecido» es lo que manda §4.1 (basta que rechace una fuente) y no cambia. "
        f"Altimetría (Δ1 frente al escalón del modelo en sus colocaciones): {lista('a')}; rechaza "
        f"(p = {f(a['p'], '.4f')}, decide el {NOMBRE_EST.get(a['decide'], 'n/d')}) con una potencia de {f(a['potencia'], '.2f')} frente a "
        f"su efecto relevante, y {'se sostiene' if a['rechaza_calibracion_x2'] else 'no se sostiene'} con la calibración × 2 "
        f"(p = {f(a['p_calibracion_x2'], '.2f')}); un rechazo con tan poca potencia tiende a exagerar el efecto. "
        f"Boya 32302: {lista('b')}; no rechaza (p = {f(b['p'], '.2f')}), con potencia {f(b['potencia_sin_termino_por_lado'], '.2f')} "
        f"sin el término por lado y {f(b['potencia_con_termino_por_lado'], '.2f')} con él (las dos suponen un desplazamiento "
        "uniforme; frente al perfil de escalones del modelo en la boya, la potencia es menor). Con 1991–1992 en T0b, "
        f"p = {f(b['sensibilidad_1991_1992']['p'], '.3f')}. "
        f"En el cuerpo (P50 y P90), la boya ve {tramo('b', ('P50', 'P90'))} del escalón del modelo; en la cola (P99 y "
        f"máximos), {tramo('b', ('P99', 'max'))}. La altimetría ve {tramo('a', ESTS_V1)} en los cuatro estadísticos. "
        "Las dos fuentes comparan ventanas de 3 a 5 años (la altimetría, épocas: GEOSAT 1985–1988 frente a HY-2 "
        "2011–2023), con la variabilidad interanual dentro. La F8 no debe leer esto ni como «clima» ni como «artefacto "
        "del producto»: en el cuerpo la boya no ve el escalón y en la cola ve buena parte. La deriva dentro de T1 sigue "
        "como hipótesis (§4.1)."
    )
    return {"rotulo": ROTULO, "texto": texto, "fuentes": datos}


# --- Serie corregida (§20.8.3, precisado en §20.9, S-36) ----------------------------------------------------------


def hallazgo_serie_corregida(prueba_serie: list[dict]) -> dict | None:
    """La variante se mantiene con su marca (opción a). Aumento máximo por alcance y cambio, frente al 2 % (ADR-0006),
    umbral elegido después de ver los datos (S-36), y las marcas para la F9."""
    agr = [x for x in prueba_serie if x["agranda"]]
    if not agr:
        return None
    por = {}
    for x in agr:
        clave = (x["variante"], x["alcance"], x["cambio"])
        aumento = abs(x["S_corregida"]) - abs(x["S_sin_corregir"])
        if clave not in por or aumento > por[clave]["aumento"]:
            por[clave] = {"estadistico": x["estadistico"], "aumento": aumento}
    marcas = [
        {"variante": v, "alcance": a, "cambio": c, "estadistico": d["estadistico"], "aumento": d["aumento"],
         "marca": f"{a}: agranda el escalón de {c} en {d['aumento']:.3f} en {d['estadistico']}"}
        for (v, a, c), d in sorted(por.items())
    ]  # fmt: skip
    maximo = max(m["aumento"] for m in marcas)
    return {
        "rotulo": ROTULO,
        "texto": (
            f"La serie corregida agranda el escalón en {len(agr)} estadísticos (§6.2). Opción (a), aceptada por el "
            "usuario: la variante por cuantiles se mantiene con su marca, porque la tabla de §6.2 la manda con V3b "
            "establecido, y cambiarla por un factor constante después de ver los datos se saltaría esa regla. "
            f"El mayor aumento es {maximo:.3f}, el {maximo / UMBRAL_RELEVANCIA:.0%} del efecto relevante del 2 % de la "
            f"ADR-0006 (log 1.02 = {UMBRAL_RELEVANCIA:.4f}). Ese umbral del 2 % se propuso después de ver el aumento, en "
            "una revisión externa: es un criterio posterior a los datos, no una regla fijada antes (§20.9, S-36). La "
            "tolerancia de 0.001 del §20.7 era numérica. Con una función por cuantiles aplicada a tramos con "
            "distribuciones distintas, la promesa de §6.2 de no fabricar un escalón en 1993 solo vale para un factor "
            "constante."
        ),
        "aumento_maximo": maximo,
        "fraccion_del_efecto_relevante": maximo / UMBRAL_RELEVANCIA,
        "umbral_elegido_despues_de_los_datos": True,
        "marcas_para_la_F9": marcas,
    }


def lectura_linea_448(v1_categoria: str, e0: str, e0_corregida: bool) -> dict | None:
    """§6.2: «salvo que V1 salga establecido, y entonces E0 ya no es escenario de diseño»."""
    if v1_categoria != "establecido" or e0 == "sensibilidad" or not e0_corregida:
        return None
    return {
        "rotulo": ROTULO,
        "texto": (
            "V1 salió establecido, pero E0 sigue activo por las marcas (§6.1). La excepción de §6.2 («salvo que V1 salga "
            "establecido») se apoya en que entonces E0 deja de ser escenario de diseño; aquí no deja de serlo, así que la "
            "variante corregida se aplica a toda la serie para E0. Esta lectura ya estaba en el código antes de leer "
            "valores (commit 7d4a877: la variante de E0 se aplica si E0 no pasa a sensibilidad); la nota solo la hace "
            "explícita para el revisor de ciencia."
        ),
    }


def efecto_cuerpo_t0(t, hs, corregida, inicio_t0="1980-01-01", fin_t0="1992-12-31") -> dict | None:
    """§20.9 (S-32): cuánto mueve la variante de E0 el cuerpo (P50), el P90 y el P99 de T0, para la F9."""
    t = pd.DatetimeIndex(t)
    s = (t >= pd.Timestamp(inicio_t0)) & (t < pd.Timestamp(fin_t0) + pd.Timedelta(days=1))
    x, y = np.asarray(hs, float)[s], np.asarray(corregida, float)[s]
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    q = {
        e: float(np.quantile(y[ok], p) / np.quantile(x[ok], p)) for e, p in (("P50", 0.5), ("P90", 0.9), ("P99", 0.99))
    }
    return {
        "rotulo": ROTULO,
        "cociente_corregida_sin_corregir_en_T0": q,
        "texto": (
            f"En la variante de E0, T0 se corrige con la misma función: su P50 queda multiplicado por {f(q['P50'])}, "
            f"su P90 por {f(q['P90'])} y su P99 por {f(q['P99'])}. GEOSAT (y ERS-1 y TOPEX, sin marca de incoherencia) "
            "indican que el producto ya subestima en T0; bajar el cuerpo de T0 va en sentido contrario. En la cola el "
            "efecto es al alza, así que no baja el diseño."
        ),
    }


# --- §20.9: S-29, S-30, S-31, S-33 y C-10 ---------------------------------------------------------------------


def forma_por_mision(v3b_por_mision: dict[str, dict], v3b_establecido: bool) -> dict:
    """S-29: la forma de V3b (B(alto) − B(P50)) por misión del lado A. Marca «forma no coherente entre misiones» si V3b
    salió establecido y la forma solo excluye el 0 en una de las misiones. Descriptiva: no cambia la variante."""
    filas, excluye = {}, {}
    for m, r in v3b_por_mision.items():
        clave = next(iter(r["estimado"]), None)
        est = r["estimado"].get(clave) if clave else None
        ic_ = (r.get("ic95") or {}).get(clave) if clave else None
        excluye[m] = bool(ic_ and _ok(ic_[0]) and _ok(ic_[1]) and (ic_[0] > 0 or ic_[1] < 0))
        filas[m] = {"estadistico": clave, "estimado": est, "ic95": ic_, "excluye_cero": excluye[m]}
    marca = bool(v3b_establecido and len(excluye) == 2 and sum(excluye.values()) == 1)
    lleva = [m for m, v in excluye.items() if v]

    def ic_txt(v):
        lo, hi = (v["ic95"] or [None, None])[:2]
        return f"IC {f(lo, '+.3f')} a {f(hi, '+.3f')}"

    texto = "La forma de V3b por misión: " + "; ".join(
        f"{m} {f(v['estimado'], '+.3f')} ({ic_txt(v)})" for m, v in filas.items()
    )
    if marca:
        texto += (
            f". Solo {lleva[0]} la separa de 0: el factor de cola de la variante por cuantiles lo aporta una sola misión, "
            "lo que apunta a calibración o procesado de esa misión más que al producto. No cambia la variante (§6.2 es "
            "anterior a los datos y conservadora); la F9 y la F12 lo deben tener en cuenta."
        )
    else:
        texto += ". Sin marca."
    return {
        "rotulo": ROTULO,
        "por_mision": filas,
        "marca": "forma no coherente entre misiones" if marca else None,
        "texto": texto,
    }


def periodos_para_f10(per: pd.DataFrame) -> dict:
    """S-30: lo que recibe la F10 de §3.6, con el rótulo correcto (APD es período medio, Tm02) y el DPD descriptivo."""
    cols = [
        "boya",
        "umbral_hs",
        "lectura",
        "comparacion",
        "lado",
        "mediana_modelo_menos_boya_s",
        "ic95_lo",
        "ic95_hi",
        "clase",
    ]
    filas = [{k: r.get(k) for k in cols} for r in per.to_dict(orient="records")]
    return {
        "rotulo": ROTULO,
        "filas": filas,
        "texto": (
            "La lectura principal (APD frente a VTM02) mide el período medio (Tm02), no Tp: su clase se refiere a Tm02. "
            "El sesgo absoluto de Tm02 (modelo − boya) está en las filas de cada lado; si supera los 0.3 s de la ADR-0006, "
            "parte puede venir de la definición de APD de NDBC (sin verificar). La única evidencia sobre Tp es DPD frente "
            "a VTPK (filas «descriptivo»; resolución de banda no disponible). La F10 no debe retirar sus variantes de Tp "
            "con la lectura de APD."
        ),
    }


def equivariancia(variantes: list[dict]) -> dict | None:
    """S-31: la variante por cuantiles no es un factor; la F7 y la F9 deben volver a elegir u* y ajustar."""
    if not any(v["tipo"] == "corrección por cuantiles" for v in variantes):
        return None
    return {
        "rotulo": ROTULO,
        "texto": (
            "La variante por cuantiles no es un factor constante: estira el tramo entre el P90 y el P99 más que la cola "
            "por encima. La equivariancia de §6.2 no aplica a esta variante: la F7 debe volver a elegir el umbral u* y la "
            "F9 volver a ajustar sobre la serie corregida, no escalar el nivel de retorno. Debe constar en el contrato "
            "de la F7."
        ),
    }


def limitaciones_sigma(sig: pd.DataFrame, correlacion: dict) -> dict:
    """S-33 y C-10: ruido correlacionado (σ̂ subestimado más de lo que cubre × 2) y σ cuantizada por la resolución."""
    filas = []
    for m, rho in sorted(correlacion.items()):
        if _ok(rho) and rho > -0.4:
            phi = max(0.0, min(0.95, 1 + 2 * rho))  # diferencias de un AR(1): corr = −(1 − φ)/2
            fs = 1 / math.sqrt(1 - phi)
            n = 3  # superobservación mínima
            infl = math.sqrt(1 + 2 * sum((1 - k / n) * phi**k for k in range(1, n)))
            filas.append(
                {"mision": m, "correlacion_diferencias": rho, "phi_ar1": phi, "factor_sigma": fs,
                 "factor_sigma_superobs_3_puntos": fs * infl, "cubre_sensibilidad_x2": bool(fs * infl <= 2)}
            )  # fmt: skip
    cuant = sorted(m for m, g in sig.groupby("mision") if g["sigma"].round(9).nunique() < len(g))
    return {
        "rotulo": ROTULO,
        "ruido_correlacionado": filas,
        "sigma_cuantizada": cuant,
        "texto": (
            "En las misiones con correlación de diferencias sucesivas lejos de −0.5, el ruido de 1 Hz está correlacionado: "
            "con un AR(1), σ̂ subestima σ, y el σ de una superobservación de 3 puntos queda corto en el factor indicado; "
            "si supera 2, no lo cubre la sensibilidad σ × 2 (en V1a, efecto estimado de unos +0.01 sobre Δ1; no toca el diseño). Además, la σ de varias misiones "
            "coincide entre niveles: la MAD se cuantiza por la resolución de 0.01 m de la Hs de 1 Hz (rejilla de unos "
            "0.0105 m en σ̂)."
        ),
    }
