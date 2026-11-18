# ruff: noqa: E501  (textos largos en español; se formatean al leer el JSON)
"""Notas descriptivas **posteriores a los datos** (contrato §20.8, aceptado por el usuario el 2026-11-18).

No cambian ninguna categoría, escenario, corrección ni cifra publicada: añaden texto a ``resumen_f05.json`` y a las
entregas. Las cifras salen de los resultados de la corrida, no se escriben a mano.
"""

from __future__ import annotations

import math

ROTULO = "Posterior a los datos (§20.8): nota descriptiva; no cambia ninguna categoría ni decisión."
UMBRAL_RELEVANCIA = math.log(1.02)  # efecto relevante del 2 % (ADR-0006)


def discrepancia_v1(comps: dict, potencias: dict, s_mod_p90: dict, familia_v1: dict) -> dict | None:
    """Si una fuente de V1 rechaza y la otra no, por qué discrepan: potencia, estimado frente al escalón del modelo y
    robustez a la calibración × 2."""
    rech = set(familia_v1.get("fuentes_que_rechazan", []))
    if len(rech) != 1:
        return None
    fuentes = {"a": ("V1a", "altimetría", "V1a"), "b": ("V1b", "boya 32302", "V1b_lado")}
    datos = {}
    for k, (comp, nombre, pot) in fuentes.items():
        c = comps.get(comp, {})
        if "principal" not in c:
            return None
        datos[k] = {
            "fuente": nombre,
            "rechaza": c["principal"]["rechaza"],
            "p": c["principal"]["p"],
            "p_calibracion_x2": c["calibracion_x2"]["p"],
            "rechaza_calibracion_x2": c["calibracion_x2"]["rechaza"],
            "estimado_P90": c["principal"]["estimado"].get("P90"),
            "escalon_modelo_P90": s_mod_p90.get(comp),
            "potencia_frente_al_efecto": potencias.get(pot),
        }
    r = datos["a" if "a" in rech else "b"]
    o = datos["b" if "a" in rech else "a"]
    cociente = (
        r["estimado_P90"] / r["escalon_modelo_P90"]
        if r["estimado_P90"] is not None and r["escalon_modelo_P90"]
        else None
    )
    texto = (
        f"Las fuentes de V1 discrepan. Rechaza la {r['fuente']} (p = {r['p']:.4f}) con una potencia de "
        f"{r['potencia_frente_al_efecto']:.2f} frente a su efecto relevante; su Δ1 en el P90 ({r['estimado_P90']:+.3f}) "
        f"es {cociente:.1f} veces el escalón del modelo en sus colocaciones ({r['escalon_modelo_P90']:.3f}), y "
        f"{'se sostiene' if r['rechaza_calibracion_x2'] else 'no se sostiene'} con la calibración × 2 "
        f"(p = {r['p_calibracion_x2']:.2f}). Con tan poca potencia, un rechazo tiende a exagerar el efecto. La "
        f"{o['fuente']}, con potencia {o['potencia_frente_al_efecto']:.2f} (con el término por lado si es la boya), no "
        f"rechaza (p = {o['p']:.2f}); su Δ1 en el P90 es {o['estimado_P90']:+.3f} frente a un escalón del modelo de "
        f"{o['escalon_modelo_P90']:.3f}. "
    )
    if r["fuente"] == "altimetría":
        texto += (
            "La altimetría compara épocas (GEOSAT 1985–1988 frente a HY-2 2011–2023) y recoge todo lo que cambió entre "
            "misiones; la boya es la única fuente que sitúa el cambio entre 1991 y 1993-02. La lectura para la F8 no es "
            "«la altimetría atribuye C1 al producto»: la evidencia con más potencia es compatible con un cambio real "
            "en la cuenca. «V1 establecido» es lo que manda la regla de §4.1 (basta que rechace una fuente) y no cambia."
        )
    return {"rotulo": ROTULO, "texto": texto, "fuentes": datos}


def hallazgo_serie_corregida(prueba_serie: list[dict]) -> dict | None:
    """§6.2 con la opción (a): la variante se mantiene con su marca. Aumento máximo por alcance y cambio, frente al
    2 % de la ADR-0006, y las marcas para la F9."""
    agr = [f for f in prueba_serie if f["agranda"]]
    if not agr:
        return None
    por = {}
    for f in agr:
        clave = (f["variante"], f["alcance"], f["cambio"])
        aumento = abs(f["S_corregida"]) - abs(f["S_sin_corregir"])
        if clave not in por or aumento > por[clave]["aumento"]:
            por[clave] = {"estadistico": f["estadistico"], "aumento": aumento}
    marcas = [
        {"variante": v, "alcance": a, "cambio": c, "estadistico": x["estadistico"], "aumento": x["aumento"],
         "marca": f"agranda el escalón de {c} en {x['aumento']:.3f} en {x['estadistico']}"}
        for (v, a, c), x in sorted(por.items())
    ]  # fmt: skip
    maximo = max(m["aumento"] for m in marcas)
    return {
        "rotulo": ROTULO,
        "texto": (
            f"La serie corregida agranda el escalón en {len(agr)} estadísticos (§6.2). Opción (a), aceptada por el "
            "usuario: la variante por cuantiles se mantiene con su marca, porque la tabla de §6.2 la manda con V3b "
            "establecido, y cambiarla por un factor constante después de ver los datos se saltaría esa regla. Ningún "
            f"aumento llega al efecto relevante del 2 % de la ADR-0006 (log 1.02 = {UMBRAL_RELEVANCIA:.4f}): el mayor es "
            f"{maximo:.3f}. La tolerancia de 0.001 del §20.7 era numérica. Con una función por cuantiles aplicada a "
            "tramos con distribuciones distintas, la promesa de §6.2 de no fabricar un escalón en 1993 solo vale para "
            "un factor constante."
        ),
        "aumento_maximo": maximo,
        "llega_al_efecto_relevante": bool(maximo >= UMBRAL_RELEVANCIA),
        "marcas_para_la_F9": marcas,
    }


def lectura_linea_448(v1_categoria: str, e0: str, e0_corregida: bool) -> dict | None:
    """§6.2: «salvo que V1 salga establecido, y entonces E0 ya no es escenario de diseño». Si V1 es establecido pero E0
    sigue activo (por sus marcas), E0 sigue siendo escenario de diseño y recibe su variante corregida."""
    if v1_categoria != "establecido" or e0 == "sensibilidad" or not e0_corregida:
        return None
    return {
        "rotulo": ROTULO,
        "texto": (
            "V1 salió establecido, pero E0 sigue activo por las marcas (§6.1). La excepción de §6.2 («salvo que V1 salga "
            "establecido») se apoya en que entonces E0 deja de ser escenario de diseño; aquí no deja de serlo, así que la "
            "variante corregida se aplica a toda la serie para E0. Es una lectura de la regla, para el revisor de ciencia."
        ),
    }
