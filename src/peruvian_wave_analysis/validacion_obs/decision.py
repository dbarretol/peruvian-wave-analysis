"""Reglas de decisión de la F5 (contrato v6, §4.1, §6.1, §6.2, §6.3, §19 y §20.6). Sin cálculo estadístico.

- **Categorías (ADR-0005):** establecido (rechaza), descartado (no rechaza y potencia ≥ 80 % frente al efecto
  relevante, con los dos signos) o no concluyente.
- **Combinación de fuentes (P12, §4.1):** Bonferroni, α = 0.025 por fuente; las fuentes que no atribuyen (regla del ⅓)
  no cuentan.
- **Escenarios (§6.1):** una marca nunca retira un escenario; solo impide que se retire.
- **Corrección (§6.2) y fila de la fuente (§6.3).**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..validacion import config as CV
from ..validacion import potencia as P
from ..validacion import simulacion as S
from . import config as C

ESTABLECIDO, DESCARTADO, NO_CONCLUYENTE = "establecido", "descartado", "no concluyente"
SIN_ESCALON = "sin escalón del modelo que atribuir"

M_INCOHERENCIA = "incoherencia entre misiones de T0"
M_CONTRADICCION = "contradicción entre fuentes"
M_SIGNO = "signo opuesto al escalón del modelo"
M_TRASLADO = "traslado dudoso"
M_REPRESENTATIVIDAD = "sensible a la representatividad"
M_SISTEMA = "no robusto a un cambio de sistema de la boya"
M_CALIBRACION = "no robusto a la calibración × 2"
M_CUENCA = "establecido en la cuenca"
M_EPOCA = "cambio de época sin escalón en 1993"


# --- Potencia ------------------------------------------------------------------------------------------------


def replicas_desde_tabla(filas: pd.DataFrame) -> list[dict]:
    """Reconstruye las réplicas de ``replicas_potencia.parquet`` (Δ̂, EE y críticos) como en la etapa 2."""
    out: dict[int, dict] = {}
    cols_est = sorted((c for c in filas.columns if c.startswith("est_")), key=lambda c: int(c[4:]))
    cols_ee = [c.replace("est_", "ee_") for c in cols_est]
    crit = sorted(c for c in filas.columns if c.startswith("critico_"))
    for d in filas.to_dict(orient="records"):  # no itertuples: renombra las columnas con punto (critico_0.025)
        # Las columnas sobrantes de las pruebas con menos estadísticos son NaN, y ``rechaza`` las ignora igual que a
        # los no estimables: la potencia sale idéntica a la de la etapa 2.
        est = np.array([d[c] for c in cols_est], float)
        ee = np.array([d[c] for c in cols_ee], float)
        pr = S.Prueba(est, ee, {float(c.removeprefix("critico_")): d[c] for c in crit}, d["p"])
        out.setdefault(int(d["replica"]), {})[d["prueba"]] = pr
    return [out[i] for i in sorted(out)]


class Potencias:
    """Potencia frente al efecto relevante (mínimo de los dos signos) a partir de las réplicas de b = 0."""

    def __init__(self, reps: list[dict], efectos: dict[str, float], factores: dict[str, float]):
        self.reps, self.efectos, self.factores = reps, efectos, dict(factores)

    def _con_factores(self, f):
        antes = dict(P.FACTORES)
        P.FACTORES.clear()
        P.FACTORES.update(self.factores)
        try:
            return f()
        finally:
            P.FACTORES.clear()
            P.FACTORES.update(antes)

    def individual(self, prueba: str, alfa: float) -> float:
        ef = self.efectos[prueba]
        return self._con_factores(
            lambda: min(P.potencia(self.reps, prueba, ef, alfa), P.potencia(self.reps, prueba, -ef, alfa))
        )

    def conjunta_v1(self, con_lado: bool) -> float:
        """Al menos una fuente rechaza con su efecto (α = 0.025 cada una); la boya con o sin el término por lado."""
        b = "V1b_lado" if con_lado else "V1b"

        def f():
            vals = []
            for s in (1, -1):
                ea, eb = s * self.efectos["V1a"], s * self.efectos[b]
                fa, fb = P.FACTORES.get("V1a", 1.0), P.FACTORES.get(b, 1.0)
                vals.append(
                    float(
                        np.mean(
                            [
                                S.rechaza(rp["V1a"], ea, CV.ALFA_FUENTE, fa) or S.rechaza(rp[b], eb, CV.ALFA_FUENTE, fb)
                                for rp in self.reps
                            ]
                        )
                    )
                )
            return min(vals)

        return self._con_factores(f)


# --- Categorías y familias -----------------------------------------------------------------------------------


def categoria(rechaza: bool, potencia: float, puede_descartar: bool = True) -> str:
    if rechaza:
        return ESTABLECIDO
    return DESCARTADO if (puede_descartar and np.isfinite(potencia) and potencia >= 0.8) else NO_CONCLUYENTE


def familia(fuentes: dict[str, dict], potencia_conjunta: float) -> dict:
    """§4.1. ``fuentes``: por fuente («a», «b»), con ``atribuye``, ``rechaza``, ``signo`` y ``descartada``
    (no rechaza y su potencia individual ≥ 80 %; en la boya, con el término por lado)."""
    activas = {k: f for k, f in fuentes.items() if f["atribuye"]}
    marcas: list[str] = []
    if not activas:
        return {"categoria": NO_CONCLUYENTE, "marcas": marcas, "fuentes_activas": [], "motivo": SIN_ESCALON}
    rech = {k: f for k, f in activas.items() if f["rechaza"]}
    desc = {k: f for k, f in activas.items() if f["descartada"] and not f["rechaza"]}
    if rech:
        if len({f["signo"] for f in rech.values()}) > 1:
            marcas.append(M_CONTRADICCION)
            cat, motivo = NO_CONCLUYENTE, "dos fuentes rechazan con signos opuestos"
        elif "a" in rech and "b" in desc:
            marcas.append(M_EPOCA)
            cat, motivo = NO_CONCLUYENTE, "la altimetría establece y la boya descarta (v5, punto 4)"
        elif desc:
            cat, motivo = NO_CONCLUYENTE, "una fuente rechaza y otra sale descartada"
        else:
            cat, motivo = ESTABLECIDO, "rechaza " + " y ".join(sorted(rech))
    else:
        cat = DESCARTADO if (np.isfinite(potencia_conjunta) and potencia_conjunta >= 0.8) else NO_CONCLUYENTE
        motivo = "ninguna fuente rechaza"
    return {
        "categoria": cat,
        "marcas": marcas,
        "fuentes_activas": sorted(activas),
        "fuentes_que_rechazan": sorted(rech),
        "potencia_conjunta": potencia_conjunta,
        "motivo": motivo,
    }


def marcas_fuente(
    res: dict, res_cal2: dict, res_lado: dict | None, s_mod: dict[str, float], traslado: str, incoherente: bool = False
) -> list[str]:
    """Marcas de una fuente que rechaza (§4.1, §3.5 y §3.7). Una fuente que no rechaza solo lleva la de traslado."""
    m = []
    if traslado == "dudoso":
        m.append(M_TRASLADO)
    if incoherente:
        m.append(M_INCOHERENCIA)
    if res["rechaza"]:
        e = res["decide"]
        if e in s_mod and np.isfinite(s_mod[e]) and s_mod[e] != 0 and np.sign(s_mod[e]) != res["signo"]:
            m.append(M_SIGNO)
        if not res_cal2["rechaza"]:
            m.append(M_CALIBRACION)
        if res_lado is not None and not res_lado["rechaza"]:
            m.append(M_SISTEMA)
    return m


# --- Escenarios (§6.1) -----------------------------------------------------------------------------------------


def escenarios(v1: dict, v2: dict, v3a_cat: str, absorbe_v2: bool | None) -> dict:
    """E0, E1 y E2 según V1 y V2. ``v1``/``v2``: familia con ``marcas_fuente`` por fuente añadido."""
    out = {"E0": "activo", "E1": "activo", "E2": "activo", "base": None, "marcas": {}, "motivos": []}
    # V1 → E0 y E1
    mf_a = v1["marcas_fuente"].get("a", [])
    a_rechaza = "a" in v1.get("fuentes_que_rechazan", [])
    if v1["categoria"] == ESTABLECIDO and a_rechaza:
        if not mf_a and not v1["marcas"]:
            out.update(E0="sensibilidad", base="E1")
            out["motivos"].append("V1 establecido en la altimetría, sin marcas: E0 pasa a sensibilidad y E1 a base.")
        else:
            out["motivos"].append(f"V1 establecido en la altimetría con marcas {mf_a + v1['marcas']}: no retira E0.")
    elif v1["categoria"] == ESTABLECIDO:
        out["marcas"]["V1"] = [M_CUENCA]
        out["motivos"].append("V1 establecido solo en la boya: E0 y E1 siguen activos (establecido en la cuenca).")
    elif M_EPOCA in v1["marcas"]:
        out["motivos"].append("La altimetría establece y la boya descarta: E0 y E1 siguen activos.")
    else:
        out["motivos"].append(f"V1 {v1['categoria']}: E0 y E1 siguen activos.")
    # V2 → E2
    mf2 = v2["marcas_fuente"].get("a", [])
    a2 = "a" in v2.get("fuentes_que_rechazan", [])
    if v2["categoria"] == ESTABLECIDO and a2 and not mf2 and not v2["marcas"] and v3a_cat == ESTABLECIDO and absorbe_v2:
        out["E2"] = "retirado"
        out["motivos"].append("V2 establecido en la altimetría sin marcas, V3a establecido y escalón absorbido.")
    else:
        if v2["categoria"] == ESTABLECIDO and not a2:
            out["marcas"]["V2"] = [M_CUENCA]
        out["motivos"].append(f"V2 {v2['categoria']}: E2 sigue activo (§6.1).")
    return out


# --- Corrección (§6.2) ----------------------------------------------------------------------------------------


def variantes(cat_v3a: str, cat_v3b: str, v3: dict) -> dict:
    """Variantes de la corrección según V3a y V3b (§6.2). Cada una: tipo, parámetro y factor en la cola."""
    bc, ba = v3["B_cola"]["estimado"], v3["B_alta"]["estimado"]
    sin = {"nombre": "sin_corregir", "tipo": "sin corrección", "B": 0.0}
    cola = {"nombre": "B_cola", "tipo": "factor constante", "B": bc}
    alta = {"nombre": "B_alta", "tipo": "factor constante", "B": ba, "estadistico": v3["B_alta"]["estadistico"]}
    cuant = {"nombre": "cuantiles", "tipo": "corrección por cuantiles", "B_q": v3["B_q"]}
    tabla = {
        (ESTABLECIDO, ESTABLECIDO): [cuant],
        (ESTABLECIDO, DESCARTADO): [cola],
        (ESTABLECIDO, NO_CONCLUYENTE): [cola, alta],
        (NO_CONCLUYENTE, ESTABLECIDO): [sin, cuant],
        (NO_CONCLUYENTE, DESCARTADO): [sin, cola],
        (NO_CONCLUYENTE, NO_CONCLUYENTE): [sin, cola, alta],
    }
    vs = [sin] if cat_v3a == DESCARTADO else tabla[(cat_v3a, cat_v3b)]
    vs = [v for v in vs if v["tipo"] != "factor constante" or v["B"] is not None]
    for v in vs:
        if v["tipo"] == "corrección por cuantiles":
            alto = [b for b in v["B_q"].values() if b is not None]
            v["B_cola_alta"] = alto[-1] if alto else 0.0
        v["factor_cola"] = float(np.exp(-(v.get("B", v.get("B_cola_alta", 0.0)))))
    peor = max(vs, key=lambda v: v["factor_cola"])["nombre"] if vs else None
    return {"V3a": cat_v3a, "V3b": cat_v3b, "variantes": vs, "mas_desfavorable": peor, "mismo_rango": len(vs) > 1}


def funcion_cuantiles(B_q: dict) -> tuple[np.ndarray, np.ndarray]:
    """(q, B) de la corrección por cuantiles: los estimables entre P50 y P99.9; constante fuera."""
    qs = [(CV.CUANTILES[e], b) for e, b in B_q.items() if b is not None]
    if not qs:
        return np.array([0.5]), np.array([0.0])
    q, b = map(np.array, zip(*sorted(qs), strict=True))
    return q, b


def corregir(t: pd.DatetimeIndex, hs: np.ndarray, variante: dict, seleccion: np.ndarray) -> np.ndarray:
    """Aplica una variante a los registros seleccionados (el resto queda igual). La de cuantiles usa la F empírica de
    la parte corregida con peso mensual, y se fuerza monótona (no puede invertir el orden de dos registros)."""
    out = np.asarray(hs, float).copy()
    x = out[seleccion]
    if variante["tipo"] == "sin corrección":
        return out
    if variante["tipo"] == "factor constante":
        out[seleccion] = x * np.exp(-variante["B"])
        return out
    q, b = funcion_cuantiles(variante["B_q"])
    from ..homogeneidad import cuantiles as Q  # noqa: PLC0415

    w = Q.pesos_mensuales(pd.DatetimeIndex(t)[seleccion], np.isfinite(x))
    ok = np.isfinite(x)
    o = np.argsort(x[ok], kind="stable")
    F = np.cumsum(w[ok][o]) / w[ok].sum()
    Bx = np.interp(F, q, b)  # constante fuera de [q_min, q_max]
    y = x[ok][o] * np.exp(-Bx)
    y = np.maximum.accumulate(y)
    z = np.empty(ok.sum())
    z[o] = y
    xx = x.copy()
    xx[ok] = z
    out[seleccion] = xx
    return out


# --- §20.6 y fila de la fuente (§6.3) -----------------------------------------------------------------------


def rango_subida(potencia_json: dict) -> dict:
    """Envolvente sobre los tres SI (ρ = 0.8) entre × 1 y × 2, por configuración (§20.6, punto 2)."""
    out = {}
    puntos = {p["nombre"]: p for p in potencia_json["rejilla"]}
    for conf in ("con_maximos", "sin_maximos"):
        filas = []
        for nombre, v in potencia_json["subida_esperada"].items():
            p = puntos[nombre]
            if p["rho"] == CV.RHO_NOMINAL and p["factor_cal"] in (1.0, 2.0):
                filas.append((p["factor_cal"], v[conf]["subida_media"], nombre))
        lo = min((f for f in filas if f[0] == 1.0), key=lambda f: f[1])
        hi = max((f for f in filas if f[0] == 2.0), key=lambda f: f[1])
        x2 = [f[1] for f in filas if f[0] == 2.0]
        out[conf] = {
            "limite_inferior": lo[1],
            "punto_inferior": lo[2],
            "limite_superior": hi[1],
            "punto_superior": hi[2],
            "rejilla_x2_min": min(x2),
            "rejilla_x2_max": max(x2),
        }
    return out


def hallazgo_20_6(rango: dict, configuracion: str | None = None) -> str:
    x2 = rango["con_maximos"]
    texto = (
        "La subida esperada de la regla de variantes supera el 2 % de la ADR-0006 dentro de la rejilla publicada: "
        f"de {100 * x2['rejilla_x2_min']:.2f} a {100 * x2['rejilla_x2_max']:.2f} % con calibración y ruido × 2 "
        "(con máximos). Declarado por la rejilla × 2, no por la regla nominal (§20.6, punto 1)."
    )
    if configuracion is not None:  # §20.8: se añaden las cifras de la configuración elegida, sin quitar las anteriores
        c = rango[configuracion]
        nombre = {"con_maximos": "con máximos", "sin_maximos": "sin máximos"}[configuracion]
        texto += (
            f" Configuración elegida por la función de estructura: {nombre}; × 2 de {100 * c['rejilla_x2_min']:.2f} "
            f"a {100 * c['rejilla_x2_max']:.2f} %; rango que recibe la F12, {100 * c['limite_inferior']:.2f}–"
            f"{100 * c['limite_superior']:.2f} % (añadido posterior a los datos, §20.8)."
        )
    return texto


def fila_fuente(v3: dict, gradiente: dict, rango_conf: dict, mde_v3a: float | None) -> dict:
    """§6.3. Sin B_cola estimable en el lado A: sin corrección y el rango de C-06."""
    bc = v3["B_cola"]
    if bc["estimado"] is None:
        return {
            "observacion_valida": False,
            "rango": list(C.C06),
            "rotulo": "C-06, fuera del rango de Hs del sitio",
            "subida_esperada": rango_conf,
        }
    fila = {
        "observacion_valida": True,
        "B_cola": bc["estimado"],
        "ic95_B_cola": bc["ic95"],
        "comparacion_C06": {"rango": list(C.C06), "rotulo": "fuera del rango de Hs del sitio; al lado, no en unión"},
        "subida_esperada": rango_conf,
        "subida_esperada_nota": "Ya incluye el sesgo del estimador (§20.5, punto 4); la F12 no lo suma otra vez.",
        "efecto_minimo_V3a": mde_v3a,
        "separada_de": "la fila del sesgo de profundidad (ADR-0015), sin compensarlas",
    }
    if gradiente.get("valido") is not True:
        fila["traslado_gradiente"] = gradiente.get("estado")
        fila["B_P90_23km"] = gradiente.get("B_P90_23km")
    return fila
