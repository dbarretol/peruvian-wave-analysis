"""Familias confirmatorias A (escalón en la cola) y D (Tp en Hs alto), con su error de tipo I y su potencia
(contrato de la F4, §3.5 y §3.6).

Cada réplica de simulación deriva su generador de nombres (``comun.semillas``), no del orden de ejecución: el
resultado no depende del número de procesos.

**Error de tipo I bajo dependencia (§3.5).** En cada réplica los años se reasignan a las posiciones con una cópula
AR(1) (ρ = 0.3, supuesto) sobre su máximo anual, sin escalón. Se mide la fracción de réplicas que rechazan al 5 %.

**Potencia (§3.6).** En cada réplica se reasignan primero los años (al azar; o con AR(1) si se aplicó alguna
corrección de la regla de §3.5) y después se multiplica un lado por (1 + δ) (A) o se desplaza su Tp (D).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..comun import semillas
from . import config as C
from . import permutacion as P
from .avance import Avance
from .cuantiles import TablaCola, anio_de_contraste

CUANTIL_CORTE = 0.90  # cola guardada para las permutaciones: sobre el P90 sin ponderar del contraste


@dataclass
class Contraste:
    nombre: str
    tabla: TablaCola
    lado: np.ndarray  # por posición (= año de la tabla, en orden de tiempo): 0 lado 1, 1 lado 2
    anios: np.ndarray
    parcial: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def identidad(self) -> np.ndarray:
        return np.arange(self.anios.size)


def preparar_contraste(
    nombre: str, hs: pd.Series, lado1: tuple[str, str], lado2: tuple[str, str], mes_inicio: int
) -> Contraste:
    """``lado1``/``lado2``: (inicio, fin) inclusive al día. Lo que cae fuera de los dos lados no entra."""
    t = hs.index
    fin = lambda b: pd.Timestamp(b) + pd.Timedelta(days=1)  # noqa: E731
    m1 = (t >= pd.Timestamp(lado1[0])) & (t < fin(lado1[1]))
    m2 = (t >= pd.Timestamp(lado2[0])) & (t < fin(lado2[1]))
    s = hs[m1 | m2]
    corte = float(np.nanquantile(s.to_numpy(float), CUANTIL_CORTE))
    tabla = TablaCola.desde_serie(s, mes_inicio, corte)
    a_lado2 = np.unique(anio_de_contraste(t[m2], mes_inicio))
    a_lado1 = np.unique(anio_de_contraste(t[m1], mes_inicio))
    if np.intersect1d(a_lado1, a_lado2).size:
        raise ValueError(f"{nombre}: un año de contraste cae a los dos lados; revise el mes de inicio")
    lado = np.isin(tabla.anios, a_lado2).astype(int)
    return Contraste(nombre, tabla, lado, tabla.anios, ~tabla.completo)


# --- Familia A ---------------------------------------------------------------------------------
def prueba_A(c: Contraste, esquema: str, B: int, *nombres_semilla: str) -> P.ResultadoMaxT:
    return P.prueba_maxT(c.tabla, c.lado, c.identidad, esquema, B, semillas.generador(*nombres_semilla))


def _orden(c: Contraste, dependencia: bool, rho: float, rng: np.random.Generator) -> np.ndarray:
    if dependencia:
        return P.orden_ar1(c.tabla.maximos, rho, rng)
    return P.orden_al_azar(c.anios.size, rng)


def _tarea_A(args) -> list[float]:
    """Réplicas [i0, i1) de una simulación de la familia A: p mínimo ajustado de cada réplica."""
    c, esquema, B, delta, lado_iny, dependencia, rho, nombres, i0, i1 = args
    out = []
    for i in range(i0, i1):
        rng = semillas.generador(*nombres, str(i))
        orden = _orden(c, dependencia, rho, rng)
        tabla = c.tabla
        if delta:
            tabla = tabla.con_factores(P.factores_inyeccion(orden, c.lado, lado_iny, delta))
        r = P.prueba_maxT(tabla, c.lado, orden, esquema, B, rng)
        out.append(r.p_min)
    return out


def simular_A(
    c: Contraste,
    esquema: str,
    B: int,
    replicas: int,
    delta: float,
    lado_iny: int,
    dependencia: bool,
    rho: float,
    nombres: tuple[str, ...],
    procesos_n: int = 1,
    lote: int = 25,
    avance: Avance | None = None,
) -> np.ndarray:
    """p mínimo ajustado de cada réplica (reasignación + inyección opcional). Con ``avance``, cada lote terminado
    se guarda y una corrida interrumpida retoma desde ahí."""
    tareas = [
        (c, esquema, B, delta, lado_iny, dependencia, rho, nombres, i, min(i + lote, replicas))
        for i in range(0, replicas, lote)
    ]
    av = avance or Avance(None, "")
    firmas = [["A", c.nombre, *t[1:]] for t in tareas]
    res = av.ejecutar(
        _tarea_A, tareas, firmas, [t[-1] - t[-2] for t in tareas], procesos_n, " ".join(map(str, nombres[1:])) or "A"
    )
    return np.array([p for r in res for p in r])


# --- Familia D ---------------------------------------------------------------------------------
@dataclass
class EventosTp:
    tp: np.ndarray  # Tp en el pico de cada evento
    pos: np.ndarray  # posición (año de contraste) de cada evento
    lado: np.ndarray  # lado por posición
    anios: np.ndarray


def preparar_tp(
    picos: pd.DataFrame, columna_tp: str, lado1: tuple[str, str], lado2: tuple[str, str], mes_inicio: int
) -> EventosTp:
    t = pd.DatetimeIndex(picos.index)
    fin = lambda b: pd.Timestamp(b) + pd.Timedelta(days=1)  # noqa: E731
    m1 = (t >= pd.Timestamp(lado1[0])) & (t < fin(lado1[1]))
    m2 = (t >= pd.Timestamp(lado2[0])) & (t < fin(lado2[1]))
    m = (m1 | m2) & np.isfinite(picos[columna_tp].to_numpy(float))
    a = anio_de_contraste(t[m], mes_inicio)
    # todas las posiciones de año de contraste de los dos lados, aunque no tengan eventos
    a1 = np.arange(
        anio_de_contraste([pd.Timestamp(lado1[0])], mes_inicio)[0],
        anio_de_contraste([pd.Timestamp(lado1[1])], mes_inicio)[0] + 1,
    )
    a2 = np.arange(
        anio_de_contraste([pd.Timestamp(lado2[0])], mes_inicio)[0],
        anio_de_contraste([pd.Timestamp(lado2[1])], mes_inicio)[0] + 1,
    )
    anios = np.concatenate([a1, a2])
    lado = np.concatenate([np.zeros(a1.size, int), np.ones(a2.size, int)])
    return EventosTp(picos[columna_tp].to_numpy(float)[m], np.searchsorted(anios, a), lado, anios)


def _dif_medianas(tp: np.ndarray, en2: np.ndarray) -> np.ndarray:
    """Diferencia de medianas (lado 2 − lado 1) para cada fila de ``en2`` (k × n_eventos, booleana)."""
    out = np.empty(en2.shape[0])
    for i, m in enumerate(en2):
        a, b = tp[~m], tp[m]
        out[i] = np.median(b) - np.median(a) if a.size and b.size else np.nan
    return out


@dataclass(frozen=True)
class ResultadoD:
    diferencia: float
    p: float
    B: int
    n_eventos: tuple[int, int]


def prueba_D(ev: EventosTp, tp: np.ndarray, orden: np.ndarray, esquema: str, B: int, rng) -> ResultadoD:
    """Permutación de años de contraste (las mismas unidades que la familia A), bilateral."""
    units = P.unidades(ev.lado, esquema)
    obs = np.array([[int(ev.lado[u[0]]) for u in units]], dtype=np.int8)
    lado_anio = P.pertenencia(obs, units, orden)[0]
    en2_obs = lado_anio[ev.pos].astype(bool)
    d_obs = float(_dif_medianas(tp, en2_obs[None, :])[0])
    A = P.sorteos(len(units), int(obs.sum()), B, rng)
    perms = P.pertenencia(A, units, orden)[:, ev.pos].astype(bool)
    d = _dif_medianas(tp, perms)
    d = d[np.isfinite(d)]
    p = (1 + np.sum(np.abs(d) >= abs(d_obs) - 1e-12)) / (d.size + 1)
    return ResultadoD(d_obs, float(p), B, (int((~en2_obs).sum()), int(en2_obs.sum())))


def _tarea_D(args) -> list[float]:
    ev, esquema, B, efecto, dependencia, rho, nombres, i0, i1 = args
    out = []
    for i in range(i0, i1):
        rng = semillas.generador(*nombres, str(i))
        if dependencia:
            # clave: mediana de Tp de cada año (los años sin eventos, al final)
            med = np.array(
                [np.median(ev.tp[ev.pos == k]) if np.any(ev.pos == k) else -np.inf for k in range(ev.anios.size)]
            )
            orden = P.orden_ar1(med, rho, rng)
        else:
            orden = P.orden_al_azar(ev.anios.size, rng)
        tp = ev.tp
        if efecto:
            lado_de_anio = np.empty(ev.anios.size, int)
            lado_de_anio[orden] = ev.lado
            tp = tp + efecto * (lado_de_anio[ev.pos] == 1)
        out.append(prueba_D(ev, tp, orden, esquema, B, rng).p)
    return out


def simular_D(
    ev, esquema, B, replicas, efecto, dependencia, rho, nombres, procesos_n=1, lote=50, avance: Avance | None = None
) -> np.ndarray:
    tareas = [
        (ev, esquema, B, efecto, dependencia, rho, nombres, i, min(i + lote, replicas))
        for i in range(0, replicas, lote)
    ]
    av = avance or Avance(None, "")
    firmas = [["D", *t[1:]] for t in tareas]
    res = av.ejecutar(
        _tarea_D, tareas, firmas, [t[-1] - t[-2] for t in tareas], procesos_n, " ".join(map(str, nombres[1:])) or "D"
    )
    return np.array([p for r in res for p in r])


# --- Regla de §3.5 y resultado de ADR-0005 ----------------------------------------------------
@dataclass
class ReglaDependencia:
    esquema: str
    critico: float
    dependencia_en_potencia: bool
    tipo_I: dict  # esquema → error de tipo I medido
    tipo_I_tras_recalibrar: float | None = None
    recalibrable: bool = True  # falso si más del α de las réplicas cae en el menor p posible con ese B


def aplicar_regla(simular_nula, alfa: float, maximo: float, desde: str = "anios") -> ReglaDependencia:
    """``simular_nula(esquema, etiqueta)`` → p (mínimos) de réplicas AR(1) sin cambio. Regla de §3.5:
    ≤ máximo → años; si no, bloques de dos años; si sigue alto, valor crítico recalibrado (con bloques) y
    verificación con réplicas independientes. ``desde = "bloques2"`` empieza en bloques (familia D cuando A en C2
    ya usa bloques; §21, K-05)."""
    medidos = {}
    if desde == "anios":
        p_anios = simular_nula("anios", "tipo-I")
        medidos["anios"] = float(np.mean(p_anios <= alfa))
        if medidos["anios"] <= maximo:
            return ReglaDependencia("anios", alfa, False, medidos)
    p_b = simular_nula("bloques2", "tipo-I")
    medidos["bloques2"] = float(np.mean(p_b <= alfa))
    if medidos["bloques2"] <= maximo:
        return ReglaDependencia("bloques2", alfa, True, medidos)
    c = P.valor_critico_recalibrado(p_b, alfa)
    p_v = simular_nula("bloques2", "verificacion")
    recalibrable = bool(c >= float(np.min(p_b)))
    return ReglaDependencia("bloques2", c, True, medidos, float(np.mean(p_v <= c)), recalibrable)


def resultado_adr0005(rechaza: bool, potencia: float | None, minimo: float, sin_raiz: bool = False) -> str:
    if rechaza:
        return C.ESTABLECIDO
    if sin_raiz:
        return C.SIN_RAIZ
    if potencia is not None and potencia >= minimo:
        return C.DESCARTADO
    return C.NO_CONCLUYENTE
