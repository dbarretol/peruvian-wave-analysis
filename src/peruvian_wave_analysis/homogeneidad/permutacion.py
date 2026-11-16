"""Permutación de años de contraste con max-T (familia A) y su error de tipo I bajo dependencia (contrato §3.5).

**Posiciones y años.** Los años de contraste de los dos lados ocupan posiciones 0..n−1 en orden de tiempo; el lado
de cada posición es fijo (``lado``: 0 = lado 1, 1 = lado 2). ``orden[p]`` dice qué año de la tabla ocupa la
posición p: la identidad para los datos observados; otra cosa cuando una réplica de potencia o de error de tipo I
reasigna los años (al azar, o con dependencia AR(1)).

**Unidades de permutación.** Por años: cada posición es una unidad. Por bloques de dos años: posiciones
consecutivas del mismo lado, de dos en dos; con un número impar, el último bloque de cada lado es de un año (v7).
Una permutación reparte las unidades entre los lados conservando cuántas unidades tiene cada lado.

**Estadísticos.** T_k = log(estadístico del lado 2 / estadístico del lado 1), para P99, P99.9 (con peso mensual) y
la media de los máximos de los años completos. Bilateral.

**max-T (Westfall y Young, 1993).** z_k = (T_k − μ_k)/σ_k con μ y σ de la distribución de permutación. p ajustado
de k = (1 + #{b : max_j |z_j^(b)| ≥ |z_k|}) / (B + 1). La familia rechaza si el menor p ajustado ≤ el valor
crítico (α, o el recalibrado por la regla de §3.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cuantiles import TablaCola

QS_A = (0.99, 0.999)
NOMBRES_A = ("P99", "P99.9", "media_maximos")
TAM_LOTE = 250


def unidades(lado: np.ndarray, esquema: str) -> list[np.ndarray]:
    """Unidades de permutación (listas de posiciones). ``esquema``: «anios» o «bloques2»."""
    lado = np.asarray(lado, int)
    if esquema == "anios":
        return [np.array([p]) for p in range(lado.size)]
    if esquema != "bloques2":
        raise ValueError(f"Esquema desconocido: {esquema}")
    out = []
    for s in (0, 1):
        pos = np.flatnonzero(lado == s)
        for i in range(0, pos.size, 2):
            out.append(pos[i : i + 2])
    return out


def sorteos(n_unidades: int, n_lado2: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B × n_unidades, 1 si la unidad va al lado 2. Cada fila tiene n_lado2 unos."""
    r = rng.random((B, n_unidades)).argsort(axis=1)
    return (r < n_lado2).astype(np.int8)


def pertenencia(asig_unidades: np.ndarray, units: list[np.ndarray], orden: np.ndarray) -> np.ndarray:
    """De asignación por unidad (k × n_unidades) a pertenencia al lado 2 por año de la tabla (k × n_anios)."""
    A = np.atleast_2d(asig_unidades)
    pos_lado = np.zeros((A.shape[0], orden.size), dtype=np.int8)
    for j, u in enumerate(units):
        pos_lado[:, u] = A[:, [j]]
    out = np.zeros_like(pos_lado)
    out[:, orden] = pos_lado
    return out


def estadisticos(tabla: TablaCola, P2: np.ndarray) -> np.ndarray:
    """T (k × 3): log del cociente lado 2 / lado 1 de P99, P99.9 y media de máximos."""
    P2 = np.atleast_2d(P2).astype(float)
    P1 = 1.0 - P2
    q1, q2 = tabla.cuantiles(P1, QS_A), tabla.cuantiles(P2, QS_A)
    m1, m2 = tabla.media_maximos(P1), tabla.media_maximos(P2)
    return np.column_stack([np.log(q2 / q1), np.log(m2 / m1)])


@dataclass(frozen=True)
class ResultadoMaxT:
    T: np.ndarray  # observado, (3,)
    z: np.ndarray
    p_crudo: np.ndarray
    p_ajustado: np.ndarray
    B: int

    @property
    def p_min(self) -> float:
        return float(self.p_ajustado.min())


def prueba_maxT(
    tabla: TablaCola,
    lado: np.ndarray,
    orden: np.ndarray,
    esquema: str,
    B: int,
    rng: np.random.Generator,
) -> ResultadoMaxT:
    lado = np.asarray(lado, int)
    units = unidades(lado, esquema)
    obs_u = np.array([[int(lado[u[0]]) for u in units]], dtype=np.int8)
    T_obs = estadisticos(tabla, pertenencia(obs_u, units, orden))[0]
    n2 = int(obs_u.sum())
    Tp = np.empty((B, T_obs.size))
    for i in range(0, B, TAM_LOTE):
        k = min(TAM_LOTE, B - i)
        Tp[i : i + k] = estadisticos(tabla, pertenencia(sorteos(len(units), n2, k, rng), units, orden))
    return ajuste_maxT(T_obs, Tp)


def ajuste_maxT(T_obs: np.ndarray, Tp: np.ndarray) -> ResultadoMaxT:
    """p crudos y ajustados por max-T a partir del estadístico observado (k,) y los permutados (B × k)."""
    B = Tp.shape[0]
    mu, sd = Tp.mean(axis=0), Tp.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    z_obs = (T_obs - mu) / sd
    z_p = np.abs((Tp - mu) / sd)
    max_p = z_p.max(axis=1)
    p_adj = (1 + (max_p[:, None] >= np.abs(z_obs)[None, :] - 1e-12).sum(axis=0)) / (B + 1)
    p_raw = (1 + (z_p >= np.abs(z_obs)[None, :] - 1e-12).sum(axis=0)) / (B + 1)
    return ResultadoMaxT(np.asarray(T_obs), z_obs, p_raw, p_adj, B)


def holm(p) -> np.ndarray:
    """p ajustados de Holm."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = p.size
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


# --- Reasignación de años --------------------------------------------------------------------


def orden_al_azar(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    a = np.empty(n)
    a[0] = rng.standard_normal()
    e = rng.standard_normal(n) * np.sqrt(1 - rho**2)
    for i in range(1, n):
        a[i] = rho * a[i - 1] + e[i]
    return a


def orden_ar1(clave: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Reasigna los años a las posiciones para que ``clave`` (p. ej., el máximo anual) siga una dependencia AR(1).

    Cópula gaussiana: la posición con el k-ésimo valor latente recibe el año con la k-ésima clave. Conserva la
    distribución de los años y les da una autocorrelación de rango cercana a ρ."""
    clave = np.asarray(clave, float)
    a = ar1(clave.size, rho, rng)
    orden = np.empty(clave.size, dtype=int)
    orden[np.argsort(a, kind="stable")] = np.argsort(clave, kind="stable")
    return orden


def factores_inyeccion(orden: np.ndarray, lado: np.ndarray, lado_inyectado: int, delta: float) -> np.ndarray:
    """Factor por año de la tabla: 1 + δ en los años que ocupan posiciones del lado inyectado."""
    f = np.ones(orden.size)
    f[orden[np.asarray(lado) == lado_inyectado]] = 1.0 + delta
    return f


def valor_critico_recalibrado(p_min_nula, alfa: float) -> float:
    """Mayor c entre los p mínimos simulados con P(p_min ≤ c) ≤ α (rechazo con p_min ≤ c)."""
    p = np.sort(np.asarray(p_min_nula, float))
    k = int(np.floor(alfa * p.size))
    if k == 0:
        return float(p[0]) * (1 - 1e-9)
    # el valor k-ésimo (1-based) deja k réplicas ≤ c; si hay empates en el siguiente, se baja para no pasarse
    c = float(p[k - 1])
    while (p <= c).sum() > k and c > 0:
        menores = p[p < c]
        c = float(menores.max()) if menores.size else 0.0
    return c
