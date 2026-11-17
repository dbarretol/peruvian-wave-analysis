"""Cuantiles con el mismo peso para cada mes del calendario (contrato de la F4, §3.2 y §3.3).

**Definición.** Dentro de un conjunto de registros (un tramo, un lado de un contraste), el registro i del mes del
calendario m pesa w_i = 1 / (M · N_m), donde N_m es el número de registros válidos del mes m en el conjunto y M el
número de meses del calendario presentes. Cada mes suma 1/M. El cuantil q es el **menor valor x con F_w(x) ≥ q**
(cuantil por inversión de la función de distribución ponderada, sin interpolar). Es equivariante por escala:
multiplicar todos los valores por (1 + δ) multiplica el cuantil por (1 + δ) exactamente.

**Años de contraste.** Un año de contraste empieza en ``mes_inicio`` y se nombra por el año calendario de ese mes
(con inicio en marzo, el año de contraste 2016 va de 2016-03 a 2017-02). Un año con menos meses que los 12 de un
año completo es **parcial**: se permuta como uno más y queda fuera de la media de máximos anuales (§3.5, F04-S-03).

**Cálculo rápido para las permutaciones.** ``TablaCola`` guarda solo los registros sobre un umbral de corte, con su
año y su mes, y el número de registros por año y mes. Para una matriz de pertenencia (permutaciones × años), el
peso de cada registro de la cola y la suma acumulada desde arriba dan los cuantiles altos de todas las permutaciones
a la vez. Si la cola guardada no alcanza para un cuantil en alguna permutación, se lanza un error: el corte es bajo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def anio_de_contraste(tiempos, mes_inicio: int = 1) -> np.ndarray:
    t = pd.DatetimeIndex(tiempos)
    a = t.year.to_numpy(dtype=int)
    return a - (t.month.to_numpy() < mes_inicio).astype(int) if mes_inicio != 1 else a


def pesos_mensuales(tiempos, validos=None) -> np.ndarray:
    """w_i = 1/(M·N_m) sobre los registros válidos; 0 en los no válidos."""
    t = pd.DatetimeIndex(tiempos)
    v = np.ones(len(t), bool) if validos is None else np.asarray(validos, bool)
    mes = t.month.to_numpy()
    n = np.bincount(mes[v], minlength=13).astype(float)
    M = int((n[1:] > 0).sum())
    w = np.zeros(len(t))
    if M == 0:
        return w
    w[v] = 1.0 / (M * n[mes[v]])
    return w


def cuantil_ponderado(x, w, q) -> np.ndarray:
    """Menor x con F_w(x) ≥ q. ``q`` escalar o vector. Ignora valores no finitos y pesos nulos."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    qs = np.atleast_1d(np.asarray(q, float))
    if x.size == 0:
        return np.full(qs.shape, np.nan)
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    F = np.cumsum(w) / w.sum()
    i = np.searchsorted(F, qs - 1e-12, side="left")
    return x[np.minimum(i, x.size - 1)]


def cuantiles_mensuales(serie: pd.Series, q) -> np.ndarray:
    """Cuantiles de una serie con peso mensual igual."""
    v = np.isfinite(serie.to_numpy(float))
    return cuantil_ponderado(serie.to_numpy(float), pesos_mensuales(serie.index, v), q)


def maximos_anuales(serie: pd.Series, mes_inicio: int = 1, meses=None) -> pd.DataFrame:
    """Máximo por año de contraste, con el número de meses del calendario con dato y si el año es completo.

    ``meses``: si se da, solo esos meses (máximo estacional); completo = tiene todos esos meses."""
    s = serie[np.isfinite(serie.to_numpy(float))]
    if meses is not None:
        s = s[s.index.month.isin(list(meses))]
    a = anio_de_contraste(s.index, mes_inicio)
    df = pd.DataFrame({"x": s.to_numpy(float), "anio": a, "mes": s.index.month})
    g = df.groupby("anio")
    n_meses = 12 if meses is None else len(meses)
    out = pd.DataFrame({"maximo": g["x"].max(), "meses": g["mes"].nunique()})
    out["completo"] = out["meses"] == n_meses
    return out


@dataclass
class TablaCola:
    """Registros de la cola (sobre ``corte``) de un conjunto de años, preparados para permutar años.

    valores: cola en orden descendente; anio_idx y mes_idx: índice de año (0..n_anios−1) y mes (0..11) de cada
    valor; conteo: registros válidos por (año, mes), forma (n_anios, 12); maximos: máximo de cada año; completo:
    si el año es completo; anios: etiquetas."""

    valores: np.ndarray
    anio_idx: np.ndarray
    mes_idx: np.ndarray
    conteo: np.ndarray
    maximos: np.ndarray
    completo: np.ndarray
    anios: np.ndarray
    corte: float

    @classmethod
    def desde_serie(
        cls, serie: pd.Series, mes_inicio: int, corte: float, factores: dict[int, float] | None = None
    ) -> TablaCola:
        """``factores``: multiplicador por año de contraste (inyección de un escalón); por defecto 1."""
        s = serie[np.isfinite(serie.to_numpy(float))]
        a = anio_de_contraste(s.index, mes_inicio)
        anios = np.unique(a)
        pos = np.searchsorted(anios, a)
        x = s.to_numpy(float)
        if factores:
            f = np.array([factores.get(int(y), 1.0) for y in anios])
            x = x * f[pos]
        mes = s.index.month.to_numpy() - 1
        conteo = np.zeros((anios.size, 12))
        np.add.at(conteo, (pos, mes), 1.0)
        maximos = np.full(anios.size, -np.inf)
        np.maximum.at(maximos, pos, x)
        completo = (conteo > 0).sum(axis=1) == 12
        cola = x > corte
        o = np.argsort(-x[cola], kind="stable")
        return cls(x[cola][o], pos[cola][o], mes[cola][o], conteo, maximos, completo, anios, float(corte))

    def con_factores(self, f: np.ndarray) -> TablaCola:
        """Misma tabla con cada año multiplicado por f[año] (f > 0). Reordena la cola.

        Un valor que estaba bajo el corte queda, escalado, bajo corte·max(f): solo la parte de la cola sobre ese
        nuevo corte es completa, y es la que se conserva."""
        f = np.asarray(f, float)
        v = self.valores * f[self.anio_idx]
        corte = self.corte * float(f.max())
        o = np.argsort(-v, kind="stable")
        o = o[v[o] > corte]
        return TablaCola(
            v[o], self.anio_idx[o], self.mes_idx[o], self.conteo, self.maximos * f, self.completo, self.anios, corte
        )

    def cuantiles(self, pertenencia: np.ndarray, qs) -> np.ndarray:
        """Cuantiles con peso mensual del conjunto de años marcado en cada fila de ``pertenencia`` (k × n_anios,
        0/1). Devuelve (k × len(qs))."""
        P = np.atleast_2d(np.asarray(pertenencia, float))
        N = P @ self.conteo  # k × 12
        M = (N > 0).sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(N > 0, 1.0 / (M * N), 0.0)
        w = P[:, self.anio_idx] * inv[:, self.mes_idx]  # k × n_cola
        S = np.cumsum(w, axis=1)
        S_excl = np.concatenate([np.zeros((S.shape[0], 1)), S[:, :-1]], axis=1)
        out = np.empty((P.shape[0], len(qs)))
        for j, q in enumerate(qs):
            k = (S_excl <= (1.0 - q) + 1e-12).sum(axis=1)
            if np.any(k >= self.valores.size) or np.any(S[:, -1] <= (1.0 - q) + 1e-12):
                raise ValueError(f"La cola guardada (corte {self.corte:.3f}) no alcanza para el cuantil {q}")
            out[:, j] = self.valores[k - 1]
        return out

    def media_maximos(self, pertenencia: np.ndarray) -> np.ndarray:
        """Media de los máximos de los años completos marcados (k,)."""
        P = np.atleast_2d(np.asarray(pertenencia, float)) * self.completo[None, :]
        n = P.sum(axis=1)
        return (P @ self.maximos) / np.where(n > 0, n, np.nan)


@dataclass
class HistAnual:
    """Histograma por (año, mes) sobre los valores distintos de una serie, para cuantiles con peso mensual de
    cualquier conjunto de años con multiplicidad (bootstrap de bloques anuales). Mismo cuantil que
    ``cuantil_ponderado`` con ``pesos_mensuales``."""

    valores: np.ndarray  # valores distintos, crecientes
    H: np.ndarray  # (n_anios, 12, n_valores)
    maximos: np.ndarray
    completo: np.ndarray
    anios: np.ndarray

    @classmethod
    def desde_serie(cls, serie: pd.Series, mes_inicio: int = 1, meses=None) -> HistAnual:
        s = serie[np.isfinite(serie.to_numpy(float))]
        if meses is not None:
            s = s[s.index.month.isin(list(meses))]
        a = anio_de_contraste(s.index, mes_inicio)
        anios = np.unique(a)
        pos = np.searchsorted(anios, a)
        valores, inv = np.unique(s.to_numpy(float), return_inverse=True)
        mes = s.index.month.to_numpy() - 1
        H = np.zeros((anios.size, 12, valores.size))
        np.add.at(H, (pos, mes, inv.ravel()), 1.0)
        maximos = np.full(anios.size, -np.inf)
        np.maximum.at(maximos, pos, s.to_numpy(float))
        n_meses = 12 if meses is None else len(meses)
        completo = (H.sum(axis=2) > 0).sum(axis=1) == n_meses
        return cls(valores, H, maximos, completo, anios)

    def cuantiles(self, P: np.ndarray, qs) -> np.ndarray:
        """P: (k × n_anios) multiplicidad de cada año. Devuelve (k × len(qs))."""
        P = np.atleast_2d(np.asarray(P, float))
        N = np.einsum("ky,ym->km", P, self.H.sum(axis=2))
        M = (N > 0).sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(N > 0, 1.0 / (M * N), 0.0)
        out = np.empty((P.shape[0], len(qs)))
        for i in range(P.shape[0]):
            h = np.einsum("y,ymv->mv", P[i], self.H)
            w = (inv[i][:, None] * h).sum(axis=0)
            F = np.cumsum(w)
            F /= F[-1]
            j = np.searchsorted(F, np.asarray(qs) - 1e-12, side="left")
            out[i] = self.valores[np.minimum(j, self.valores.size - 1)]
        return out

    def media_maximos(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float)) * self.completo[None, :]
        n = P.sum(axis=1)
        return (P @ np.where(np.isfinite(self.maximos), self.maximos, 0.0)) / np.where(n > 0, n, np.nan)
