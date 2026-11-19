"""Pruebas de la dependencia temporal de la F6 (contrato §3.5 y §8) y de la etapa 0 sintética."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.clima import etapa0 as E0
from peruvian_wave_analysis.clima import sinteticos as S
from peruvian_wave_analysis.comun import semillas
from peruvian_wave_analysis.extremos import dependencia as D
from peruvian_wave_analysis.extremos.decluster import decluster_rachas


def _bloque(x, inicio="2001-01-01", anio=2001) -> D.Bloque:
    t = pd.date_range(inicio, periods=len(x), freq="3h")
    return D.Bloque(anio, np.asarray(x, float), t.month.to_numpy(), D.dia_fijo(t))


# --- θ de Ferro y Segers ---------------------------------------------------------------------------------


def test_theta_dos_ramas_a_mano():
    # max T ≤ 2: 2 (ΣT)² / (n ΣT²) ≥ 1.8 con T ∈ {1, 2}; se acota en 1
    assert D.theta_intervalos(np.array([1, 2, 1])) == 1.0
    # max T > 2: 2 (Σ(T−1))² / (n Σ(T−1)(T−2)) = 2·81 / (4·72)
    assert D.theta_intervalos(np.array([1, 1, 1, 10])) == pytest.approx(162 / 288)
    assert np.isnan(D.theta_intervalos(np.array([], int)))


def test_declustering_automatico_con_empates():
    T = np.array([1, 1, 5, 1, 8, 1, 5])  # ordenados: 8, 5, 5, 1, 1, 1, 1
    d = D.declustering_automatico(T, theta=0.5)  # C − 1 = ⌊3.5⌋ = 3; T_(3) = 5 > T_(4) = 1
    assert (d.C, d.T_C, d.T_C_horas) == (4, 1, 3.0)
    d = D.declustering_automatico(T, theta=2 / 7)  # C − 1 = 2, pero T_(2) = T_(3) = 5: se baja a C − 1 = 1
    assert (d.C, d.T_C) == (2, 5)
    assert D.declustering_automatico(T, theta=0.01).T_C == 8  # C − 1 = 0: una tormenta, T_(C) = máximo
    assert D.declustering_automatico(T, theta=1.0).T_C == 0  # todas separadas


def test_intervalos_retenidos_y_n_int():
    exc = np.zeros(12, bool)
    exc[[0, 2, 5, 7, 11]] = True
    cond = np.ones(12, bool)
    cond[4:6] = False  # la excedencia en 5 queda fuera y el intervalo 2 → 7 cruza la pausa
    T, n = D.intervalos(exc, cond)
    assert T.tolist() == [2, 4] and n == 4  # 0→2 y 7→11; 2→7 descartado
    d = D.declustering_automatico(T, n_exc=n)
    assert d.n_int == 2  # n_int = intervalos retenidos (v5, S-19), no N − 1 = 3


@pytest.mark.parametrize("theta", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("q", [0.95, 0.99])
def test_theta_recupera_proceso_ferro_segers(theta, q):
    """§8: 200 réplicas de largo 136 360; media a ±0.02 y P5–P95 dentro de ±0.07 del θ verdadero."""
    est = []
    for i in range(200):
        x = S.ferro_segers(S.LARGO, theta, semillas.generador("prueba", "fs", str(theta), str(i)))
        u = np.quantile(x, q)
        T, _ = D.intervalos(x > u, np.ones(x.size, bool))
        est.append(D.theta_intervalos(T))
    est = np.array(est)
    assert abs(est.mean() - theta) <= 0.02
    p5, p95 = np.quantile(est, [0.05, 0.95])
    assert p5 >= theta - 0.07 and p95 <= theta + 0.07


def test_estacion_sin_agrupamiento():
    """§8 (S-03, la prueba clave): excedencias independientes con tasa sinusoidal (86 % en invierno).

    Con la estimación dentro del invierno, la media de θ̂ en 200 réplicas es ≥ 0.95 (θ = 1). La de todo el año con u
    fijo se calcula para documentar su sesgo, sin tolerancia."""
    inv, anual, frac = [], [], []
    for i in range(200):
        s = S.generar("fs", 1.0, True, semillas.generador("prueba", "estacion", str(i)))
        u = float(np.quantile(s, 0.99))
        bl = D.bloques_anuales(s)
        inv.append(D.declustering_bloques(bl, u, S.INVIERNO).theta)
        anual.append(D.declustering_bloques(bl, u, None).theta)
        d = D.declustering_bloques(bl, u, None)
        frac.append(D.declustering_bloques(bl, u, S.INVIERNO).n_exc / d.n_exc)
    assert np.mean(inv) >= 0.95
    assert abs(np.mean(frac) - S.FRACCION_INVIERNO) < 0.01
    # sesgo documentado (sin tolerancia): con semillas de esta prueba, ≈ 0.97 en invierno y ≈ 0.67 en todo el año
    assert np.mean(anual) < np.mean(inv)


# --- Rachas, tormentas sembradas y corte natural ---------------------------------------------------------


def _tormentas_sembradas(con_huecos: bool):
    """Fondo bajo u = 1; tormentas de 2.0 con duraciones conocidas, algunas con una bajada momentánea.

    Separación mínima (IET) 48 h = 16 pasos. Pares separados por 17 pasos (51 h, se separan) y por 15 (45 h, se
    funden en una). Entre tormentas: 40 a 70 pasos (largos).
    """
    rng = np.random.default_rng(7)
    x = np.full(6000, 0.5)
    pos, n_tormentas, huecos_largos, huecos_cortos, rachas_esperadas = 20, 0, [], [], []
    for k in range(60):
        dur = 3 + k % 5
        if k % 3 == 0:  # tormenta con una bajada momentánea de 2 pasos
            x[pos : pos + dur] = 2.0
            x[pos + dur : pos + dur + 2] = 0.8
            x[pos + dur + 2 : pos + 2 * dur + 2] = 2.0
            rachas_esperadas += [dur, dur]
            huecos_cortos.append(3)  # de la última excedencia a la primera de la siguiente racha
            fin = pos + 2 * dur + 2
        else:
            x[pos : pos + dur] = 2.0
            rachas_esperadas.append(dur)
            fin = pos + dur
        n_tormentas += 1
        if k % 10 == 4:  # par: 17 pasos (se separa) o 15 (se funde)
            sep = 17 if k % 20 == 4 else 15
            x[fin - 1 + sep : fin - 1 + sep + 2] = 2.0
            rachas_esperadas.append(2)
            (huecos_largos if sep == 17 else huecos_cortos).append(sep)
            n_tormentas += 1 if sep == 17 else 0
            fin = fin - 1 + sep + 2
        gap = int(rng.integers(40, 70))
        huecos_largos.append(gap + 1)
        pos = fin + gap
    huecos_largos.pop()  # el último no tiene tormenta siguiente
    x = x[: pos - gap + 10]
    if con_huecos:
        x[pos - gap - 200 : pos - gap - 195] = np.nan  # hueco en el fondo
        x[np.flatnonzero(x == 0.8)[0]] = np.nan  # un hueco dentro de una tormenta no la parte
    return x, n_tormentas, sorted(rachas_esperadas), sorted(huecos_cortos + huecos_largos)


@pytest.mark.parametrize("con_huecos", [False, True])
def test_tormentas_sembradas(con_huecos):
    x, n_tormentas, rachas_esp, huecos_esp = _tormentas_sembradas(con_huecos)
    df = pd.DataFrame({"Hs": x}, index=pd.date_range("2001-01-01", periods=x.size, freq="3h"))
    picos = decluster_rachas(df, u=1.0, iet_horas=48.0)
    assert len(picos) == n_tormentas  # ni partidas ni fundidas
    exc = np.nan_to_num(x, nan=-np.inf) > 1.0
    dur, cortada, huecos = D.rachas(exc, np.ones(x.size, bool))
    if not con_huecos:
        assert sorted(dur.tolist()) == rachas_esp
        assert sorted(huecos.tolist()) == huecos_esp
    assert not cortada.any()
    c = D.corte_natural(huecos * D.PASO_H)
    assert c.corte_h is not None
    # entre las bajadas momentáneas (3 pasos = 9 h) y las separaciones entre tormentas (≥ 15 pasos = 45 h)
    assert 3 * D.PASO_H < c.corte_h < 15 * D.PASO_H


def test_racha_cortada_por_el_borde():
    exc = np.array([1, 1, 0, 0, 1, 1, 0, 1], bool)
    cond = np.array([0, 1, 1, 1, 1, 1, 1, 1], bool)
    dur, cortada, huecos = D.rachas(exc, cond)
    assert dur.tolist() == [1, 2, 1]
    assert cortada.tolist() == [True, False, True]  # vecina fuera de cond con excedencia; borde del bloque
    assert huecos.tolist() == [3, 2]


def test_corte_natural_tres_poblaciones_y_unimodal():
    rng = np.random.default_rng(3)
    h = np.r_[
        rng.lognormal(np.log(6), 0.2, 300), rng.lognormal(np.log(90), 0.2, 200), rng.lognormal(np.log(3000), 0.2, 100)
    ]
    c = D.corte_natural(h)
    assert c.mas_de_dos_modas and c.n_modas >= 3
    assert 10 < c.corte_h < 60  # primer mínimo a la derecha de la primera moda
    assert D.corte_natural(rng.lognormal(np.log(50), 0.5, 400)).corte_h is None
    # segunda moda con menos del 10 %: sin corte
    assert (
        D.corte_natural(np.r_[rng.lognormal(np.log(6), 0.2, 500), rng.lognormal(np.log(300), 0.1, 30)]).corte_h is None
    )


# --- Extremograma ----------------------------------------------------------------------------------------


def test_extremograma_independencia_y_dependencia():
    rng = np.random.default_rng(11)
    t = pd.date_range("1980-01-01", periods=S.LARGO, freq="3h")
    s = pd.Series(rng.random(S.LARGO), index=t)
    u = float(np.quantile(s, 0.99))
    ex = D.extremograma_bloques(D.bloques_anuales(s), u, None)
    n_exc = int((s > u).sum())
    banda = 4 * np.sqrt(0.01 * 0.99 / n_exc)
    assert np.all(np.abs(ex.rho[1:] - 0.01) < banda)
    fs = S.generar("fs", 0.5, False, np.random.default_rng(12))
    ex2 = D.extremograma_bloques(D.bloques_anuales(fs), float(np.quantile(fs, 0.99)), None)
    assert ex2.rho[1] > 0.4  # ρ(3 h) ≈ 1 − θ


def test_referencia_estacional_a_mano():
    """Dos meses: enero con tasa 0 y febrero con tasa 1/2; una excedencia en enero cuyo t + k cae en febrero."""
    t = pd.date_range("2001-01-31 18:00", periods=6, freq="3h")  # 18, 21 (ene) y 00..09 (feb)
    x = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    b = D.Bloque(2001, x, t.month.to_numpy(), D.dia_fijo(t))
    c = D.conteos_extremograma(b, 1.0, np.isin(b.mes, [1]), kmax=2)
    # meses: p_ene = 1/2, p_feb = 2/4. Condicionante: t = 0 (ene). k=1 → ene; k=2 → feb
    e = D.extremograma(c)
    assert e.rho[1] == 0.0 and e.rho[2] == 1.0
    assert e.ref[1] == pytest.approx(0.5) and e.ref[2] == pytest.approx(0.5)
    # solo t en el mes condicionante: la excedencia de febrero no condiciona
    assert c.N.sum(axis=1)[0] == 1


def test_escala_extremograma_ventana_truncada_y_cruce():
    ref = np.full(41, 0.01)
    rho = np.r_[1.0, np.zeros(40)]
    assert D.escala_extremograma(rho, ref)[0] == 3.0  # ρ(3 h) bajo el nivel → 3 h; ρ(0) no entra
    rho = np.r_[1.0, np.where(np.arange(1, 41) <= 10, 0.5, 0.0)]  # alto hasta k = 10
    # media de ±4 pasos ≤ 0.015 recién cuando la ventana deja atrás k = 10: k − 4 > 10 → k = 15
    esc, ultimo = D.escala_extremograma(rho, ref)
    assert esc == 15 * 3.0 and ultimo == 10 * 3.0


def test_extremograma_condicionado_al_invierno():
    t = pd.date_range("2001-01-01", "2001-12-31 21:00", freq="3h")
    x = np.zeros(t.size)
    x[(t.month == 1) & (t.hour == 0)] = 2.0  # excedencias solo en enero
    b = D.Bloque(2001, x, t.month.to_numpy(), D.dia_fijo(t))
    c = D.conteos_extremograma(b, 1.0, D.condicion(b, S.INVIERNO), kmax=8)
    assert c.N.sum() == 0 and c.num.sum() == 0


# --- Bootstrap por bloques y τ ---------------------------------------------------------------------------


def test_bloques_no_crean_intervalos_entre_anios():
    """S-02: al unir años remuestreados no aparece ningún intervalo; el punto es la función sobre los años en orden."""
    a = _bloque(np.r_[np.zeros(10), 2.0], "2001-01-01", 2001)  # excedencia en el último paso
    b = _bloque(np.r_[2.0, np.zeros(5), 2.0], "2002-01-01", 2002)  # en el primero y el séptimo
    d = D.declustering_bloques([a, b], 1.0, None)
    assert d.n_int == 1 and d.n_exc == 3  # solo 0 → 6 dentro de 2002
    d2 = D.declustering_bloques([b, a, b], 1.0, None)  # réplica remuestreada
    assert d2.n_int == 2
    s = pd.Series(np.r_[a.x, b.x], index=pd.date_range("2001-12-30 15:00", periods=18, freq="3h"))
    bl = D.bloques_anuales(s)
    assert [x.anio for x in bl] == [2001, 2002]


def test_tau_ar1_con_huecos_y_anios():
    s = S.generar("ar1", 48.0, False, np.random.default_rng(5))
    s.iloc[1000:1200] = np.nan
    s.iloc[50_000:50_008] = np.nan
    bl = D.bloques_anuales(s)
    tau = D.tau_bloques(bl)
    assert abs(tau.tau_e_h - 48.0) <= 6.0
    assert 30.0 < tau.tau_int_h < 66.0
    # ACF dentro de cada año y promediada: no depende del orden de los años (no los concatena)
    rng = np.random.default_rng(0)
    orden = rng.permutation(len(bl))
    assert np.allclose(D.tau_bloques([bl[i] for i in orden]).acf, tau.acf, equal_nan=True)


def test_rejilla_incompleta_es_error():
    t = pd.date_range("2001-01-01", periods=10, freq="3h").delete(4)
    with pytest.raises(ValueError, match="rejilla"):
        D.bloques_anuales(pd.Series(np.ones(9), index=t))


# --- Sintéticos y etapa 0 --------------------------------------------------------------------------------


def test_sinteticos_marginales_y_ciclo():
    x = S.ferro_segers(200_000, 0.5, np.random.default_rng(1))
    # marginal estacionaria de la ec. 6 (X₁ = W₁/θ): Fréchet de escala 1/θ, P(X ≤ z) = exp(−1/(θ z))
    for z in (1.0, 5.0, 50.0):
        assert np.mean(x <= z) == pytest.approx(np.exp(-1 / (0.5 * z)), abs=0.01)
    a = S.ar1(200_000, 48.0, np.random.default_rng(2))
    assert np.std(a) == pytest.approx(1.0, abs=0.03)
    assert np.corrcoef(a[:-16], a[16:])[0, 1] == pytest.approx(np.exp(-1), abs=0.03)
    assert S.LARGO == S.eje().size and S.eje()[-1] == pd.Timestamp("2026-08-31 21:00")


def test_razon_de_escalas_casos_construidos():
    assert E0.razon_escalas(48.0, 60.0, 72.0) == pytest.approx(1.5)  # concuerdan
    assert E0.razon_escalas(12.0, None, 120.0) == pytest.approx(10.0)  # no concuerdan, sin corte
    assert E0.razon_escalas(0.0, None, 6.0) == pytest.approx(2.0)  # piso de un paso en T_(C)
    assert np.isnan(E0.razon_escalas(24.0, None, float("nan")))
    assert E0.factor_concordancia([1.0] * 95 + [10.0] * 5) == pytest.approx(np.quantile([1.0] * 95 + [10.0] * 5, 0.95))


def test_replica_etapa0_determinista():
    a = E0.replica("fs", 0.5, True, 3)
    b = E0.replica("fs", 0.5, True, 3)
    assert a == b or all((a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)
    assert a["ref_3h"] > 0.015  # la referencia estacional supera 0.01 con ciclo


@pytest.mark.lento
def test_etapa0_determinista_con_1_y_2_procesos(tmp_path, monkeypatch):
    monkeypatch.setattr(E0, "PROCESOS", (("ar1", 48.0), ("fs", 0.5)))
    salidas = []
    for p in (1, 2):
        raiz = tmp_path / f"datos{p}"
        raiz.mkdir()
        E0.etapa0(raiz, procesos_=p, replicas=3)
        salidas.append((E0.ruta_salida(raiz) / "replicas_etapa0.csv").read_bytes())
    assert salidas[0] == salidas[1]
