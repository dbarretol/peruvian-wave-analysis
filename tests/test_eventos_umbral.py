"""F7: rejilla de umbrales, empates, R1, R2 y R3 (contrato de la F7, §3.4 y §8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import semillas
from peruvian_wave_analysis.eventos import r2
from peruvian_wave_analysis.eventos import umbral as U
from peruvian_wave_analysis.extremos import gpd


def _picos(n, xi=0.0, sigma=0.35, u_id=2.7, semilla=1):
    rng = np.random.default_rng(semilla)
    return np.round(u_id + gpd.cuantil_supervivencia(rng.random(n), xi, sigma), 2)


# --- Rejilla y empates (C-09) ---------------------------------------------------------------------


def test_objetivos_geometricos_y_30_arriba():
    nk = U.objetivos(685)
    assert nk.size == 20 and np.all(np.diff(nk) < 0)
    assert nk[-1] == round(685 / 20) and nk[0] == round(685 * 20 ** (-1 / 20))
    assert U.objetivos(334)[-1] == 30  # r = N0/30 deja 30 picos arriba
    with pytest.raises(ValueError):
        U.objetivos(30)


def test_rejilla_con_empates_conteo_estricto():
    p = _picos(400)
    assert np.unique(p).size < p.size  # hay picos repetidos con la resolución de 0.01 m
    rej = U.rejilla(p)
    d0 = U.valores_distintos(p)
    medios = (d0[:-1] + d0[1:]) / 2
    for u, n in zip(rej.u, rej.n_real, strict=True):
        assert int(np.sum(p > u)) == n
        assert np.all(np.abs(p - u) > 1e-9)  # ningún pico sobre el umbral: ningún exceso nulo
        assert np.min(np.abs(medios - u)) < 1e-12  # punto medio entre dos valores distintos consecutivos
    assert np.all(np.diff(rej.u) > 0) and np.all(rej.n_real >= 30)
    # el n real es el más cercano posible al objetivo
    d = U.valores_distintos(p)
    cuentas = np.array([np.sum(p > m) for m in (d[:-1] + d[1:]) / 2])
    for n_obj, n in zip(rej.n_objetivo, rej.n_real, strict=True):
        assert abs(n - n_obj) == np.min(np.abs(cuentas - n_obj))


def test_rejilla_empate_elige_el_mayor_numero():
    # valores distintos con conteos 10, 12, ...: objetivo 11 queda a igual distancia → 12
    p = np.repeat(np.round(np.arange(4.0, 2.7, -0.01), 2), 2)
    p[0] = 4.5  # el mayor solo
    d = U.valores_distintos(p)
    cuentas = np.array([np.sum(p > m) for m in (d[:-1] + d[1:]) / 2])
    assert 10 in cuentas and 12 in cuentas and 11 not in cuentas


def test_rejilla_con_repetidos_marca():
    p = np.concatenate([np.full(200, 2.75), np.round(np.linspace(2.8, 4.0, 60), 2)])
    rej = U.rejilla(p)
    assert rej.con_repetidos and len(rej.k_repetidos) > 0
    assert np.unique(rej.u).size == rej.u.size


# --- Ajuste en cada umbral ------------------------------------------------------------------------


def test_ajuste_en_umbral_y_rl1_bajo_el_umbral():
    p = _picos(300)
    a = U.ajustar_en(p, 3.5, anios=46.0)
    assert a.valido and a.tasa_por_anio < 1
    assert not a.rl[1.0].existe and a.rl[1.0].causa == "bajo_el_umbral"  # no es un fallo
    assert np.isfinite(a.rl_valor(100.0))
    assert a.sigma_mod == pytest.approx(a.ajuste.sigma - a.ajuste.xi * 3.5)


# --- R1 --------------------------------------------------------------------------------------------


def test_r1_acotada():
    rl = np.array([5.0, 4.8, 4.6, 4.50, 4.51, 4.52, 4.53, 4.70, 4.9, 5.1])
    a, b = U.regla_r1(rl, "R1a"), U.regla_r1(rl, "R1b")
    assert a.ventana == (3, 6) and a.indice == 4  # largo par: el más bajo de los dos centrales
    assert b.ventana == (3, 6) and b.indice == 3
    assert a.estado == U.CON_RESULTADO


def test_r1_abarca_todo_y_sin_tramo():
    assert U.regla_r1(np.full(10, 4.6), "R1a").estado == U.NO_DISCRIMINO
    r = U.regla_r1(np.full(10, 4.6), "R1b")
    assert r.estado == U.NO_DISCRIMINO and r.indice == 0
    assert U.regla_r1(4.0 * 1.05 ** np.arange(10), "R1a").estado == U.SIN_RESULTADO


def test_r1_dos_tramos_a_distinta_altura():
    # tramo largo abajo (0–4) y tramo corto arriba (7–9)
    rl = np.array([4.5, 4.5, 4.51, 4.52, 4.52, 4.8, 5.0, 5.2, 5.21, 5.22])
    a, b = U.regla_r1(rl, "R1a"), U.regla_r1(rl, "R1b")
    assert a.ventana == (0, 4) and a.indice == 2
    assert b.ventana == (7, 9) and b.indice == 7


def test_r1_empate_de_largo_toma_el_mas_bajo():
    rl = np.array([4.5, 4.5, 4.5, 4.8, 5.0, 5.2, 5.2, 5.2, 6.0, 7.0])
    assert U.regla_r1(rl, "R1a").ventana == (0, 2)
    assert U.regla_r1(rl, "R1b").ventana == (5, 7)


def test_r1_rl_nulo_corta_el_tramo():
    rl = np.full(10, 4.6)
    rl[4] = np.nan
    t = U.tramos_estables(rl)
    assert (0, 3) in t and (5, 9) in t and all(not (i <= 4 <= j) for i, j in t)
    assert U.regla_r1(rl, "R1a").ventana == (5, 9)


# --- R3 --------------------------------------------------------------------------------------------


def test_a2r_contra_la_formula():
    x = np.array([0.1, 0.3, 0.7, 1.4])
    z = 1 - np.exp(-x / 0.5)
    i = np.arange(1, 5)
    esperado = 4 / 2 - 2 * z.sum() - np.sum((2 - (2 * i - 1) / 4) * np.log(1 - z))
    assert U.a2_cola_derecha(x, 0.0, 0.5) == pytest.approx(esperado, rel=1e-12)


def test_replicas_redondeadas_sin_valores_bajo_el_umbral():
    e = U.redondear_replica(2.705, np.array([0.0001, 0.004, 0.006, 0.31, 1.234]))
    h = 2.705 + e
    np.testing.assert_allclose(h, [2.71, 2.71, 2.71, 3.02, 3.94], atol=1e-9)
    assert np.all(e > 0)


def test_tolerancia_r3_con_piso():
    assert U.tolerancia_r3(1.0) == pytest.approx(2 * np.sqrt(0.0475 / 99))
    assert U.tolerancia_r3(1.0) == pytest.approx(0.0438, abs=1e-3)
    assert U.tolerancia_r3(0.5) == pytest.approx(2 * np.sqrt(0.25 / 99))


def test_regla_r3_empate_por_monte_carlo_y_sin_resultado():
    p = np.array([0.30, 0.52, 0.58, 0.60, 0.61, 0.40])
    tol = U.tolerancia_r3(0.61)
    r = U.regla_r3(p)
    assert r.indice == int(np.flatnonzero(p >= 0.61 - tol)[0]) == 1
    assert U.regla_r3(np.array([0.02, 0.05, 0.09])).estado == U.SIN_RESULTADO
    assert U.regla_r3(np.array([np.nan, 1.0, 0.97])).indice == 1


def test_p_valor_r3_formula_y_semilla():
    p = _picos(200)
    u = 2.805
    exc = p[p > u] - u
    a = gpd.ajustar_gpd(exc)
    r1 = U.p_valor_r3(exc, a, u, semillas.generador("prueba", "r3"), B=19)
    r2_ = U.p_valor_r3(exc, a, u, semillas.generador("prueba", "r3"), B=19)
    assert r1 == r2_ and r1.p == pytest.approx((1 + r1.m) / 20)


# --- R2 --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2])
def test_r2_informacion_de_la_gpd_con_un_tramo(xi):
    """m = 1: la información esperada de la GPD (Smith, 1985), calculada a mano."""
    s = 0.4
    info = r2.informacion_esperada(np.array([0.0]), xi, s)
    c = 1 / ((1 + xi) * (1 + 2 * xi))
    esperado = np.array([[1 / (s * s * (1 + 2 * xi)), c / s], [c / s, 2 * c]])
    np.testing.assert_allclose(info, esperado, rtol=1e-8)


@pytest.mark.parametrize("xi", [-0.2, 0.1])
def test_r2_informacion_por_tramos_frente_a_monte_carlo(xi):
    v, s = np.array([0.0, 0.2, 0.5]), 0.35
    info = r2.informacion_esperada(v, xi, s)
    y = gpd.cuantil_supervivencia(np.random.default_rng(1).random(400_000), xi, s)
    u = r2.score_por_observacion(y, v, np.r_[s, [xi] * 3])
    mc = u.T @ u / y.size
    assert np.max(np.abs(info - mc)) / np.abs(mc).max() < 0.02
    assert np.abs(u.mean(axis=0)).max() < 0.02 * np.sqrt(np.diag(mc)).max()  # E[u] = 0 bajo H0


def test_r2_score_contra_diferencias_finitas_y_continuidad():
    v = np.array([0.0, 0.3])
    phi = np.array([0.4, 0.1, -0.05])
    y = np.array([0.05, 0.2, 0.31, 0.9])
    u = r2.score_por_observacion(y, v, phi)
    h = 1e-6
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        fd = (r2.log_densidad(y, v, phi + e) - r2.log_densidad(y, v, phi - e)) / (2 * h)
        np.testing.assert_allclose(u[:, k], fd, rtol=1e-5, atol=1e-7)
    # densidad continua en v_2 y que integra 1
    f = np.exp(r2.log_densidad(np.array([0.3 - 1e-9, 0.3 + 1e-9]), v, phi))
    assert f[0] == pytest.approx(f[1], rel=1e-6)
    yy = np.linspace(1e-9, 60, 600_001)
    assert np.trapezoid(np.exp(r2.log_densidad(yy, v, phi)), yy) == pytest.approx(1.0, abs=2e-3)


def test_r2_caso_de_dos_tramos_calculado_a_mano():
    """m = 2: S con U e i calculados aparte (diferencias finitas de la log-verosimilitud total y Monte Carlo)."""
    y = np.sort(gpd.cuantil_supervivencia(np.random.default_rng(7).random(300), 0.05, 0.4))
    v = np.array([0.0, 0.4])
    t = r2.prueba_score(y, v)
    a = gpd.ajustar_gpd(y)
    phi = np.array([a.sigma, a.xi, a.xi])
    h = 1e-6
    U_ = np.array([(r2.log_densidad(y, v, phi + h * e).sum() - r2.log_densidad(y, v, phi - h * e).sum()) / (2 * h)
                   for e in np.eye(3)])  # fmt: skip
    assert abs(U_[0]) < 1e-3 and abs(U_[1] + U_[2]) < 1e-3  # en el estimador restringido, U_σ = 0 y U_ξ1 + U_ξ2 = 0
    z = gpd.cuantil_supervivencia(np.random.default_rng(8).random(400_000), a.xi, a.sigma)
    uu = r2.score_por_observacion(z, v, phi)
    info = y.size * uu.T @ uu / z.size
    S = float(U_ @ np.linalg.solve(info, U_))
    assert t.evaluable and t.gl == 1
    assert t.S == pytest.approx(S, rel=0.03)


def test_r2_no_evaluable_rompe_la_regla():
    ok = r2.PruebaScore(1.0, 3, 0.5, 0.1, 0.4, True)
    malo = r2.PruebaScore(np.nan, 3, np.nan, -0.6, 0.4, False, "xi_no_regular")
    rech = r2.PruebaScore(9.0, 3, 0.01, 0.1, 0.4, True)
    r = r2.regla_r2([rech, ok, ok, ok])
    assert r.indice == 2 and r.ventana == (2, 6)  # v_2 es el u_k de posición 2
    assert r2.regla_r2([ok, ok, malo, ok]).indice == 6
    assert r2.regla_r2([ok, ok, ok, malo]).estado == U.SIN_RESULTADO


def test_r2_xi_no_regular_es_no_evaluable():
    y = gpd.cuantil_supervivencia(np.random.default_rng(2).random(200), -0.7, 0.4)
    t = r2.prueba_score(y, np.array([0.0, 0.1]))
    assert not t.evaluable and t.causa == "xi_no_regular"


def test_r2_j_solo_hasta_9_con_diez_umbrales():
    p = _picos(500)
    rej = U.rejilla(p)
    vs = r2.umbrales_r2(rej.u)
    assert vs.size == 10 and np.all(vs == rej.u[0::2])
    pr = r2.pruebas_r2(p, vs)
    assert len(pr) == 9 and [t.gl for t in pr] == list(range(9, 0, -1))


@pytest.mark.lento
def test_tamano_de_r2_y_r3_con_gpd_exacta(capsys):
    """Tamaño de la prueba de score (χ², ≈ 5 %) y de «p < 0.10» en R3 (≈ 10 %), 500 réplicas, ±3 puntos (§8).

    El artículo encuentra la prueba de score conservadora con χ²: si el tamaño sale bajo, se publica (§3.4)."""
    rech2 = rech3 = 0
    R = 500
    for i in range(R):
        rng = semillas.generador("prueba", "tamano", str(i))
        y = gpd.cuantil_supervivencia(rng.random(250), 0.0, 0.35)
        rech2 += r2.prueba_score(y, np.array([0.0, 0.2, 0.4])).p <= 0.05
        p = np.round(2.7 + y, 2)
        exc = p[p > 2.705] - 2.705
        a = gpd.ajustar_gpd(exc)
        rech3 += U.p_valor_r3(exc, a, 2.705, rng).p < 0.10
        if (i + 1) % 50 == 0:
            with capsys.disabled():
                print(f"  tamaño: {i + 1}/{R} réplicas", flush=True)
    with capsys.disabled():
        print(f"  tamaño R2 = {rech2 / R:.3f}, R3 = {rech3 / R:.3f}", flush=True)
    assert abs(rech2 / R - 0.05) <= 0.03
    assert abs(rech3 / R - 0.10) <= 0.03


# --- R2 frente al apéndice de Northrop y Coleman (score en forma cerrada, p. 13 del manuscrito) --------------------


def _score_apendice(y, v, sigma1, phi):
    """Score de una observación en θ = (σ₁, φ₁, …, φ_m), φ_j = ξ_j/σ_j, transcrito del apéndice (p. 13).

    S_j = Y − v_j, T_j = 1 + φ_j S_j, γ_j = 1 + φ_j w_j, h_j = φ_j Π_{i<j} γ_i, R_j = log T_j / (σ₁ h_j);
    ∂ log σ_j/∂φ_k = w_k/γ_k·I(k ≤ j−1); ∂ log T_j/∂φ_k = S_j/T_j·I(k = j);
    ∂R_j/∂φ_k = σ₁⁻¹{h_j⁻¹[S_j/T_j − log T_j/φ_j]·I(k = j) − w_k/γ_k·I(k ≤ j−1)·log T_j/h_j};
    ∂ log p_j/∂φ_k = −σ₁⁻¹{h_k⁻¹[w_k/γ_k − log γ_k/φ_k]·I(k ≤ j−1) − w_k/γ_k·Σ_{i<j} I(k ≤ i−1)·log γ_i/h_i}.
    """
    m = v.size
    w = np.diff(v)
    gam = 1 + phi[:-1] * w
    h = np.array([phi[j] * np.prod(gam[:j]) for j in range(m)])
    logp = np.array([-sum(np.log(gam[i]) / (sigma1 * h[i]) for i in range(j)) for j in range(m)])
    out = np.zeros((y.size, m + 1))
    for n_, yy in enumerate(y):
        j = int(np.clip(np.searchsorted(v, yy, side="right") - 1, 0, m - 1))
        S = yy - v[j]
        T = 1 + phi[j] * S
        R = np.log(T) / (sigma1 * h[j])
        out[n_, 0] = -(logp[j] + 1 - R) / sigma1
        for k in range(m):
            d_logsig = w[k] / gam[k] if k <= j - 1 else 0.0
            d_logT = S / T if k == j else 0.0
            dR = ((S / T - np.log(T) / phi[j]) / h[j] if k == j else 0.0) - (
                w[k] / gam[k] * np.log(T) / h[j] if k <= j - 1 else 0.0
            )
            dR /= sigma1
            suma = sum(np.log(gam[i]) / h[i] for i in range(j) if k <= i - 1)
            dlogp = -((w[k] / gam[k] - np.log(gam[k]) / phi[k]) / h[k] if k <= j - 1 else 0.0)
            if k < m - 1:  # w_k y γ_k solo existen para k ≤ m − 2 (aparecen con k ≤ j − 1)
                dlogp += (w[k] / gam[k]) * suma
            dlogp /= sigma1
            out[n_, k + 1] = dlogp - d_logsig - d_logT - dR
    return out  # fmt: skip


@pytest.mark.parametrize("xi", [-0.2, 0.15])
def test_r2_igual_al_score_del_apendice(xi):
    """El score del apéndice (θ) y el de r2 (ψ = (σ₁, ξ₁…ξ_m)) dan el mismo S: la prueba es la de Northrop y
    Coleman, no solo un cálculo correcto. Se comprueba también la regla de la cadena entre los dos scores."""
    rng = np.random.default_rng(11)
    y = gpd.cuantil_supervivencia(rng.random(250), xi, 0.4) * (1 + 0.2 * (rng.random(250) > 0.5))  # H0 algo falsa
    v = np.array([0.0, 0.15, 0.35, 0.6])
    a = gpd.ajustar_gpd(y)
    m = v.size
    sig = a.sigma + a.xi * v  # σ_j bajo H0
    phi = a.xi / sig
    # score total del apéndice frente al de r2, transformado con el jacobiano ∂ψ/∂θ (numérico)
    U_theta = _score_apendice(y, v, a.sigma, phi).sum(axis=0)
    psi = np.r_[a.sigma, np.full(m, a.xi)]
    U_psi = r2.score_por_observacion(y, v, psi).sum(axis=0)

    def psi_de_theta(th):
        s1, ph = th[0], th[1:]
        s = [s1]
        for j in range(m - 1):
            s.append(s[-1] * (1 + ph[j] * (v[j + 1] - v[j])))  # σ_{j+1} = σ_j + ξ_j w_j = σ_j(1 + φ_j w_j)
        return np.r_[s1, ph * np.array(s)]

    th0 = np.r_[a.sigma, phi]
    J = np.empty((m + 1, m + 1))
    for k in range(m + 1):
        e = np.zeros(m + 1)
        e[k] = 1e-7
        J[:, k] = (psi_de_theta(th0 + e) - psi_de_theta(th0 - e)) / 2e-7
    np.testing.assert_allclose(U_theta, J.T @ U_psi, rtol=1e-5, atol=1e-6)
    # S con el score del apéndice y su información esperada (misma cuadratura) frente a r2.prueba_score
    yq, wq = r2.nodos_bajo_h0(v, a.xi, a.sigma)
    uq = _score_apendice(yq, v, a.sigma, phi)
    info_theta = y.size * (uq * wq[:, None]).T @ uq
    S_ap = float(U_theta @ np.linalg.solve(info_theta, U_theta))
    t = r2.prueba_score(y, v, a)
    assert t.S == pytest.approx(S_ap, rel=1e-6) and t.gl == m - 1


_DIR_MEV = Path(__file__).parent / "datos" / "r2_mev"


@pytest.mark.skipif(not (_DIR_MEV / "referencia_mev.csv").is_file(), reason="falta referencia_mev.csv (correr el .R)")
def test_r2_frente_a_mev_nc_diag():
    """p de la prueba de score frente a mev::NC.diag (Northrop y Coleman), con los números guardados (ADR-0019)."""
    ref = pd.read_csv(_DIR_MEV / "referencia_mev.csv")
    for i in (1, 2):
        x = pd.read_csv(_DIR_MEV / f"conjunto_{i}.csv")["x"].to_numpy()
        u = pd.read_csv(_DIR_MEV / f"conjunto_{i}_umbrales.csv")["u"].to_numpy()
        pr = r2.pruebas_r2(x, u)
        r = ref.loc[ref["conjunto"] == i]
        assert len(r) == len(pr) == u.size - 1
        assert [t.gl for t in pr] == r["gl"].tolist()
        # mev 2.2 (R 4.6.2) da lo mismo hasta 1e-5; la tolerancia deja margen solo para ξ̂ ≈ 0 (xi.tol de mev)
        np.testing.assert_allclose([t.S for t in pr], r["s_score"], rtol=1e-4)
        np.testing.assert_allclose([t.p for t in pr], r["p_score"], rtol=1e-3, atol=1e-4)
