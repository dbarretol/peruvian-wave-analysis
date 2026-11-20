"""Pruebas de la F6, §8: estratificación (frontera, potencia, α de familia) y reproducción de la simulación previa."""

from __future__ import annotations

import time

import numpy as np
import pytest

from peruvian_wave_analysis.clima import estratificacion as E
from peruvian_wave_analysis.clima import simulacion_previa as SP
from peruvian_wave_analysis.comun import semillas
from peruvian_wave_analysis.extremos.gpd import ajustar_gpd
from peruvian_wave_analysis.extremos.retorno import nivel_retorno
from peruvian_wave_analysis.validacion import reanudable


def test_rl_mezcla_de_un_estrato_y_de_dos_iguales():
    lam = 5.0
    cerrado = nivel_retorno(E.U_SIM, -0.1, 0.5, lam, 100.0).valor
    assert E.rl_mezcla([(lam, -0.1, 0.5)]) == pytest.approx(cerrado, rel=1e-9)
    assert E.rl_mezcla([(2.0, -0.1, 0.5), (3.0, -0.1, 0.5)]) == pytest.approx(cerrado, rel=1e-9)


def test_rejillas_y_parametros():
    assert E.rejilla(E.SIGMA, 0.0)[0] == 1.1 and E.rejilla(E.SIGMA, 0.0)[-1] == 4.0 and len(E.RS) == 30
    assert E.rejilla(E.XI_ARRIBA, -0.1)[:2] == (-0.075, -0.05) and E.rejilla(E.XI_ARRIBA, -0.1)[-1] == 0.2
    assert E.rejilla(E.XI_ABAJO, 0.0)[0] == -0.025 and E.rejilla(E.XI_ABAJO, 0.0)[-1] == -0.2
    assert E.parametros(E.SIGMA, -0.1, 2.0) == (2.0, -0.1)
    assert E.parametros(E.XI_ARRIBA, -0.1, 0.05) == (1.0, 0.05)
    assert E.partir(170, 0.15) == (144, 26)


def test_elegir_frontera():
    fila = [(1.1, [-0.5, -0.7]), (1.2, [-1.9, -2.0]), (1.3, [-2.1, -2.0]), (1.4, [-3.0, -3.0])]
    fr = E.elegir_frontera(E.SIGMA, 0.0, 0.15, fila)
    assert (fr.r, fr.xi2, fr.rotulo) == (1.3, 0.0, "") and fr.media_pct == pytest.approx(-2.05)
    # el signo no importa: cuenta |media|
    assert E.elegir_frontera(E.SIGMA, 0.0, 0.3, [(1.1, [1.0]), (1.2, [2.5])]).r == 1.2
    sin = E.elegir_frontera(E.SIGMA, 0.0, 0.2, [(1.1, [0.1]), (4.0, [1.9])])
    assert sin.r == 4.0 and "sin frontera" in sin.rotulo
    abajo = E.elegir_frontera(E.XI_ABAJO, -0.1, 0.3, [(-0.125, [-1.0]), (-0.3, [-1.5])])
    assert abajo.r is None and abajo.xi2 is None and abajo.rotulo
    arriba = E.elegir_frontera(E.XI_ARRIBA, -0.1, 0.3, [(-0.075, [-0.5]), (-0.05, [-2.2])])
    assert (arriba.r, arriba.xi2) == (1.0, -0.05)


def _pot(a05, a075, a10):
    return {0.05: a05, 0.075: a075, 0.10: a10}


def test_elegir_alfa_fam_casos_construidos():
    todos = {("B", 0.0): _pot(0.85, 0.9, 0.95), ("B", -0.1): _pot(0.81, 0.85, 0.9)}
    e = E.elegir_alfa_fam(todos)
    assert (e.alfa_fam, e.alcanza, e.k, e.bajo_objetivo_con_tope) == (0.05, True, 1, ())
    medio = {**todos, ("C", 0.0): _pot(0.7, 0.8, 0.9), ("C", -0.1): _pot(0.75, 0.82, 0.88)}
    e = E.elegir_alfa_fam(medio)
    assert (e.alfa_fam, e.alcanza, e.k) == (0.075, True, 2)
    nunca = {**todos, ("C", -0.1): _pot(0.6, 0.65, 0.7)}
    e = E.elegir_alfa_fam(nunca)
    assert (e.alfa_fam, e.alcanza) == (0.10, False)
    assert e.bajo_objetivo_con_tope == (("C", -0.1),)
    assert e.por_nivel[0.10] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        E.elegir_alfa_fam({})


def test_xi_comun_entre_el_modelo_unico_y_el_de_xi_por_estrato():
    rng = semillas.generador("prueba", "xi-comun")
    a, b = E.muestra(0.0, 0.0, 2.0, 150, 40, rng)
    co = E.ajustar_xi_comun(a, b)
    a1, a2, au = ajustar_gpd(a), ajustar_gpd(b), ajustar_gpd(np.concatenate([a, b]))
    assert co.valido
    assert au.loglik - 1e-6 <= co.loglik <= a1.loglik + a2.loglik + 1e-6  # modelos anidados
    assert co.sigma2 / co.sigma1 == pytest.approx(2.0, rel=0.35)
    assert E.razon_verosimilitudes(a, b, co) >= 0.0


def test_tasas_de_rechazo():
    assert E.tasa_rechazo([0.01, 0.2, np.nan, 0.04], 0.05) == 0.5
    assert E.rechazo_lrt([5.0, 1.0, np.nan, 4.0], 0.05) == 0.5


def test_potencia_de_anderson_darling_reproduce_la_tabla():
    """§8: con conteos dados reproduce la potencia de la simulación previa (ξ₁ = −0.1, f = 0.15, r = 1.8, N = 170:
    0.79 con α = 0.05 y 0.71 con α = 0.025, 1 000 réplicas)."""
    n1, n2 = E.partir(170, 0.15)
    p = [
        E.replica_pruebas(-0.1, -0.1, 1.8, n1, n2, semillas.generador("prueba", "ad", str(i)), con_lrt=False)["ad_p"]
        for i in range(600)
    ]
    # error de Monte Carlo de la diferencia ≈ √(0.17²/600 + 0.17²/1000) ≈ 0.021; tolerancia ≈ 3.5 σ
    assert E.tasa_rechazo(p, 0.05) == pytest.approx(0.79, abs=0.075)
    assert E.tasa_rechazo(p, 0.025) == pytest.approx(0.71, abs=0.08)


def test_bajo_el_nulo_anderson_darling_tiene_su_tamano():
    n1, n2 = E.partir(230, 0.3)
    p = [
        E.replica_pruebas(0.0, 0.0, 1.0, n1, n2, semillas.generador("prueba", "ad-nulo", str(i)), con_lrt=False)["ad_p"]
        for i in range(600)
    ]
    assert E.tasa_rechazo(p, 0.05) == pytest.approx(0.05, abs=0.03)


def test_criterio_del_minimo():
    def fila(n2, su, sc, au, ac, fallos=0.0):
        f = {"n2": n2, "unico_sesgo_pct": su, "comun_sesgo_pct": sc, "unico_ancho_pct": au, "comun_ancho_pct": ac}
        f["comun_fallos"] = fallos
        f["cumple"] = SP.cumple_minimo(f)
        return f

    assert fila(20, -3, -1, 20, 21.9)["cumple"]
    assert not fila(20, -3, -1, 20, 22.1)["cumple"]  # ancho > 1.1 ×
    assert not fila(20, -1, -1.5, 20, 20)["cumple"]  # más sesgo
    assert not fila(20, -3, -1, 20, 20, fallos=0.02)["cumple"]
    casos = [fila(10, -3, -1, 20, 30), fila(20, -3, -1, 20, 20), fila(30, -3, -1, 20, 20)]
    assert SP.minimo_n2(casos) == 20
    casos.append(fila(40, -1, -2, 20, 20))
    assert SP.minimo_n2(casos) is None


def test_fronteras_por_oleadas_cierran_al_cruzar_y_se_reanudan(tmp_path, monkeypatch):
    """La búsqueda por oleadas da la misma frontera que recorrer la rejilla y no calcula más allá del cruce."""
    llamadas = []

    def falso(xi1, f, r, xi2, lam, rng):
        llamadas.append((f, r, xi2))
        return -10 * (r - 1.0) * f - 0.01 if xi2 == xi1 else 30 * (xi2 - xi1) + 0.01  # lejos de los bordes

    monkeypatch.setattr(E, "delta_grande", falso)
    filas = [(E.SIGMA, 0.0, 0.1), (E.SIGMA, 0.0, 0.5), (E.XI_ARRIBA, 0.0, 0.3)]
    mensajes = []
    fr, rej = SP.fronteras(tmp_path, 0, filas, aviso=mensajes.append)
    # σ con f = 0.1: |−(r−1)| ≥ 2 → r = 3.0; con f = 0.5: 5 (r − 1) ≥ 2 → r = 1.4; ξ+: 30 Δξ ≥ 2 → Δξ = 0.075
    assert [x.r for x in fr[:2]] == [3.0, 1.4] and fr[2].xi2 == pytest.approx(0.075)
    assert len(rej["sigma_0_0.1"]) == 20 and len(rej["sigma_0_0.5"]) == 4 and len(rej["xi+_0_0.3"]) == 3
    assert any("oleada" in m for m in mensajes) and any("tareas" in m for m in mensajes)
    n = len(llamadas)
    fr2, _ = SP.fronteras(tmp_path, 0, filas, aviso=mensajes.append)  # todo guardado: nada se recalcula
    assert len(llamadas) == n and [x.como_dict() for x in fr2] == [x.como_dict() for x in fr]


def test_reanudable_informa_el_avance(tmp_path):
    mensajes = []
    tareas = [(f"t{i}", i) for i in range(40)]
    reanudable.ejecutar(abs, tareas, tmp_path, 0, mensajes.append)
    avances = [m for m in mensajes if "/40 tareas" in m]
    assert len(avances) == 20 and avances[-1].startswith("40/40") and "faltan" in avances[-1]


@pytest.mark.lento
def test_frontera_reproduce_la_simulacion_previa():
    """§8: la frontera recalculada reproduce §5.4 (ξ₁ = 0, f = 0.15: r = 2.1 con −2.88 %, desviación por semilla
    0.45 %). Error de Monte Carlo de la diferencia de medias de 8 semillas ≈ 0.22; tolerancia 3 σ."""
    d = SP.trabajo_frontera((E.SIGMA, 0.0, 0.15, 2.1, SP.LAM))
    assert len(d) == 8
    assert float(np.mean(d)) == pytest.approx(-2.88, abs=0.67)


@pytest.mark.lento
def test_minimo_xi_comun_reproduce_un_caso_de_la_tabla(capsys):
    """§8: un caso de §5.4 (ξ₁ = −0.1, f = 0.15, r = 1.8, N = 320; 1 000 réplicas): sesgo de la mediana del RL100
    −3.58 % sin estratos y −0.53 % con ξ común; ancho 17.3 % y 17.6 %. Error de Monte Carlo de la diferencia de
    medianas ≈ 0.4 puntos; tolerancia 1.2."""
    t0 = time.time()
    reps = []
    for k in range(10):
        reps += SP.trabajo_modelos(("prueba-minimo", "sigma_xi1=-0.1_f=0.15", -0.1, -0.1, 1.8, 320,
                                    list(range(100 * k, 100 * (k + 1))), False))  # fmt: skip
        with capsys.disabled():
            print(f"  minimo xi comun: {100 * (k + 1)}/1000 replicas, {time.time() - t0:.0f} s", flush=True)
    fila = SP.resumen_modelos(-0.1, -0.1, 1.8, 0.15, 320, reps, ("unico", "comun"))
    assert fila["unico_fallos"] == 0 and fila["comun_fallos"] == 0
    assert fila["unico_sesgo_pct"] == pytest.approx(-3.58, abs=1.2)
    assert fila["comun_sesgo_pct"] == pytest.approx(-0.53, abs=1.2)
    assert fila["unico_ancho_pct"] == pytest.approx(17.3, abs=2.0)
    assert fila["comun_ancho_pct"] == pytest.approx(17.6, abs=2.0)
