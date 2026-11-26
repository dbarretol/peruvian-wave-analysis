"""F7: sesgo de R1b y referencia de la sensibilidad (ADR-0037; contrato §3.4 y §5.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.eventos import sesgo_regla as SR


def _fila(celda, regla, i, *, k, u, n, rl, err, sust=None, sigma=0.35, caso="A", forma=0.0, n0=334):
    return {"celda": celda, "caso": caso, "forma": forma, "N0": n0, "sigma": sigma, "replica": i, "regla": regla,
            "k_usado": k, "u_usado": u, "n_usado": n, "rl100": rl, "error_log": err, "sustituto": sust}  # fmt: skip


def _df(filas):
    df = pd.DataFrame(filas)
    df["con_sustituto"] = df["sustituto"].notna()
    return df


def test_sesgo_por_celda_media_y_error_tipico():
    errs = [-0.02, -0.04, 0.0, -0.02]
    df = _df([_fila("A_0_N334", "R1b", i, k=12, u=3.2, n=80, rl=4.5, err=e) for i, e in enumerate(errs)])
    s = SR.sesgo_por_celda(df).iloc[0]
    assert s["sesgo"] == pytest.approx(-0.02)
    assert s["error_tipico"] == pytest.approx(np.std(errs, ddof=1) / 2)
    assert s["k_medio"] == 12 and s["excedencias_medias"] == 80
    # otra σ no entra
    assert SR.sesgo_por_celda(_df([_fila("A_0_N334_s0.5", "R1b", 0, k=1, u=1, n=1, rl=1, err=0, sigma=0.5)])).empty


def test_diferencias_con_signo_y_exclusion_por_sustituto():
    filas = []
    for i in range(4):
        filas.append(_fila("A_0_N334", "R1b", i, k=12, u=3.2, n=80, rl=4.0, err=0,
                           sust="k10" if i == 3 else None))  # fmt: skip
        filas.append(_fila("A_0_N334", "R2", i, k=3, u=2.8, n=250, rl=4.0 * (1 + 0.1 * i), err=0,
                           sust="k10" if i == 2 else None))  # fmt: skip
    d = SR.diferencias(_df(filas), "R2")
    assert d["d_rl"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert d["delta_u_m"].tolist() == pytest.approx([-0.4] * 4)
    assert d["comparable"].tolist() == [True, True, False, False]
    ref = SR.referencia(d)
    assert ref["replicas_comparables"] == 2
    assert ref["fraccion_excluida_solo_por_la_comparada"] == pytest.approx(0.25)
    assert ref["fraccion_excluida_por_r1b"] == pytest.approx(0.25)
    assert ref["d_rl"] == pytest.approx([np.quantile([0, 0.1], 0.025), np.quantile([0, 0.1], 0.975)])


def test_referencia_junta_celdas_y_es_asimetrica():
    rng = np.random.default_rng(0)
    filas = []
    for c, n0 in (("A_0_N334", 334), ("A_0_N685", 685)):
        for i in range(400):
            d = rng.exponential(0.05) - 0.01  # cola derecha larga
            filas.append(_fila(c, "R1b", i, k=12, u=3.2, n=80, rl=4.0, err=0, n0=n0))
            filas.append(_fila(c, "R2", i, k=3, u=2.8, n=250, rl=4.0 * (1 + d), err=0, n0=n0))
    ref = SR.referencia(SR.diferencias(_df(filas), "R2"))
    lo, hi = ref["d_rl"]
    assert hi > abs(lo)  # con signo: no simétrico
    assert set(ref["por_celda_descriptivo"]) == {"A_0_N334", "A_0_N685"}
    assert ref["replicas_comparables"] == 800


def test_ubicar():
    iv = [-0.1, 0.16]
    assert SR.ubicar(0.0, iv) == "dentro" and SR.ubicar(-0.1, iv) == "dentro"
    assert SR.ubicar(-0.2, iv) == "bajo" and SR.ubicar(0.2, iv) == "sobre"
    assert SR.ubicar(None, iv) == SR.NO_COMPARABLE and SR.ubicar(float("nan"), iv) == SR.NO_COMPARABLE


def test_seleccion_a_mano():
    filas = []
    # k de R1b: 5, 5, 15, 15; error de k10: 0, 0, -0.02, -0.02; R1b: -0.01, -0.01, -0.05, -0.05
    for i, (k, e10, e1b) in enumerate([(5, 0, -0.01), (5, 0, -0.01), (15, -0.02, -0.05), (15, -0.02, -0.05)]):
        filas.append(_fila("A_0.1_N334", "R1b", i, k=k, u=3, n=80, rl=4, err=e1b))
        filas.append(_fila("A_0.1_N334", "k10", i, k=10, u=3, n=100, rl=4, err=e10))
    s = SR.seleccion(_df(filas))
    assert s["tramo_k"].tolist() == ["(4, 8]", "(12, 16]"]
    assert s["error_r1b"].tolist() == pytest.approx([-0.01, -0.05])
    assert s["error_k10"].tolist() == pytest.approx([0.0, -0.02])
    assert s["correlacion_k_r1b_error_k10"].iloc[0] == pytest.approx(-1.0)
    assert s["r1b_menos_k10_k_bajo_10"].iloc[0] == pytest.approx(-0.01)
    assert s["r1b_menos_k10_k_sobre_10"].iloc[0] == pytest.approx(-0.03)
    assert s["replicas_k_bajo_10"].iloc[0] == 2


def test_sesgo_regla_de_extremo_a_extremo(tmp_path, monkeypatch):
    raiz = tmp_path / "WAVE-DATA"
    sim = SR.ruta_simulacion(raiz)
    sim.mkdir(parents=True)
    rng = np.random.default_rng(1)
    filas = []
    for forma in (-0.1, 0.0, 0.1):
        for n0 in (334, 685):
            c = f"A_{forma:g}_N{n0}"
            for i in range(30):
                rl = 4.5 * np.exp(rng.normal(0, 0.1))
                for regla, k, u in (("R1b", 13, 3.3), ("R2", 3, 2.85), ("R3", 8, 3.05), ("k1", 1, 2.75),
                                    ("k10", 10, 3.15)):  # fmt: skip
                    filas.append(_fila(c, regla, i, k=k, u=u, n=100, rl=rl * (1 + 0.01 * k), err=0.01 * k,
                                       forma=forma, n0=n0))  # fmt: skip
    # una celda de la sensibilidad de σ: no entra en la referencia
    filas += [_fila("A_0_N334_s0.5", r, 0, k=5, u=3.0, n=90, rl=9.0, err=0.5, sigma=0.5) for r in ("R1b", "R2", "R3")]
    pd.DataFrame(filas).to_csv(sim / "replicas.csv", index=False)
    (sim / "MANIFEST.json").write_text("{}", encoding="utf-8")
    original = etapa.exigir_publicada
    monkeypatch.setattr(etapa, "exigir_publicada", lambda p: None if p == sim else original(p))
    salida = SR.ejecutar(raiz)
    original(salida)
    ref = json_estricto.leer(salida / "referencia_sensibilidad.json")
    assert len(ref["celdas"]) == 6 and set(ref["comparadas"]) == {"R2", "R3"}
    assert ref["comparadas"]["R2"]["replicas_comparables"] == 180
    assert ref["comparadas"]["R2"]["delta_u_m"] == pytest.approx([-0.45, -0.45])
    for f in ("sesgo_por_celda.csv", "seleccion.csv", "diferencias.csv"):
        assert (salida / f).exists()
