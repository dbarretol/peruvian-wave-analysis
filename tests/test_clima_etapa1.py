"""F6, etapa 1 de extremo a extremo con un WAVE-DATA sintético (contrato §8, «Etapa»; §6.1 punto 1; G1, G4, G7b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.clima import estratificacion as E
from peruvian_wave_analysis.clima import etapa1 as E1
from peruvian_wave_analysis.clima import insumos as I
from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.comun.errores import InsumoFaltante
from peruvian_wave_analysis.datos import config as CD
from peruvian_wave_analysis.datos.run import ruta_candidatos, ruta_serie

NOMBRE = I.NOMBRE


def _publicar(raiz, salida, archivos: dict, codigo):
    with etapa.staging_protegido(salida, [codigo], base_codigo=codigo.parent) as st:
        for nombre, escribir in archivos.items():
            escribir(st / nombre)
        etapa.publicar(st, salida, etapa.Gate(True, {}), [codigo], {}, raiz, None, base_codigo=codigo.parent)


def _serie():
    rng = np.random.default_rng(3)
    t = pd.date_range("1980-01-01", "2026-08-31 21:00", freq="3h", name="tiempo")
    n = t.size
    esc = 1.6 + 0.4 * np.cos(2 * np.pi * (t.month.to_numpy() - 7) / 12)
    z = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        z[i] = 0.97 * z[i - 1] + 0.243 * e[i]
    hs = np.round(esc * np.exp(0.35 * z), 2)
    # dirección: régimen de 210° y, por semanas al azar (≈ 25 %), de 300°
    semana = (np.arange(n) // 56).astype(int)
    otro = rng.random(semana.max() + 1) < 0.25
    vped = np.where(otro[semana], 300.0, 210.0) + rng.normal(0, 6, n)
    df = pd.DataFrame(index=t)
    for v in CD.VARIABLES:
        df[v] = np.nan
    df["VHM0"] = hs
    df["VHM0_SW1"] = np.round(hs * 0.9, 2)
    df["VHM0_WW"] = np.round(hs * 0.3, 2)
    df["VHM0_SW2"] = np.round(hs * 0.2, 2)
    df["VTPK"] = np.round(12 + 2 * z, 2)
    df["VTM02"] = np.round(df["VTPK"] * 0.7, 2)
    df["VPED"] = vped % 360
    for q in ("WW", "SW1", "SW2"):
        df[f"marca_particion_ausente_{q}"] = False
    df["marca_tp_saturado"] = False
    df["valido"] = True
    df.loc[t[5000:5100], "valido"] = False  # un hueco
    df.loc[~df["valido"], list(CD.VARIABLES)] = np.nan
    return df


def _sembrar(raiz):
    df = _serie()
    codigo = raiz.parent / "codigo"
    codigo.mkdir(exist_ok=True)
    (codigo / "m.py").write_text("X = 1\n", encoding="utf-8")
    _publicar(raiz, ruta_serie(raiz, NOMBRE), {"serie_nodo.parquet": lambda p: df.to_parquet(p)}, codigo)
    _publicar(
        raiz,
        ruta_candidatos(raiz, NOMBRE),
        {"candidatos.csv": lambda p: pd.DataFrame({"tiempo": ["2017-01-01 00:00"], "regla": ["detector"]}).to_csv(
            p, index=False)},
        codigo,
    )  # fmt: skip
    r4 = {"escenarios_hs": [
        {"nombre": "E0", "periodo": ["1980-01-01", "2026-08-31"]},
        {"nombre": "E1", "periodo": ["1993-02-01", "2026-08-31"]},
        {"nombre": "E2", "fecha_escalon": "2016-03-01", "periodo": ["1980-01-01", "2026-08-31"]},
    ]}  # fmt: skip
    _publicar(raiz, I.ruta_f04(raiz, NOMBRE), {"resumen_f04.json": lambda p: json_estricto.escribir(p, r4)}, codigo)
    hs = df["VHM0"]
    corr = pd.DataFrame({"VHM0": hs, "VHM0_cuantiles_asimilado": hs * 1.03, "VHM0_cuantiles_toda": hs * 1.05})
    _publicar(
        raiz,
        I.ruta_f05(raiz, NOMBRE),
        {
            "serie_corregida.parquet": lambda p: corr.to_parquet(p),
            "resumen_f05.json": lambda p: json_estricto.escribir(p, {"escenarios": {"E0": "activo"}}),
            "correccion.json": lambda p: json_estricto.escribir(p, {"variantes": []}),
        },
        codigo,
    )


@pytest.fixture
def chico(monkeypatch):
    """Frontera y réplicas reducidas (la lógica es la misma; los números no son los de la corrida)."""
    monkeypatch.setattr(E, "N_GRANDE", 3000)
    monkeypatch.setattr(E, "SEMILLAS_FRONTERA", 2)
    monkeypatch.setattr(E, "RS", (1.5, 2.5, 4.0))
    monkeypatch.setattr(E, "PASOS_XI_ARRIBA", (0.1, 0.3))
    monkeypatch.setattr(E, "PASOS_XI_ABAJO", (-0.2,))
    monkeypatch.setattr(E1, "R_POTENCIA", 60)
    monkeypatch.setattr(E1, "TROZO", 30)


def test_insumo_faltante_es_error(tmp_path):
    raiz = tmp_path / "datos"
    raiz.mkdir()
    with pytest.raises(InsumoFaltante):
        E1.etapa1(raiz, procesos_=0)


def test_ramas_y_dudosos():
    t = pd.date_range("1990-01-01", periods=10, freq="3h")
    s = pd.DataFrame({"VHM0": np.arange(10.0)}, index=t)
    c = pd.DataFrame({"VHM0_cuantiles_toda": np.arange(10.0) * 2, "VHM0_cuantiles_asimilado": np.arange(10.0) * 3},
                     index=t)  # fmt: skip
    s.iloc[2, 0] = np.nan
    r = I.armar_ramas(s, c, t[4])
    assert list(r) == list(I.RAMAS) and len(r["E1_sin_corregir"]) == 6 and r["E1_cuantiles"].iloc[0] == 12.0
    assert np.isnan(r["E0_cuantiles"].iloc[2])  # un registro no válido sigue sin valor en la rama corregida
    picos = pd.DataFrame({"t_inicio": [t[0], t[5]], "t_fin": [t[2], t[6]]})
    assert list(I.marcar_dudosos(picos, pd.DatetimeIndex([t[1]]))) == [True, False]
    assert list(I.marcar_dudosos(picos, pd.DatetimeIndex([t[6]]))) == [False, True]


def test_etapa1_publica_sin_leer_excesos_y_se_reanuda(tmp_path, chico, capsys):
    raiz = tmp_path / "datos"
    _sembrar(raiz)
    salida = E1.etapa1(raiz, procesos_=0)
    r = json_estricto.leer(salida / "estratos_etapa1.json")
    assert r["excesos_leidos"] is False
    assert r["tau"]["separacion_provisional_h"] % 3 == 0 and r["tau"]["separacion_provisional_h"] >= 48
    assert set(r["conteos"]) == set(I.RAMAS)
    assert r["bimodalidad_vped"]["candidata"] and r["alfa_fam"]["k"] == 2
    assert {c["candidata"] for c in r["candidatas"]} == {"B", "C"}
    assert r["alfa_fam"]["valor"] in E.ALFAS_FAM
    assert len(r["potencia_sigma"]) == 4  # 2 candidatas × 2 bases
    for c in r["candidatas"]:
        assert c["n1"] + c["n2"] == c["N"] == r["conteos"][I.RAMA_DECISION]["n_picos"]
    # E1 es más corta: menos picos
    assert r["conteos"]["E1_sin_corregir"]["n_picos"] < r["conteos"]["E0_sin_corregir"]["n_picos"]
    man = etapa.exigir_publicada(salida)
    assert set(man["insumos"]) == {"serie_nodo", "serie_corregida", "resumen_f05", "correccion", "resumen_f04",
                                   "candidatos"}  # fmt: skip
    salida_txt = capsys.readouterr().out
    assert "min]" in salida_txt and "tareas" in salida_txt  # avance en pantalla
    # misma corrida otra vez: mismo resultado (semillas por nombre)
    r2 = json_estricto.leer(E1.etapa1(raiz, procesos_=0) / "estratos_etapa1.json")
    for k in ("conteos", "fronteras", "potencia_sigma", "alfa_fam", "g7b"):
        assert r2[k] == r[k]


def test_g7b_detiene_si_alguna_potencia_llega_al_80():
    forma = [
        {"candidata": "B", "xi1": 0.0, "tipo": "xi+", "xi2": 0.2, "ad_alfa_fam": 0.3, "lrt_alfa_fam": 0.81},
        {"candidata": "B", "xi1": -0.1, "tipo": "xi+", "xi2": 0.1, "ad_alfa_fam": 0.2, "lrt_alfa_fam": 0.5},
    ]
    d = E1.detiene_g7b(forma)
    assert len(d) == 1 and "lrt_alfa_fam" in d[0]
    assert E1.detiene_g7b(forma[1:]) == []


def test_candidatas_y_g4():
    cont = {
        "E0_sin_corregir": {"anios": 46.0, "n_picos": 200, "B": {"invierno": 170, "verano": 30},
                            "C": {"sector_mayor": 150, "sector_menor": 50}},
        "E1_sin_corregir": {"anios": 33.0, "n_picos": 140, "B": {"invierno": 120, "verano": 20}},
    }  # fmt: skip
    c = E1.candidatas(cont)
    assert [x["candidata"] for x in c] == ["B", "C"]
    assert (
        c[0]["estrato_chico"] == "verano"
        and c[0]["f"] == pytest.approx(0.15)
        and c[0]["lambda"] == pytest.approx(200 / 46)
    )
    assert E1.g4(cont) == ["E1_sin_corregir, B, verano: 20 picos (< 30)"]
