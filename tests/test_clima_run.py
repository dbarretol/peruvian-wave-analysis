"""F6, etapa principal ``clima`` (contrato §2, §3, §6, §7 y §8, «Etapa» y «Figuras») con un WAVE-DATA sintético."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis import estilo
from peruvian_wave_analysis.clima import escalas as ES
from peruvian_wave_analysis.clima import estratos_sitio as ET
from peruvian_wave_analysis.clima import etapa0 as E0
from peruvian_wave_analysis.clima import etapa1 as E1
from peruvian_wave_analysis.clima import figuras as F
from peruvian_wave_analysis.clima import gev_ramas as GV
from peruvian_wave_analysis.clima import run as R
from peruvian_wave_analysis.comun import etapa, json_estricto, orquestador
from peruvian_wave_analysis.comun.errores import GateFallido
from peruvian_wave_analysis.extremos import dependencia as D

from . import test_clima_etapa1 as T1
from .test_clima_etapa1 import chico  # noqa: F401 — fixture

B_PRUEBA = 20
_SERIE_ORIGINAL = T1._serie


def _serie_unimodal():
    df = _SERIE_ORIGINAL()
    rng = np.random.default_rng(11)
    df["VPED"] = (205.0 + rng.normal(0, 8, len(df))) % 360
    df.loc[~df["valido"], "VPED"] = np.nan
    return df


def _serie_mixta():
    """VPED bimodal y los estados del régimen de 300° más altos: los mayores picos son del grupo menor."""
    df = _SERIE_ORIGINAL()
    otro = df["VPED"].between(255, 345)
    for c in ("VHM0", "VHM0_SW1", "VHM0_WW", "VHM0_SW2"):
        df.loc[otro, c] = np.round(df.loc[otro, c] * 1.3, 2)
    return df


def _sembrar(raiz, monkeypatch, unimodal: bool = True):
    monkeypatch.setattr(T1, "_serie", _serie_unimodal if unimodal else _serie_mixta)
    T1._sembrar(raiz)
    codigo = raiz.parent / "codigo"
    r0 = {"factor_concordancia": 20.0, "regla_no_discrimina": True, "valores_del_sitio_leidos": False}
    T1._publicar(raiz, E0.ruta_salida(raiz), {"etapa0.json": lambda p: json_estricto.escribir(p, r0)}, codigo)


@pytest.mark.lento
def test_clima_de_extremo_a_extremo(tmp_path, chico, monkeypatch):  # noqa: F811
    raiz = tmp_path / "datos"
    _sembrar(raiz, monkeypatch)
    E1.etapa1(raiz, procesos_=0)
    salida = R.clima(raiz, procesos_=0, b=B_PRUEBA)
    man = etapa.exigir_publicada(salida)
    for f in ("resumen_f06.json", "clima_mensual.csv", "clase_fisica.parquet", "dominancia.csv",
              "picos_provisionales.csv", "origen_mayores.csv", "dependencia.json", "rachas.csv",
              "declustering_dudoso.csv", "tp_condicional.csv", "peralte.csv", "gev_bloques.csv", "estratos.json",
              "figuras/rosa_direccion.png", "figuras/gev.png", "figuras/eventos/mayor_01.png"):  # fmt: skip
        assert f in man["salidas"], f
    r = json_estricto.leer(salida / "resumen_f06.json")
    assert r["gate"]["G3"] == [] and r["gate"]["G2"] and r["gate"]["G5"] == []
    assert r["gate"]["G7"]["comprobado_antes_de_leer_excesos"]
    assert r["decisiones"]["estratificacion"]["C"] == "no candidata"
    assert r["decisiones"]["poblacion"] == "población única"
    est = json_estricto.leer(salida / "estratos.json")["estratificacion"]
    b = est["candidatas"]["B"]
    assert b["desenlace"] in (ET.ESTABLECIDO, ET.NO_CONCLUYENTE, ET.DESCARTADO)
    assert set(b["modelos_por_rama"]) == {"E0_sin_corregir", "E0_cuantiles", "E1_sin_corregir", "E1_cuantiles"}
    assert est["escenarios"][0]["nombre"] == "sin_estratos"
    dep = json_estricto.leer(salida / "dependencia.json")
    assert dep["B"] == B_PRUEBA and set(dep["ic"]) == set(ES.NOMBRES)
    gev = pd.read_csv(salida / "gev_bloques.csv")
    assert list(gev["rama"]) == list(R.I.RAMAS) and gev["n"].iloc[0] == 46 and gev["n"].iloc[2] == 32
    assert "estratos_etapa1" in man["insumos"]


def test_g7_se_detiene_si_cambio_el_codigo_de_la_etapa1(tmp_path, chico, monkeypatch):  # noqa: F811
    raiz = tmp_path / "datos"
    _sembrar(raiz, monkeypatch)
    E1.etapa1(raiz, procesos_=0)
    real = orquestador.huella_etapa
    monkeypatch.setattr(orquestador, "huella_etapa", lambda et, *a, **k: "otra" if et is E1.ETAPA1 else real(et))
    leidos = []
    monkeypatch.setattr(R.I, "cargar", lambda *a, **k: leidos.append(1))
    with pytest.raises(GateFallido, match="G7"):
        R.clima(raiz, procesos_=0, b=B_PRUEBA)
    assert not leidos  # no se cargó la serie: ningún exceso leído


def test_g7_se_detiene_sin_etapa1(tmp_path, monkeypatch):
    raiz = tmp_path / "datos"
    _sembrar(raiz, monkeypatch)
    with pytest.raises(GateFallido, match="G7"):
        R.clima(raiz, procesos_=0, b=B_PRUEBA)


def test_poblacion_mixta_detiene_antes_de_publicar(tmp_path, chico, monkeypatch):  # noqa: F811
    raiz = tmp_path / "datos"
    _sembrar(raiz, monkeypatch, unimodal=False)  # VPED bimodal: C es candidata y domina los mayores
    r1 = json_estricto.leer(E1.etapa1(raiz, procesos_=0) / "estratos_etapa1.json")
    assert r1["bimodalidad_vped"]["candidata"]
    with pytest.raises(GateFallido, match="población mixta"):
        R.clima(raiz, procesos_=0, b=B_PRUEBA)
    assert not (R.ruta_salida(raiz, R.I.NOMBRE) / etapa.MANIFIESTO).exists()


def test_replicas_iguales_con_1_y_2_procesos(tmp_path):
    t = pd.date_range("1990-01-01", "1999-12-31 21:00", freq="3h")
    rng = np.random.default_rng(5)
    s = pd.Series(np.exp(rng.normal(0, 0.4, t.size)), index=t)
    bl = D.bloques_anuales(s)
    u = ES.umbrales(s)
    a = ES.replicas(bl, u, 1, tmp_path / "a", b=6)
    b = ES.replicas(bl, u, 2, tmp_path / "b", b=6)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ES.vector(bl, u), ES.vector(bl, u))


# --- Piezas -------------------------------------------------------------------------------------------------


def test_desenlace_y_escenarios():
    assert ET.desenlace(True, [0.5]) == ET.ESTABLECIDO
    assert ET.desenlace(False, [0.85, 0.9]) == ET.DESCARTADO
    assert ET.desenlace(False, [0.85, 0.7]) == ET.NO_CONCLUYENTE
    e = ET.escenarios(ET.NO_CONCLUYENTE, menos_de_30=True)
    assert [x["nombre"] for x in e] == ["sin_estratos", "estratificado_xi_comun"]
    assert not e[1]["envolvente"] and e[1]["rotulo"] == ET.ROTULO_MENOS_30  # ADR-0029
    assert ET.escenarios(ET.NO_CONCLUYENTE, menos_de_30=True, excluir_menos_de_30=False)[1]["envolvente"]
    assert len(ET.escenarios(ET.DESCARTADO, menos_de_30=False)) == 1


def test_eleccion_entre_dos_establecidas():
    def c(des, p, est):
        return {"desenlace": des, "prueba": {"p": p, "estadistico": est}, "escenarios": ET.escenarios(des, False)}

    cands = {"B": c(ET.ESTABLECIDO, 0.02, 3.0), "C": c(ET.ESTABLECIDO, 0.01, 2.0)}
    assert R.elegir_establecida(cands) == "C"
    cands = {"B": c(ET.ESTABLECIDO, 0.001, 9.0), "C": c(ET.ESTABLECIDO, 0.001, 5.0)}
    assert R.elegir_establecida(cands) == "B"  # empate en la cota: mayor estadístico
    cands = {"B": c(ET.ESTABLECIDO, 0.01, 3.0), "C": c(ET.NO_CONCLUYENTE, 0.2, 1.0)}
    assert [e["nombre"] for e in R.escenarios_familia(cands)] == [
        "sin_estratos", "estratificado_xi_comun_B", "estratificado_xi_comun_C"]  # fmt: skip


def test_grupos_de_meses():
    meses = pd.Series([11] * 3 + [12] * 1 + [1] * 2 + [2] * 6 + [4] * 1)
    g = ET.grupos_de_meses(meses, 5)
    assert g[11] == g[12] == g[1]  # 3 + 1 + 2 = 6 (empieza tras el mayor hueco, sin partir diciembre–enero)
    assert g[2] != g[11] and g[4] == g[2]  # el resto (abril, 1) se une al último grupo


def test_alarma_adr0029():
    m = {"xi_comun": {"delta_pct": 1.0}, "xi_por_estrato": {"delta_pct": 5.0}}
    assert ET.alarma_adr0029(m) == ["xi_por_estrato: ΔRL100 = +5.0 % (> 2 %)"]


def test_concordancia():
    ic = {n: {"punto": np.nan} for n in ES.NOMBRES}
    ic |= {f"theta_q{q:g}": {"punto": 0.5} for q in ES.CUANTILES_THETA}
    ic["T_C_h_q0.99"] = {"punto": 24.0}
    ic["escala_extremograma_h"] = {"punto": 48.0}
    c = ES.concordancia(ic, 3.0, False)
    assert c["razon_max_min"] == 2.0 and c["concuerdan"] and "sin corte natural de los huecos" in c["senales_para_f7"]
    ic["escala_extremograma_h"] = {"punto": 240.0}
    assert "escalas de dependencia discordantes" in ES.concordancia(ic, 3.0, False)["senales_para_f7"]


def test_gev_bloques_completos_y_mes_de_inicio():
    t = pd.date_range("1993-02-01", "2026-08-31 21:00", freq="3h")
    hs = pd.Series(1.0 + (t.month == 7) * 2.0 + np.arange(t.size) * 1e-7, index=t)
    mx = GV.maximos(hs)
    assert mx["bloque"].min() == 1994 and mx["bloque"].max() == 2025
    m0, conteo = GV.mes_inicio(hs)
    assert m0 == 1 and conteo[7] == len(mx)
    hs2 = pd.Series(1.0 + (t.month == 1) * 2.0, index=t)  # máximos en enero: el bloque empieza en otro mes
    assert GV.mes_inicio(hs2)[0] != 1


def test_g3_detecta_eje_que_recorta_el_maximo():
    fig, ax = estilo.nueva_figura()
    ax.plot([0, 1, 2], [1.0, 5.0, 2.0])
    ax.axhline(3.0)
    assert F.revisar(fig, "f") == []
    ax.set_ylim(0, 4)
    assert len(F.revisar(fig, "f")) == 1
    ax.set_ylim(0, 6)
    ax.axvline(10.0)  # una línea vertical fuera del eje x también cuenta
    ax.set_xlim(0, 2)
    assert len(F.revisar(fig, "f")) == 1
    matplotlib.pyplot.close(fig)
