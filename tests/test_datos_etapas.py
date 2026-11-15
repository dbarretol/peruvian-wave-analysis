"""F3: las dos etapas publicadas de extremo a extremo con un WAVE-DATA sintético (contrato de la F3, §2, §3.9, §5)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.comun.errores import GateFallido
from peruvian_wave_analysis.datos import carga, periodos, run
from peruvian_wave_analysis.datos import config as C
from peruvian_wave_analysis.datos import resoluciones as R
from peruvian_wave_analysis.manifiesto import sha256
from peruvian_wave_analysis.sitio import serie as serie_f1
from tests.test_datos import LATS, LONS, escribir_nc

NOMBRE = "sitio-prueba"
PICO = 300  # posición del pico sembrado


@pytest.fixture(scope="module")
def datos(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("WAVE-DATA")
    dir_serie = raiz / "raw" / "waverys" / "serie"
    dir_serie.mkdir(parents=True)
    t1 = pd.date_range("2024-01-01", periods=400, freq="3h")
    t2 = pd.date_range(t1[-1] + pd.Timedelta(hours=3), periods=400, freq="3h")
    hs = np.round(1.5 + 0.5 * np.random.default_rng(0).random(800), 2)
    hs[PICO] += 8.0  # pico aislado de 8 m (k·s ≈ 2–3 m con este ruido), por encima del P95
    archivos = {}
    for nombre, t, sl in (("a_2024.nc", t1, slice(0, 400)), ("b_2025.nc", t2, slice(400, 800))):
        r = escribir_nc(dir_serie / nombre, t, {"VHM0": hs[sl]})
        archivos[nombre] = sha256(r)
    # gate publicado de la serie de la F1
    codigo = raiz.parent / "codigo_f1"
    codigo.mkdir(exist_ok=True)
    (codigo / "m.py").write_text("X = 1\n", encoding="utf-8")
    salida_gate = serie_f1.ruta_gate(raiz, NOMBRE)
    with etapa.staging_protegido(salida_gate, [codigo], base_codigo=codigo.parent) as st:
        json_estricto.escribir(
            st / "serie_gate.json",
            {"archivos_sha256": archivos, "pasos": 800, "inicio": str(t1[0]), "fin": str(t2[-1])},
        )
        etapa.publicar(st, salida_gate, etapa.Gate(True, {}), [codigo], {}, raiz, None, base_codigo=codigo.parent)
    final = raiz / "processed" / "F01-sitio-y-fuentes" / "mapa-sitio-final" / NOMBRE
    final.mkdir(parents=True)
    (final / "ficha_nodo.json").write_text(json.dumps({"nodo_elegido": {"prof_nodo_m": 194.0}}), encoding="utf-8")
    est = raiz / "raw" / "waverys" / "estatico"
    est.mkdir(parents=True)
    xr.Dataset(
        {"deptho": (("latitude", "longitude"), np.full((LATS.size, LONS.size), 420.0))},
        coords={"latitude": LATS, "longitude": LONS},
    ).to_netcdf(est / "deptho_prueba.nc")
    run.candidatos(raiz, NOMBRE, anios_calibracion=400)
    return raiz


def test_candidatos_publica_sin_exigir_resoluciones(datos):
    d = run.ruta_candidatos(datos, NOMBRE)
    man = etapa.exigir_publicada(d)
    assert man["gate"]["aprobado"]
    cand = pd.read_csv(d / "candidatos.csv", parse_dates=["tiempo"])
    t_pico = pd.Timestamp("2024-01-01") + PICO * pd.Timedelta(hours=3)
    fila = cand[(cand["tiempo"] == t_pico) & (cand["regla"] == "detector")]
    assert len(fila) == 1 and fila["decision"].iloc[0] == "manual" and fila["motivo"].iloc[0] == "cola_p95"
    maximos = pd.read_csv(d / "maximos.csv", parse_dates=["tiempo"])
    assert len(maximos) == 20 and t_pico in set(maximos["tiempo"])
    assert sorted(p.name for p in (d / "figuras").glob("maximo_*.png"))[0] == "maximo_01.png"
    res = json.loads((d / "resumen_candidatos.json").read_text(encoding="utf-8"))
    assert res["k_rev"] <= res["k_desc"] and res["s_cuantos"] >= 2
    assert all(not f["aplica"] for f in res["firmas_ausencia"].values())


def _resoluciones(raiz, ruta, descartar_pico=False, confirmar=True):
    d = run.ruta_candidatos(raiz, NOMBRE)
    cand = pd.read_csv(d / "candidatos.csv", parse_dates=["tiempo"])
    maximos = pd.read_csv(d / "maximos.csv", parse_dates=["tiempo"])["tiempo"]
    t_pico = pd.Timestamp("2024-01-01") + PICO * pd.Timedelta(hours=3)
    base = {"motivo": "prueba", "fecha": "2026-11-15", "autor": "prueba"}
    ent = [
        {"tiempo": str(t), "regla": r, "decision": "mantener", **base}
        for t, r in zip(
            cand.loc[cand["decision"] == "manual", "tiempo"],
            cand.loc[cand["decision"] == "manual", "regla"],
            strict=True,
        )
    ]
    for t in maximos:
        e = {"tiempo": str(t), "regla": "maximo", "decision": "mantener", **base}
        if descartar_pico and t == t_pico:
            e["decision"] = "descartar"
            if confirmar:
                e["confirmado_por_usuario"] = "2026-11-15"
        ent.append(e)
    ruta.write_text(yaml.safe_dump(ent, allow_unicode=True), encoding="utf-8")
    return ruta, t_pico


def test_serie_sin_resoluciones_falla_sin_borrar_candidatos(datos, tmp_path):
    vacio = tmp_path / "vacio.yaml"
    vacio.write_text("[]\n", encoding="utf-8")
    with pytest.raises(GateFallido):
        run.serie(datos, NOMBRE, ruta_resoluciones=vacio)
    assert (run.ruta_serie(datos, NOMBRE) / etapa.GATE_FALLIDO).exists()
    etapa.exigir_publicada(run.ruta_candidatos(datos, NOMBRE))  # la primera etapa sigue publicada
    ruta, _ = _resoluciones(datos, tmp_path / "r.yaml", descartar_pico=True, confirmar=False)
    with pytest.raises(GateFallido, match="sin confirmación"):
        run.serie(datos, NOMBRE, ruta_resoluciones=ruta)


def test_serie_aplica_descartes_a_todas_las_variables(datos, tmp_path):
    ruta, t_pico = _resoluciones(datos, tmp_path / "r.yaml", descartar_pico=True, confirmar=True)
    d = run.serie(datos, NOMBRE, ruta_resoluciones=ruta)
    s = pd.read_parquet(d / "serie_nodo.parquet")
    assert not s.loc[t_pico, "valido"] and s.loc[t_pico, list(C.VARIABLES)].isna().all()
    assert s["valido"].sum() == 799
    res = json.loads((d / "resumen_f03.json").read_text(encoding="utf-8"))
    assert res["descartados"] == 1 and any(x["decision"] == "descartar" for x in res["sensibilidad_f9"])
    largos = json.loads((d / "periodos_largos.json").read_text(encoding="utf-8"))
    assert largos["h_sitio_m"] == 194.0 and largos["h_modelo_m"] == 420.0
    # la carga aplica `valido`; la serie sin descartar solo pidiéndola
    assert np.isnan(carga.cargar_serie(datos, NOMBRE).loc[t_pico, "VHM0"])
    assert carga.cargar_serie(datos, NOMBRE, sin_descartar=True).loc[t_pico, "VHM0"] > 4.0


# --- Tp largo y Ks (§3.11) y resoluciones (§3.9) ------------------------------------------------


def test_numero_de_onda_y_ks_en_sus_limites():
    T = np.array([5.0, 10.0])
    assert periodos.numero_de_onda(T, 5000.0) == pytest.approx((2 * np.pi / T) ** 2 / C.G, rel=1e-9)  # profundas
    assert periodos.numero_de_onda(np.array([200.0]), 2.0) == pytest.approx(
        2 * np.pi / 200.0 / np.sqrt(C.G * 2.0), rel=1e-3
    )  # someras
    assert periodos.asomeramiento(np.array([5.0]), 5000.0) == pytest.approx(1.0, abs=1e-9)
    assert periodos.asomeramiento(np.array([18.0]), 194.0)[0] < 1.0  # profundidad intermedia: Ks < 1


def test_resumen_de_periodos_largos_a_mano():
    idx = pd.date_range("2000-01-01", "2001-12-31 21:00", freq="3h")
    df = pd.DataFrame({"VHM0": 1.0, "VTPK": 10.0}, index=idx)
    df.loc[df.index[:30], ["VHM0", "VTPK"]] = [5.0, 18.0]  # 30 estados extremos de 2000 con Tp 18 s
    df.loc[df.index[-50:], ["VHM0", "VTPK"]] = [4.0, 12.0]  # 50 de 2001 con Tp 12 s
    r = periodos.resumen(df, [2000, 2001], 194.0, 420.0)
    sobre = r["sobre_p99"]  # P99 de 5840 registros: los 80 extremos
    assert sobre["n"] == 80 and sobre["fraccion_tp_ge_16s"] == pytest.approx(30 / 80)
    assert r["maximo_anual"]["n"] == 2 and r["maximo_anual"]["fraccion_tp_ge_18s"] == pytest.approx(0.5)
    ks = periodos.asomeramiento(np.array([18.0, 12.0]), 194.0) / periodos.asomeramiento(np.array([18.0, 12.0]), 420.0)
    assert r["maximo_anual"]["razon_ks_media"] == pytest.approx(ks.mean())


def test_resoluciones_validan_su_forma(tmp_path):
    p = tmp_path / "r.yaml"
    base = {"tiempo": "2000-01-01 03:00", "regla": "maximo", "motivo": "m", "fecha": "2026-11-15", "autor": "a"}
    p.write_text(yaml.safe_dump([base | {"decision": "borrar"}]), encoding="utf-8")
    with pytest.raises(R.ErrorResoluciones):
        R.leer(p)
    p.write_text(yaml.safe_dump([base | {"decision": "mantener"}, base | {"decision": "descartar"}]), encoding="utf-8")
    with pytest.raises(R.ErrorResoluciones, match="repetida"):
        R.leer(p)
    p.write_text(
        yaml.safe_dump([{k: v for k, v in base.items() if k != "autor"} | {"decision": "mantener"}]), encoding="utf-8"
    )
    with pytest.raises(R.ErrorResoluciones, match="autor"):
        R.leer(p)
    assert R.leer(tmp_path / "no-existe.yaml") == []
