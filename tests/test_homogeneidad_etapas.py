"""F4 de extremo a extremo con un WAVE-DATA sintético (contrato de la F4, §5, última fila): escalón inyectado en
1993 → la etapa lo encuentra, lo publica como «establecido, sin atribuir» y deja la prueba de la F5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peruvian_wave_analysis.comun import etapa, json_estricto
from peruvian_wave_analysis.datos import config as CD
from peruvian_wave_analysis.datos.run import ruta_candidatos, ruta_serie
from peruvian_wave_analysis.homogeneidad import config as C
from peruvian_wave_analysis.homogeneidad import run

NOMBRE = "sitio-prueba"


def _publicar(raiz, salida, archivos: dict, codigo):
    with etapa.staging_protegido(salida, [codigo], base_codigo=codigo.parent) as st:
        for nombre, escribir in archivos.items():
            escribir(st / nombre)
        etapa.publicar(st, salida, etapa.Gate(True, {}), [codigo], {}, raiz, None, base_codigo=codigo.parent)


def _sembrar(raiz, paso: float):
    rng = np.random.default_rng(0)
    t = pd.date_range("1980-01-01", "2026-08-31 21:00", freq="3h", name="tiempo")
    n = t.size
    esc = 1.6 + 0.4 * np.cos(2 * np.pi * (t.month.to_numpy() - 7) / 12)
    # serie con memoria de unos días (tormentas) y variabilidad entre años
    z = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        z[i] = 0.97 * z[i - 1] + 0.243 * e[i]
    anio = rng.normal(0, 0.05, 47)[t.year.to_numpy() - 1980]
    hs = esc * np.exp(0.35 * z + anio)
    hs = np.where(t >= pd.Timestamp("1993-02-01"), hs * (1 + paso), hs)
    hs = np.round(hs, 2)
    df = pd.DataFrame(index=t)
    for v in CD.VARIABLES:
        df[v] = np.nan
    df["VHM0"] = hs
    df["VHM0_SW1"] = np.round(hs * 0.8, 2)
    df["VHM0_WW"] = np.round(hs * 0.4, 2)
    df["VHM0_SW2"] = np.round(hs * 0.2, 2)
    df["VTPK"] = np.round(10 + 2 * z + rng.normal(0, 1, n), 2)
    df["VTM02"] = np.round(df["VTPK"] * 0.7, 2)
    marcas = pd.DataFrame({"marca_discrepancia_energia": rng.normal(0, 0.01, n)}, index=t)
    serie = pd.concat([df, marcas], axis=1)
    serie["valido"] = True
    vecinas = pd.DataFrame(
        {
            f"{la:.1f}_{lo:.1f}": hs * (1 + rng.normal(0, 0.02, n))
            for la in (-15.2, -15.0, -14.8)
            for lo in (-76.0, -75.8, -75.6)
        },
        index=t,
    )
    codigo = raiz.parent / "codigo"
    codigo.mkdir(exist_ok=True)
    (codigo / "m.py").write_text("X = 1\n", encoding="utf-8")
    anios = np.arange(1980, 2027)
    _publicar(
        raiz,
        ruta_candidatos(raiz, NOMBRE),
        {
            "serie_decodificada.parquet": lambda p: df.to_parquet(p),
            "excursiones_k_rev.csv": lambda p: pd.DataFrame(
                {"anio": [1999, 2017], "tiempo": ["1999-01-01", "2017-01-01"]}
            ).to_csv(p, index=False),
            "candidatos.csv": lambda p: pd.DataFrame({"tiempo": ["2017-01-01 00:00"], "regla": ["detector"]}).to_csv(
                p, index=False
            ),
        },
        codigo,
    )
    _publicar(
        raiz,
        ruta_serie(raiz, NOMBRE),
        {
            "serie_nodo.parquet": lambda p: serie.to_parquet(p),
            "vecinas_hs.parquet": lambda p: vecinas.to_parquet(p),
            "marcas_por_anio.csv": lambda p: pd.DataFrame({"anio": anios, "energia": 0}).to_csv(p, index=False),
            "marcas_por_regla.csv": lambda p: pd.DataFrame({"tiempo": [], "regla": []}).to_csv(p, index=False),
        },
        codigo,
    )
    f1 = raiz / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / NOMBRE
    f1.mkdir(parents=True)
    pd.DataFrame(
        {"fecha": ["1980-01-01"], "tipo": ["resolución"], "descripcion": ["malla"], "referencia": ["QUID"]}
    ).to_csv(f1 / "cronologia_waverys.csv", index=False)
    # pasadas sintéticas: asimiladas y placebos, sin relación con los saltos
    filas = []
    for m, a, b, cada in (
        ("TOPEX", "1993-01-01", "2005-10-08", 10),
        ("JASON-1", "2002-01-15", "2013-06-21", 10),
        ("JASON-2", "2008-07-04", "2019-09-29", 10),
        ("HY-2", "2011-10-11", "2018-06-05", 3),
        ("SARAL", "2013-03-17", "2026-07-02", 3),
        ("GFO", "2000-01-09", "2008-09-03", 3),
    ):
        tt = pd.date_range(a, b, freq=f"{cada}D") + pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=7)
        filas.append(pd.DataFrame({"mision": m, "tiempo": tt, "n": 20, "dist_min_km": 50.0}))
    pas = pd.concat(filas, ignore_index=True)
    expo = run.exposicion_por_paso(t, pas)
    locales = (
        pas.groupby("mision")
        .agg(inicio=("tiempo", "min"), fin=("tiempo", "max"), pasadas=("tiempo", "size"))
        .reset_index()
    )
    _publicar(
        raiz,
        run.ruta_altimetros(raiz, NOMBRE),
        {
            "exposicion_altimetros.parquet": lambda p: expo.to_parquet(p),
            "misiones_locales.csv": lambda p: locales.to_csv(p, index=False),
        },
        codigo,
    )


@pytest.mark.lento
def test_escalon_en_1993_queda_establecido_sin_atribuir(tmp_path):
    raiz = tmp_path / "WAVE-DATA"
    raiz.mkdir()
    _sembrar(raiz, paso=0.30)
    salida = run.homogeneidad(raiz, NOMBRE, procesos_n=1, B=99, replicas=20, multiplos=(1.0,), B_ic=19)
    man = etapa.exigir_publicada(salida)
    orden = man["extra"]["orden_de_escritura"]
    assert orden.index("potencia.csv") < orden.index("cuantiles_por_tramo.csv") < orden.index("figuras")
    r = json_estricto.leer(salida / "resumen_f04.json")
    assert r["resultados"]["A_C1"] == C.ESTABLECIDO
    e1 = [e for e in r["escenarios_hs"] if e["nombre"] == "E1"][0]
    assert e1["forma"] == "recorte"
    assert r["a_la_F5"]["C1"]["activa"]
    assert any(f.get("origen") == "C1" for f in r["a_la_F8"]["fechas_de_cambio"])
    pc = json_estricto.leer(salida / "pruebas_cambio.json")
    assert pc["A"]["C1"]["cociente"][2] > 1.2
    # §20: añadidos posteriores a ver datos, escritos después de lo observado y antes del resumen
    assert orden.index("prueba_altimetros.json") < orden.index("altimetros_por_mision_anio.csv")
    assert orden.index("altimetros_por_mision_anio.csv") < orden.index("resumen_f04.json")
    post = r["posterior_a_ver_datos"]
    assert "§20" in post["rotulo"] and {"placebos", "controles_por_mision", "a_la_F8", "lectura_de_potencia"} <= set(
        post
    )
    assert r["a_la_F5"]["C1"]["observaciones"]["T0"] == "GEOSAT (sin datos locales)"
    pm = pd.read_csv(salida / "altimetros_por_mision_anio.csv")
    assert {"GFO", "SARAL", "HY-2"} <= set(pm["mision"])
    pa = json_estricto.leer(salida / "prueba_altimetros.json")
    assert {p["clase"] for p in pa["placebos"]} <= {"cuenta", "no informativo", "sin pasos"}
    for f in (
        "cronologia_f04.csv",
        "tramos.csv",
        "cobertura_altimetria.csv",
        "indicadores_anuales.csv",
        "figuras/cuantiles_por_tramo.png",
        "figuras/exposicion_altimetros.png",
    ):
        assert (salida / f).is_file()
    # G7: dos corridas idénticas byte a byte
    salidas = dict(man["salidas"])
    run.homogeneidad(raiz, NOMBRE, procesos_n=2, B=99, replicas=20, multiplos=(1.0,), B_ic=19)
    assert etapa.exigir_publicada(salida)["salidas"] == salidas
