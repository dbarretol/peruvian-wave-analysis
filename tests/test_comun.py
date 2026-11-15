"""Infraestructura común: JSON estricto, huella, publicación con gate, orquestador y semillas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from peruvian_wave_analysis.comun import etapa, huella, json_estricto, orquestador, semillas
from peruvian_wave_analysis.comun.errores import GateFallido, HashDistinto, InsumoFaltante


# --- JSON estricto -----------------------------------------------------------------------------

def test_json_no_finitos_numpy_y_path():
    texto = json_estricto.dumps({"a": float("nan"), "b": float("inf"), "c": -np.inf, "d": np.float64(1.5),
                                 "e": np.int64(3), "f": np.array([1.0, np.nan]), "g": Path("x/y"), "h": (1, 2)})
    assert json.loads(texto) == {"a": None, "b": None, "c": None, "d": 1.5, "e": 3, "f": [1.0, None],
                                 "g": "x/y", "h": [1, 2]}


def test_json_tipo_raro_es_error():
    with pytest.raises(TypeError):
        json_estricto.dumps({"x": object()})
    with pytest.raises(TypeError):
        json_estricto.dumps({"x": {1, 2}})


@pytest.mark.parametrize("constante", ["NaN", "Infinity", "-Infinity"])
def test_json_lector_rechaza_constantes(constante):
    with pytest.raises(ValueError):
        json_estricto.loads(f'{{"x": {constante}}}')


# --- Huella -------------------------------------------------------------------------------------

BASE_PY = '''"""Docstring del módulo."""
X = 1


def f(a):
    """Docstring de f."""
    return a + X  # comentario
'''


def _huella_de(tmp_path: Path, texto: str, nombre: str = "m.py") -> str:
    d = tmp_path / "codigo"
    d.mkdir(exist_ok=True)
    (d / nombre).write_text(texto, encoding="utf-8")
    return huella.huella_codigo([d], base=tmp_path)


def test_huella_ignora_comentarios_docstrings_y_formato(tmp_path):
    h0 = _huella_de(tmp_path, BASE_PY)
    otra = BASE_PY.replace("# comentario", "# otro comentario").replace("Docstring de f.", "Otra cosa.")
    otra = otra.replace("return a + X", "return  a+X").replace('"""Docstring del módulo."""\n', "")
    assert _huella_de(tmp_path, otra) == h0
    assert _huella_de(tmp_path, BASE_PY.replace("\n", "\r\n")) == h0


def test_huella_cambia_con_la_logica(tmp_path):
    h0 = _huella_de(tmp_path, BASE_PY)
    assert _huella_de(tmp_path, BASE_PY.replace("X = 1", "X = 2")) != h0


def test_huella_yaml_por_contenido_con_tipos(tmp_path):
    h0 = _huella_de(tmp_path, "a: 1\nb: [x, y]\n", "c.yaml")
    assert _huella_de(tmp_path, "# comentario\na:   1\nb:\n  - x\n  - y\n", "c.yaml") == h0
    assert _huella_de(tmp_path, "a: '1'\nb: [x, y]\n", "c.yaml") != h0


def test_huella_no_depende_de_la_ruta_absoluta(tmp_path):
    for sub in ("uno", "dos"):
        (tmp_path / sub / "codigo").mkdir(parents=True)
        (tmp_path / sub / "codigo" / "m.py").write_text(BASE_PY, encoding="utf-8")
    assert (huella.huella_codigo([tmp_path / "uno" / "codigo"], base=tmp_path / "uno")
            == huella.huella_codigo([tmp_path / "dos" / "codigo"], base=tmp_path / "dos"))


# --- Etapa --------------------------------------------------------------------------------------

@pytest.fixture
def entorno(tmp_path):
    datos = tmp_path / "WAVE-DATA"
    (datos / "raw").mkdir(parents=True)
    insumo = datos / "raw" / "serie.csv"
    insumo.write_text("t,hs\n0,1.0\n", encoding="utf-8")
    codigo = tmp_path / "codigo"
    codigo.mkdir()
    (codigo / "m.py").write_text("X = 1\n", encoding="utf-8")
    salida = datos / "processed" / "F99" / "etapa"
    return datos, insumo, [codigo], salida


def _publicar(datos, insumo, codigo, salida, aprobado=True, contenido="1\n", semilla=7):
    with etapa.staging_protegido(salida, codigo, base_codigo=codigo[0].parent) as st:
        (st / "resultado.txt").write_text(contenido, encoding="utf-8")
        (st / "sub").mkdir()
        (st / "sub" / "b.json").write_text("{}\n", encoding="utf-8")
        return etapa.publicar(st, salida, etapa.Gate(aprobado, {"criterio": 1.0}, [] if aprobado else ["no"]),
                              codigo, {"serie": insumo}, datos, semilla, base_codigo=codigo[0].parent)


def test_publicar_y_leer(entorno):
    datos, insumo, codigo, salida = entorno
    _publicar(datos, insumo, codigo, salida)
    man = etapa.exigir_publicada(salida)
    assert man["insumos"]["serie"]["ruta"] == "raw/serie.csv"
    assert set(man["salidas"]) == {"resultado.txt", "sub/b.json"}
    assert man["semilla"] == 7 and man["gate"]["aprobado"] is True
    assert not etapa.ruta_staging(salida).exists()


def test_manifiesto_sin_marcas_de_tiempo_y_determinista(entorno):
    datos, insumo, codigo, salida = entorno
    _publicar(datos, insumo, codigo, salida)
    b1 = (salida / etapa.MANIFIESTO).read_bytes()
    _publicar(datos, insumo, codigo, salida)
    assert (salida / etapa.MANIFIESTO).read_bytes() == b1


def test_gate_fallido_borra_la_salida_anterior(entorno):
    datos, insumo, codigo, salida = entorno
    _publicar(datos, insumo, codigo, salida)
    with pytest.raises(GateFallido):
        _publicar(datos, insumo, codigo, salida, aprobado=False)
    assert sorted(p.name for p in salida.iterdir()) == [etapa.GATE_FALLIDO]
    assert json_estricto.leer(salida / etapa.GATE_FALLIDO)["motivos"] == ["no"]
    with pytest.raises(GateFallido):
        etapa.exigir_publicada(salida)


def test_excepcion_en_el_staging_borra_la_salida_anterior(entorno):
    datos, insumo, codigo, salida = entorno
    _publicar(datos, insumo, codigo, salida)
    with pytest.raises(ZeroDivisionError):
        with etapa.staging_protegido(salida, codigo, base_codigo=codigo[0].parent) as st:
            (st / "parcial.txt").write_text("x", encoding="utf-8")
            1 / 0
    assert sorted(p.name for p in salida.iterdir()) == [etapa.GATE_FALLIDO]
    assert "ZeroDivisionError" in json_estricto.leer(salida / etapa.GATE_FALLIDO)["motivos"][0]
    assert not etapa.ruta_staging(salida).exists()


def test_archivo_alterado_o_sobrante_tras_publicar(entorno):
    datos, insumo, codigo, salida = entorno
    _publicar(datos, insumo, codigo, salida)
    (salida / "resultado.txt").write_text("2\n", encoding="utf-8")
    with pytest.raises(HashDistinto):
        etapa.exigir_publicada(salida)
    _publicar(datos, insumo, codigo, salida)
    (salida / "intruso.txt").write_text("x", encoding="utf-8")
    with pytest.raises(HashDistinto):
        etapa.exigir_publicada(salida)
    _publicar(datos, insumo, codigo, salida)
    (salida / "resultado.txt").unlink()
    with pytest.raises(InsumoFaltante):
        etapa.exigir_publicada(salida)


def test_insumos_faltante_hash_distinto_y_fuera_de_datos(entorno, tmp_path):
    datos, insumo, codigo, salida = entorno
    with pytest.raises(InsumoFaltante):
        etapa.exigir_insumos({"x": datos / "no-existe"})
    with pytest.raises(HashDistinto):
        etapa.exigir_insumos({"serie": insumo}, {"serie": "0" * 64})
    fuera = tmp_path / "fuera.csv"
    fuera.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        with etapa.staging_protegido(salida, codigo) as st:
            (st / "r.txt").write_text("x", encoding="utf-8")
            etapa.publicar(st, salida, etapa.Gate(True, {}), codigo, {"f": fuera}, datos, None)


def test_tipo_no_serializable_no_toca_la_salida(entorno):
    datos, insumo, codigo, salida = entorno
    _publicar(datos, insumo, codigo, salida)
    antes = (salida / etapa.MANIFIESTO).read_bytes()
    st = etapa.abrir_staging(salida)
    (st / "r.txt").write_text("x", encoding="utf-8")
    with pytest.raises(TypeError):
        etapa.publicar(st, salida, etapa.Gate(True, {"x": object()}), codigo, {"serie": insumo}, datos, None)
    assert (salida / etapa.MANIFIESTO).read_bytes() == antes


# --- Orquestador --------------------------------------------------------------------------------

@pytest.fixture
def proyecto(tmp_path):
    """Paquete mínimo con dos etapas: «a» lee un insumo crudo y «b» lee la salida de «a»."""
    raiz = tmp_path / "repo"
    paquete = raiz / "src" / "peruvian_wave_analysis"
    (paquete / "etapas").mkdir(parents=True)
    (paquete / "__init__.py").write_text("", encoding="utf-8")
    (paquete / "comunes.py").write_text("K = 1\n", encoding="utf-8")
    (paquete / "suelto.py").write_text("Z = 1\n", encoding="utf-8")
    (paquete / "etapas" / "__init__.py").write_text("", encoding="utf-8")
    (paquete / "etapas" / "a.py").write_text("from ..comunes import K\n", encoding="utf-8")
    (paquete / "etapas" / "b.py").write_text("def f():\n    from peruvian_wave_analysis import comunes\n", encoding="utf-8")
    (raiz / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    datos = tmp_path / "WAVE-DATA"
    (datos / "raw").mkdir(parents=True)
    crudo = datos / "raw" / "x.csv"
    crudo.write_text("1\n", encoding="utf-8")
    ets = [orquestador.Etapa("a", paquete / "etapas" / "a.py", datos / "processed" / "a"),
           orquestador.Etapa("b", paquete / "etapas" / "b.py", datos / "processed" / "b")]
    return raiz, paquete, datos, crudo, ets


def _correr(et, paquete, raiz, datos, insumos, contenido="r\n"):
    rutas = orquestador.rutas_codigo(et, paquete, raiz)
    with etapa.staging_protegido(et.salida, rutas, base_codigo=raiz) as st:
        (st / "r.txt").write_text(contenido, encoding="utf-8")
        etapa.publicar(st, et.salida, etapa.Gate(True, {}), rutas, insumos, datos, None, base_codigo=raiz)


def test_cierre_de_importaciones(proyecto):
    raiz, paquete, datos, crudo, (a, b) = proyecto
    rel = {p.relative_to(raiz).as_posix() for p in orquestador.rutas_codigo(a, paquete, raiz)}
    assert "src/peruvian_wave_analysis/comunes.py" in rel and "uv.lock" in rel
    assert "src/peruvian_wave_analysis/suelto.py" not in rel
    # importación dentro de una función
    rel_b = {p.relative_to(raiz).as_posix() for p in orquestador.rutas_codigo(b, paquete, raiz)}
    assert "src/peruvian_wave_analysis/comunes.py" in rel_b


def test_estados_por_codigo_insumo_y_dependencia(proyecto):
    raiz, paquete, datos, crudo, (a, b) = proyecto
    kw = {"raiz": paquete, "proyecto": raiz}
    est = orquestador.diagnostico([a, b], datos, **kw)
    assert est["a"].estado == est["b"].estado == orquestador.DESACTUALIZADA
    _correr(a, paquete, raiz, datos, {"x": crudo})
    _correr(b, paquete, raiz, datos, {"ra": a.salida / "r.txt"})
    est = orquestador.diagnostico([a, b], datos, **kw)
    assert est["a"].estado == est["b"].estado == orquestador.AL_DIA
    # la dependencia de b en a se deduce del manifiesto
    assert orquestador.dependencias_deducidas([a, b], datos)[1].depende_de == ("a",)

    # cambio de comentario: nada cambia; cambio de lógica en un módulo importado: a desactualizada, b pendiente
    (paquete / "comunes.py").write_text("K = 1  # nota\n", encoding="utf-8")
    assert orquestador.plan([a, b], datos, **kw) == []
    (paquete / "suelto.py").write_text("Z = 2\n", encoding="utf-8")
    assert orquestador.plan([a, b], datos, **kw) == []
    (paquete / "etapas" / "a.py").write_text("from ..comunes import K\nY = 2\n", encoding="utf-8")
    est = orquestador.diagnostico([a, b], datos, **kw)
    assert est["a"].estado == orquestador.DESACTUALIZADA
    assert est["b"].estado == orquestador.PENDIENTE
    # a vuelve a publicar lo mismo: b queda al día sin correr
    _correr(a, paquete, raiz, datos, {"x": crudo})
    assert orquestador.plan([a, b], datos, **kw) == []

    # insumo cambiado
    crudo.write_text("2\n", encoding="utf-8")
    est = orquestador.diagnostico([a, b], datos, **kw)
    assert est["a"].estado == orquestador.DESACTUALIZADA and "insumos" in est["a"].motivos[0]
    # a publica otra salida: b desactualizada por insumo cambiado
    _correr(a, paquete, raiz, datos, {"x": crudo}, contenido="otro\n")
    est = orquestador.diagnostico([a, b], datos, **kw)
    assert est["a"].estado == orquestador.AL_DIA and est["b"].estado == orquestador.DESACTUALIZADA

    # uv.lock forma parte de la huella
    _correr(b, paquete, raiz, datos, {"ra": a.salida / "r.txt"})
    (raiz / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    assert orquestador.plan([a, b], datos, **kw) == ["a", "b"]


def test_dependencia_desconocida_y_ciclo(proyecto):
    raiz, paquete, datos, crudo, (a, b) = proyecto
    from dataclasses import replace
    with pytest.raises(ValueError, match="desconocida"):
        orquestador.orden_topologico([replace(a, depende_de=("z",))])
    with pytest.raises(ValueError, match="ciclo"):
        orquestador.orden_topologico([replace(a, depende_de=("b",)), replace(b, depende_de=("a",))])


# --- Semillas -----------------------------------------------------------------------------------

def test_semillas_congeladas_y_por_nombre():
    # Valores congelados el 2026-11-15: si cambian, cambian todas las salidas con muestreo.
    assert semillas.SEMILLA_RAIZ == 20261115
    assert semillas.semilla("verificacion", "A1") == semillas.semilla("verificacion", "A1")
    assert semillas.semilla("verificacion", "A1") != semillas.semilla("verificacion", "A2")
    assert semillas.semilla("a", "b") != semillas.semilla("ab")
    assert semillas.semilla("verificacion", "A1") == 644026550
    np.testing.assert_allclose(semillas.generador("prueba").random(2), [0.60125765, 0.44234809], atol=1e-8)
