"""Avance reanudable de la etapa 2 (instrucción del usuario, 2026-11-17): una corrida interrumpida se retoma sin
repetir lo ya guardado y da el mismo resultado."""

from __future__ import annotations

import pytest

from peruvian_wave_analysis.validacion import reanudable

LLAMADAS: list[int] = []


def _cuadrado(x):
    LLAMADAS.append(x)
    if x == 3 and _cuadrado.cortar:
        raise KeyboardInterrupt  # simula un corte a mitad de la corrida
    return x * x


_cuadrado.cortar = False


def test_reanuda_sin_repetir_y_con_el_mismo_resultado(tmp_path):
    tareas = [(f"t{x}", x) for x in range(6)]
    LLAMADAS.clear()
    _cuadrado.cortar = True
    with pytest.raises(KeyboardInterrupt):
        reanudable.ejecutar(_cuadrado, tareas, tmp_path, 0)
    assert sorted(p.name for p in tmp_path.glob("*.pkl")) == ["t0.pkl", "t1.pkl", "t2.pkl"]
    (tmp_path / "t4.pkl.parcial").write_bytes(b"resto de una escritura cortada")
    LLAMADAS.clear()
    _cuadrado.cortar = False
    avisos = []
    res = reanudable.ejecutar(_cuadrado, tareas, tmp_path, 0, avisos.append)
    assert res == [x * x for x in range(6)]
    assert LLAMADAS == [3, 4, 5]  # solo lo que faltaba
    assert "3 de 6 tareas ya guardadas" in avisos[0]
    assert not list(tmp_path.glob("*.parcial"))
    LLAMADAS.clear()
    assert reanudable.ejecutar(_cuadrado, tareas, tmp_path, 0) == res and LLAMADAS == []


def test_la_clave_cambia_con_el_codigo_o_los_insumos():
    a = reanudable.clave("huella1", {"x": "h1"}, 999)
    assert a == reanudable.clave("huella1", {"x": "h1"}, 999)
    assert a != reanudable.clave("huella2", {"x": "h1"}, 999)
    assert a != reanudable.clave("huella1", {"x": "h2"}, 999)


def test_limpiar(tmp_path):
    d = tmp_path / "avance"
    reanudable.ejecutar(_cuadrado, [("t", 2)], d, 0)
    reanudable.limpiar(d)
    assert not d.exists()
