"""Estilo de figuras de análisis (contrato de la F2, §3.10)."""

from __future__ import annotations

import numpy as np
import pytest

from peruvian_wave_analysis import estilo


def test_posiciones_de_trazado():
    np.testing.assert_allclose(estilo.posiciones_trazado(4), [0.2, 0.4, 0.6, 0.8])
    with pytest.raises(ValueError):
        estilo.posiciones_trazado(0)


def test_asegurar_visible_amplia_el_eje():
    fig, ax = estilo.nueva_figura()
    ax.plot([0, 1], [0, 1])
    ax.set_ylim(0, 1)
    estilo.asegurar_visible(ax, [0.5, 3.0])
    assert ax.get_ylim()[1] > 3.0
    estilo.asegurar_visible(ax, [-2.0], eje="x")
    assert ax.get_xlim()[0] < -2.0


def test_revisar_figura():
    fig, ax = estilo.nueva_figura()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel(estilo.ETIQUETAS["T"])
    ax.set_ylabel("Hs")
    ax.set_title("Máximos por ano")
    problemas = estilo.revisar_figura(fig)
    assert any("sin unidad" in p for p in problemas)
    assert any("año" in p for p in problemas)
    assert len(problemas) == 2
    ax.set_ylabel(estilo.ETIQUETAS["Hs"])
    ax.set_title("Máximos por año")
    assert estilo.revisar_figura(fig) == []
    ax.set_title("x" * 400)  # se sale del lienzo
    assert any("fuera de la figura" in p for p in estilo.revisar_figura(fig))


def test_atributos_privados_de_matplotlib_existen():
    fig, ax = estilo.nueva_figura()
    assert hasattr(fig, "_suptitle") and hasattr(ax, "_left_title") and hasattr(ax, "_right_title")


def test_guardado_determinista(tmp_path):
    from peruvian_wave_analysis import manifiesto

    hashes = []
    for i in range(2):
        fig, ax = estilo.nueva_figura()
        ax.plot([0, 1, 2], [1, 3, 2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        hashes.append(manifiesto.sha256(estilo.guardar_figura(fig, tmp_path / f"f{i}.png")))
    assert hashes[0] == hashes[1]
    with pytest.raises(ValueError):
        estilo.guardar_figura(estilo.nueva_figura()[0], tmp_path / "f.pdf")
