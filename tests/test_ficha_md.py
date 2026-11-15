"""``ficha_sitio.md`` desde salidas sintéticas: campos del contrato §2 y redacción exigida por la revisión."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from peruvian_wave_analysis.sitio import ficha_md


@pytest.fixture
def carpeta(tmp_path):
    sectores = pd.DataFrame([
        {"sector": k, "nombre_sector": nom, "fraccion_rayos_1000_km": f, "dist_min_tierra_km": dmin,
         "abierto_oceano_austral": oa, "abierto_pacifico_norte": pn, "centro": 22.5 * k, "clase": c,
         "dist_min_tierra_modelo_km": dmod}
        for k, (nom, f, dmin, oa, pn, c, dmod) in enumerate([
            ("N", 0.0, 27.0, False, False, "cerrado", 34.0), ("S", 1.0, float("nan"), True, False, "abierto", float("nan")),
            ("O", 1.0, float("nan"), False, True, "abierto", float("nan"))])])
    sectores.to_csv(tmp_path / "exposicion.csv", index=False)
    pd.DataFrame([{"origen": "GEBCO_2026", "lat": -15.0, "lon": -75.46, "area_km2": 0.2, "dist_min_km": 36.0,
                   "rumbo": 90.0, "sector": 4, "clase_sector": "cerrado", "tierra_en_modelo": True}]).to_csv(
        tmp_path / "obstaculos.csv", index=False)
    pd.DataFrame([
        {"mision": "HY-2", "regimen": "desde 1993 (asimilado)", "asimilacion": "no asimilada", "util_para_sesgo": True,
         "inicio": "2011-10-11", "fin": "2018-06-05", "nodo": "principal"},
        {"mision": "SWOT", "regimen": "desde 1993 (asimilado)", "asimilacion": "no verificable", "util_para_sesgo": False,
         "inicio": "2023-07-25", "fin": "2026-06-28", "nodo": "principal"}]).to_csv(tmp_path / "observaciones.csv", index=False)
    cob = pd.DataFrame([{"regimen": "desde 1993 (asimilado)", "pasadas_sitio_N": 18, "esperadas_sobre_P90": 1.8,
                         "esperadas_sobre_P99": 0.18, "cobertura_util_sitio": False, "pasadas_region": 294,
                         "esperadas_sobre_P90_region": 29.4, "nodo": "principal"}])
    cob.to_csv(tmp_path / "cobertura_observaciones.csv", index=False)
    cob.assign(rotulo="SENSIBILIDAD").to_csv(tmp_path / "cobertura_sensibilidad_periodo.csv", index=False)
    pd.DataFrame([("1993-01-01", "corrientes", "GLORYS12", "QUID")], columns=["fecha", "tipo", "descripcion", "referencia"]).to_csv(
        tmp_path / "cronologia_waverys.csv", index=False)
    resumen = {"nodo": {"lat": -15.0, "lon": -75.8}, "respaldo": {"lat": -14.8, "lon": -76.0},
               "dist_costa_natural_earth_km": {"principal": 23.68, "respaldo": 17.6}, "dist_costa_gebco_km": 23.16,
               "teselas_inexistentes": ["GFO/014S-283E"], "regla_respaldo_por_regimen": {"desde 1993 (asimilado)": "mantener el nodo"},
               "ciclones_tropicales": {"radio_km": 1000.0, "sistemas": 0, "total_registros_SP": 77223},
               "gate": {"cumple": True, "fallas": []},
               "gebco_observaciones": {"archivo": "g.nc", "mediciones_200km_sin_gebco": {"principal": 0}}}
    (tmp_path / "resumen_ficha.json").write_text(json.dumps(resumen), encoding="utf-8")
    return tmp_path


FICHA_NODO = {"nodo_elegido": {"lat": -15.0, "lon": -75.8, "prof_nodo_m": 194.0, "prof_min_celda_m": 110.0,
                               "prof_media_celda_m": 289.1, "frac_tierra_celda": 0.0, "dist_iso_200m_km": 0.13,
                               "vecinas_validas": 7, "vecinas_presentes": 8, "zonas_exclusion": "",
                               "zonas_colision": "AAVC frente a Ica", "dist_exclusion_km": 43.7},
              "no_evaluado": ["zonas militares", "franja de 5 millas"]}


def test_campos_del_contrato(carpeta):
    md = ficha_md.componer(carpeta, FICHA_NODO, None, 420.0, "0.1.0")
    for titulo in ("Ubicación y profundidad", "Exposición por sectores", "Obstáculos", "Fenómenos generadores",
                   "Observaciones de altimetría", "Rango plausible", "Cronología", "Zonas y colisión", "Limitaciones"):
        assert titulo in md
    assert "420 m" in md and "194 m" in md
    assert "sesgo conservador no corregido; excepción a ADR-0004 punto 1, decidida en ADR-0015" in md
    assert "**Serie:** no disponible" in md  # campo faltante con su causa


def test_redaccion_de_la_revision(carpeta):
    md = ficha_md.componer(carpeta, FICHA_NODO, None, None, "0.1.0")
    assert "una sola población" not in md
    assert "Supuesto, sin comprobar" in md
    assert "29.4" in md and "no alcanza" not in md.replace("«no alcanza»", "")  # región sin juicio de umbral
    assert "no disponible en este proyecto" in md  # ADR-0019: sin cifras de v2
    assert "v2" not in md
    assert "no disponible (falta `raw/waverys/estatico/`)" in md
    assert " nan " not in md.lower() and "| nan" not in md.lower()


def test_rango_c06_sigue_la_regla():
    assert ficha_md.C06_D == min((x[2] for x in ficha_md.C06_LECTURA), key=lambda v: -abs(v))


def test_franja_de_5_millas_dice_no_aplica_en_todas_partes(carpeta):
    """Revisión de control del 2026-11-15: la §8 no debe listarla como «no evaluada» (ADR-0017)."""
    md = ficha_md.componer(carpeta, FICHA_NODO, None, 420.0, "0.1.0")
    fila = next(l for l in md.splitlines() if l.startswith("| Capas no evaluadas"))
    assert "5 millas" not in fila and "zonas militares" in fila
    assert "| Franja de 5 millas (ADR-0017) | no aplica: el nodo está a 12.5 millas" in md
