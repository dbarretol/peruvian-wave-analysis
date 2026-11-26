"""Prueba mínima de la auditoría independiente (``auditoria/``), sobre datos sintéticos en un directorio temporal."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CARPETA = Path(__file__).resolve().parents[1] / "auditoria"


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(f"auditoria_{nombre}", CARPETA / f"{nombre}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _raiz_sintetica(tmp_path: Path) -> Path:
    raiz = tmp_path / "WAVE-DATA"
    serie = raiz / "raw" / "waverys" / "serie"
    serie.mkdir(parents=True)
    t = pd.date_range("1988-01-01", "1999-12-31 21:00", freq="3h")
    rng = np.random.default_rng(7)
    hs = 2.0 + 0.3 * np.sin(2 * np.pi * np.asarray(t.dayofyear) / 365.25) + 0.25 * rng.standard_normal(len(t))
    # tormentas: una cada ~40 días, con pico GPD sobre 2.9 m y forma triangular de 2 días
    for c in range(0, len(t) - 16, 320):
        pico = 2.9 + rng.pareto(8.0) * 0.6
        hs[c : c + 16] = np.maximum(hs[c : c + 16], pico - 0.12 * np.abs(np.arange(16) - 8))
    valores = {"VHM0": np.round(hs, 2), "VHM0_WW": np.full(len(t), 0.5), "VHM0_SW1": hs * 0.95,
               "VHM0_SW2": np.full(len(t), 0.2), "VTPK": np.full(len(t), 15.0), "VMDR": np.full(len(t), 200.0)}
    d = {v: (("time", "latitude", "longitude"), np.reshape(val, (-1, 1, 1))) for v, val in valores.items()}
    xr.Dataset(d, coords={"time": t, "latitude": [-15.0], "longitude": [-75.8]}).to_netcdf(serie / "sintetica.nc")
    umb = raiz / "processed" / "F07-eventos" / "umbral" / "marcona-200m"
    umb.mkdir(parents=True)
    r1b = {"u_m": 3.0, "n": None, "xi": None, "rl100": None, "rl100_lo": None, "rl100_hi": None}
    (umb / "umbral.json").write_text(json.dumps({"ramas": {k: {"R1b": r1b} for k in
                                                           ("E1_sin_corregir", "E0_sin_corregir")}}), encoding="utf-8")
    return raiz


def test_no_importa_el_paquete():
    for f in ("recalculo", "cruces"):
        arbol = ast.parse((CARPETA / f"{f}.py").read_text(encoding="utf-8"))
        nombres = {a.name for n in ast.walk(arbol) if isinstance(n, ast.Import) for a in n.names}
        nombres |= {n.module or "" for n in ast.walk(arbol) if isinstance(n, ast.ImportFrom)}
        assert not any(x.startswith("peruvian_wave_analysis") for x in nombres), f


def test_picos_une_rachas_dentro_del_iet():
    R = _cargar("recalculo")
    t = pd.date_range("2000-01-01", periods=40, freq="3h")
    x = pd.Series(1.0, index=t)
    x.iloc[2:5] = [3.0, 3.5, 3.1]  # racha 1
    x.iloc[10:12] = [3.2, 3.8]  # 15 h después: se une con la 1
    x.iloc[35:37] = [3.3, 3.0]  # 69 h después: evento nuevo
    p = R.picos(x, 2.7, 48)
    assert list(p.round(2)) == [3.8, 3.3]
    assert p.index[0] == t[11]


def test_recalculo_y_cruces_de_punta_a_punta(tmp_path):
    raiz = _raiz_sintetica(tmp_path)
    antes = {p for p in raiz.rglob("*")}
    salida = tmp_path / "salida"
    R, C = _cargar("recalculo"), _cargar("cruces")
    r = json.loads(R.main(["--datos", str(raiz), "--salida", str(salida)]).read_text(encoding="utf-8"))
    c = json.loads(C.main(["--datos", str(raiz), "--salida", str(salida)]).read_text(encoding="utf-8"))
    assert {p for p in raiz.rglob("*")} == antes  # no escribe en la raíz de datos cuando hay --salida
    assert r["F3"]["registros"] == len(pd.date_range("1988-01-01", "1999-12-31 21:00", freq="3h"))
    for rama in ("E1_sin_corregir", "E0_sin_corregir"):
        f = r["F7"][rama]
        assert f["picos_sobre_u"] > 20
        assert f["rl100_lo"] <= f["rl100"] <= f["rl100_hi"]
        assert c["F7"][rama]["dir_mediana"] == 200.0
    assert c["F7"]["mayores_20_E0"]["dir_entre_160_y_250"] == 20
