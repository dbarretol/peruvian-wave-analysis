"""Cruces descriptivos e independientes de afirmaciones de la F4, la F6 y la F7.

Como ``recalculo.py``, **no importa nada de ``peruvian_wave_analysis``**: lee la serie del NetCDF crudo y usa los
picos propios de ``recalculo.picos`` (rachas sobre u_id unidas con IET de 48 h). Lo publicado se lee solo para
comparar.

- F4: Tp medio con Hs sobre el P99 de cada tramo (antes y después de 1993-02).
- F6: fracción de estados con energía de swell mayor que la de mar de viento; tasa de picos por año en cada era,
  frente a ``picos_provisionales.csv``.
- F7: Tp y dirección de los picos sobre u\\*; años de los 20 mayores picos de E0 (cuántos son anteriores a 1993);
  forma de la tormenta del máximo observado (estados sobre 3.5 m alrededor del máximo).

Escribe ``cruces.json`` en ``--salida`` (por defecto ``<datos>/interim/auditoria/``).

Uso::

    uv run python auditoria/cruces.py --datos C:/WORKSPACES/TESIS-M5-WAVE/WAVE-DATA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recalculo as R  # noqa: E402  (módulo hermano de la auditoría, no del paquete)


def cruzar(raiz: Path) -> dict:
    raiz = Path(raiz)
    pr = raiz / "processed"
    s = R.leer_serie(raiz)
    hs = s["VHM0"].dropna()
    out: dict = {"nota": "Cruces independientes: no importan peruvian_wave_analysis."}

    f4 = {}
    for nombre, a, b in (("T0", None, "1993-01-31"), ("T1", R.INICIO_E1, R.FIN_MODELO)):
        x = s[a:b]
        if len(x):
            q = x["VHM0"].quantile(0.99)
            f4[f"tp_medio_sobre_p99_{nombre}"] = float(x["VTPK"][x["VHM0"] > q].mean())
    out["F4"] = f4

    sw = s["VHM0_SW1"].fillna(0) ** 2 + s["VHM0_SW2"].fillna(0) ** 2
    f6 = {"fraccion_swell_domina": float((sw > s["VHM0_WW"].fillna(0) ** 2).mean())}
    pp = pr / "F06-clima-de-oleaje" / "marcona-200m" / "picos_provisionales.csv"
    if pp.exists():
        t = pd.to_datetime(pd.read_csv(pp)["t_pico"])
        f6["tasa_publicada_T0"] = float((t < R.INICIO_E1).sum() / (13 + 1 / 12))
        f6["tasa_publicada_T1"] = float(((t >= R.INICIO_E1) & (t <= R.FIN_MODELO)).sum() / (32 + 11 / 12))
    out["F6"] = f6

    f7 = {}
    umb = pr / "F07-eventos" / "umbral" / "marcona-200m" / "umbral.json"
    us = json.loads(umb.read_text(encoding="utf-8"))["ramas"] if umb.exists() else {}
    for rama, ini in R.RAMAS.items():
        pk = R.picos(hs[ini:R.FIN_MODELO], R.U_ID, R.IET_H)
        if rama not in us:
            continue
        sel = pk[pk > float(us[rama]["R1b"]["u_m"])]
        tp, dr = s.loc[sel.index, "VTPK"], s.loc[sel.index, "VMDR"]
        f7[rama] = {"picos": len(sel), "tp_mediano_s": float(tp.median()), "tp_min_s": float(tp.min()),
                    "dir_min": float(dr.min()), "dir_max": float(dr.max()), "dir_mediana": float(dr.median())}
    pk0 = R.picos(hs[: R.FIN_MODELO], R.U_ID, R.IET_H)
    top = pk0.nlargest(min(20, len(pk0)))
    f7["mayores_20_E0"] = {"anteriores_a_1993_02": int((top.index < R.INICIO_E1).sum()),
                           "anios": sorted(int(a) for a in top.index.year),
                           "dir_entre_160_y_250": int(s.loc[top.index, "VMDR"].between(160, 250).sum())}
    tmax = hs.idxmax()
    ventana = hs[tmax - pd.Timedelta(days=3): tmax + pd.Timedelta(days=3)]
    f7["maximo_observado"] = {"hs_m": float(hs.max()), "t": str(tmax),
                              "horas_sobre_3_5_m": float((ventana > 3.5).sum() * 3)}
    out["F7"] = f7
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datos", required=True, help="raíz de WAVE-DATA")
    ap.add_argument("--salida", help="carpeta de salida (por defecto <datos>/interim/auditoria)")
    a = ap.parse_args(argv)
    salida = Path(a.salida) if a.salida else Path(a.datos) / "interim" / "auditoria"
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / "cruces.json"
    ruta.write_text(json.dumps(cruzar(Path(a.datos)), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"escrito {ruta}")
    return ruta


if __name__ == "__main__":
    main()
