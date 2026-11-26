"""Recálculo independiente de las cifras principales de la F1 a la F7.

**No importa nada de ``peruvian_wave_analysis``:** si lo importara, dejaría de ser independiente. Usa solo numpy,
pandas, scipy y xarray, con implementaciones propias y simples:

- la serie del nodo se lee del NetCDF crudo de WAVERYS (``raw/waverys/serie/*.nc``), no de la serie publicada;
- los eventos son rachas sobre u_id unidas si el hueco entre ellas es menor que el IET (sin la regla de bajada
  intermedia del contrato de la F7, §3.2). Por eso se compara con la sensibilidad publicada «sin bajada»;
- la GPD se ajusta por máxima verosimilitud con scipy, y el IC del RL100 es de perfil, buscado en una rejilla.

Lo publicado se lee solo para compararlo. Lo que no se recalcula (los umbrales u\\*, que son decisiones de la F7) se
toma de ``umbral.json``. Si una salida publicada no existe, su comparación queda como ``null``.

Escribe un único JSON en ``--salida`` (por defecto ``<datos>/interim/auditoria/``). No escribe en ``raw/`` ni en
``processed/``.

Uso::

    uv run python auditoria/recalculo.py --datos C:/WORKSPACES/TESIS-M5-WAVE/WAVE-DATA
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize, stats

LAT, LON = -15.0, -75.8
U_ID = 2.70
IET_H = 48.0
INICIO_E1 = "1993-02-01"
FIN_MODELO = "2025-12-31"
RAMAS = {"E1_sin_corregir": INICIO_E1, "E0_sin_corregir": "1980-01-01"}
VARIABLES = ("VHM0", "VHM0_WW", "VHM0_SW1", "VHM0_SW2", "VTPK", "VMDR")


def leer_serie(raiz: Path) -> pd.DataFrame:
    """Serie del nodo desde los NetCDF crudos (uno por año), sin duplicados."""
    archivos = sorted(glob.glob(str(Path(raiz) / "raw" / "waverys" / "serie" / "*.nc")))
    if not archivos:
        raise FileNotFoundError("no hay NetCDF en raw/waverys/serie/")
    partes = []
    for f in archivos:
        with xr.open_dataset(f) as d:
            v = [x for x in VARIABLES if x in d]
            p = d[v].sel(latitude=LAT, longitude=LON, method="nearest").to_dataframe()
            partes.append(p[v])
    s = pd.concat(partes).sort_index()
    return s[~s.index.duplicated()]


def cuantil_mensual(x: pd.Series, q: float) -> float:
    """Cuantil en el que cada mes del calendario pesa lo mismo."""
    m = pd.Series(x.index.month)
    w = 1.0 / m.map(m.value_counts()).to_numpy()
    o = np.argsort(x.to_numpy(), kind="stable")
    xs, ws = x.to_numpy()[o], w[o]
    c = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return float(np.interp(q, c, xs))


def picos(x: pd.Series, u_id: float, iet_h: float) -> pd.Series:
    """Rachas sobre u_id unidas si el hueco es menor o igual que el IET; un pico (el máximo) por evento."""
    sobre = x[x > u_id]
    if sobre.empty:
        return pd.Series(dtype=float)
    t = sobre.index
    nuevo = np.r_[True, (t[1:] - t[:-1]) / pd.Timedelta(hours=1) > iet_h]
    g = sobre.groupby(np.cumsum(nuevo))
    return pd.Series(g.max().to_numpy(), index=pd.DatetimeIndex(g.idxmax().to_numpy()))


def nivel_retorno(xi: float, sigma: float, lam: float, t_anios: float = 100.0) -> float:
    """Nivel de retorno sobre el umbral (exceso) para T años con tasa lam por año."""
    m = lam * t_anios
    return sigma * np.log(m) if abs(xi) < 1e-9 else sigma / xi * (m**xi - 1.0)


def ajuste_e_ic(excesos: np.ndarray, lam: float, u: float, t_anios: float = 100.0) -> dict:
    """GPD por MV (scipy) y IC de perfil del 95 % del RL_T, buscado en una rejilla de RL."""
    xi0, _, s0 = stats.genpareto.fit(excesos, floc=0)
    m = lam * t_anios
    z0 = nivel_retorno(xi0, s0, lam, t_anios)
    ll0 = stats.genpareto.logpdf(excesos, xi0, 0, s0).sum()

    def nll(xi: float, z: float) -> float:
        sig = z / np.log(m) if abs(xi) < 1e-9 else z * xi / (m**xi - 1.0)
        if sig <= 0:
            return 1e10
        v = stats.genpareto.logpdf(excesos, xi, 0, sig)
        return 1e10 if not np.all(np.isfinite(v)) else -float(v.sum())

    def perfil(z: float) -> float:
        r = optimize.minimize_scalar(lambda xi: nll(xi, z), bounds=(-0.95, 1.5), method="bounded")
        return -r.fun

    corte = ll0 - stats.chi2.ppf(0.95, 1) / 2.0
    zs = np.linspace(z0 * 0.6, z0 * 4.0, 600)
    dentro = zs[np.array([perfil(z) for z in zs]) >= corte]
    return {
        "xi": float(xi0),
        "sigma": float(s0),
        "rl100": float(u + z0),
        "rl100_lo": float(u + dentro.min()) if dentro.size else None,
        "rl100_hi": float(u + dentro.max()) if dentro.size else None,
    }


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _cerca(a, b, tol: float):
    return None if a is None or b is None else bool(abs(a - b) <= tol)


def recalcular(raiz: Path) -> dict:
    raiz = Path(raiz)
    pr = raiz / "processed"
    s = leer_serie(raiz)
    hs = s["VHM0"].dropna()
    out: dict = {"nota": "Recálculo independiente: no importa peruvian_wave_analysis."}

    # F1: profundidad del nodo en GEBCO
    gebco = sorted(glob.glob(str(raiz / "raw" / "batimetria" / "gebco" / "*.nc")))
    if gebco:
        with xr.open_dataset(gebco[-1]) as g:
            out["F1"] = {"gebco_m": float(g["elevation"].sel(lat=LAT, lon=LON, method="nearest"))}

    # F2: bandas de la verificación (lectura de lo publicado)
    v = _json(pr / "F02-biblioteca-comun" / "verificacion" / "resumen.json")
    if v:
        b = [x for e in v["escenarios"] for x in e["bandas"]]
        out["F2"] = {"bandas_cumplen": sum(bool(x["cumple"]) for x in b), "bandas": len(b)}

    # F3: registros, igualdad con la serie publicada y cuantiles
    f3 = {"registros": len(hs), "nan_hs": int(s["VHM0"].isna().sum()), "hs_max_m": float(hs.max()),
          "hs_max_t": str(hs.idxmax()), "p95_mensual": cuantil_mensual(hs, 0.95),
          "p99_mensual": cuantil_mensual(hs, 0.99)}
    pq = pr / "F03-datos" / "serie" / "marcona-200m" / "serie_nodo.parquet"
    if pq.exists():
        pub = pd.read_parquet(pq, columns=["VHM0"])
        comun = hs.index.intersection(pub.index)
        f3["comunes_con_publicada"] = len(comun)
        f3["max_dif_con_publicada_m"] = float((hs.loc[comun] - pub.loc[comun, "VHM0"]).abs().max())
    out["F3"] = f3

    # F4: razón de cuantiles después/antes de 1993-02
    t0, t1 = hs[: "1993-01-31"], hs[INICIO_E1:FIN_MODELO]
    if len(t0) and len(t1):
        out["F4"] = {f"razon_q{q}": cuantil_mensual(t1, q) / cuantil_mensual(t0, q) for q in (0.9, 0.99, 0.999)}

    # F5: factor de Hs a partir del IC de B en el P99 (lectura de lo publicado)
    r5 = _json(pr / "F05-validacion-externa" / "validacion" / "marcona-200m" / "resumen_f05.json")
    if r5 and "ic95_B_q" in json.dumps(r5):
        def buscar(o):
            if isinstance(o, dict):
                if "ic95_B_q" in o:
                    return o["ic95_B_q"]
                for w in o.values():
                    r = buscar(w)
                    if r is not None:
                        return r
            return None
        lo, hi = buscar(r5)["P99"]
        out["F5"] = {"factor_ic95": [float(np.exp(-hi)), float(np.exp(-lo))]}

    # F7: picos propios, GPD y GEV de máximos anuales
    umb = _json(pr / "F07-eventos" / "umbral" / "marcona-200m" / "umbral.json")
    sen_p = pr / "F07-eventos" / "sensibilidades" / "marcona-200m" / "sensibilidades.csv"
    sen = pd.read_csv(sen_p) if sen_p.exists() else None
    f7 = {}
    for rama, ini in RAMAS.items():
        if not umb or rama not in umb.get("ramas", {}):
            f7[rama] = None
            continue
        pub = umb["ramas"][rama]["R1b"]
        u = float(pub["u_m"])
        x = hs[ini:FIN_MODELO]
        pk = picos(x, U_ID, IET_H)
        sel = pk[pk > u]
        anios = (pd.Timestamp(FIN_MODELO) + pd.Timedelta(days=1) - pd.Timestamp(ini)) / pd.Timedelta(days=365)
        lam = len(sel) / anios
        r = {"u_m": u, "eventos": len(pk), "picos_sobre_u": len(sel), "anios": anios, "lambda": lam}
        r.update(ajuste_e_ic(sel.to_numpy() - u, lam, u) if len(sel) >= 5 else {})
        r["publicado_R1b"] = {k: pub.get(k) for k in ("n", "xi", "rl100", "rl100_lo", "rl100_hi")}
        if sen is not None:
            sb = sen[(sen.sensibilidad == "sin_bajada") & (sen.rama == rama)]
            if len(sb):
                sb = sb.iloc[0]
                r["publicado_sin_bajada"] = {"n": int(sb.n_sobre_u), "xi": float(sb.xi), "rl100": float(sb.rl100),
                                             "rl100_lo": float(sb.rl100_lo), "rl100_hi": float(sb.rl100_hi)}
                r["coincide_sin_bajada"] = {
                    "n": int(sb.n_sobre_u) == len(sel),
                    "rl100": _cerca(r.get("rl100"), float(sb.rl100), 0.01),
                    "ic": _cerca(r.get("rl100_lo"), float(sb.rl100_lo), 0.05)
                    and _cerca(r.get("rl100_hi"), float(sb.rl100_hi), 0.1),
                }
        am = x.groupby(x.index.year).max()
        primero = int(ini[:4]) if ini.endswith("-01-01") else int(ini[:4]) + 1  # solo años completos
        am = am[(am.index >= primero) & (am.index <= int(FIN_MODELO[:4]))]
        if len(am) >= 10:
            c, loc, sc = stats.genextreme.fit(am.to_numpy())
            r["gev_maximos_anuales"] = {"n": len(am), "xi": float(-c),
                                        "rl100": float(stats.genextreme.ppf(0.99, c, loc, sc)),
                                        "nota": "cifra suelta, sin IC: no es un contraste formal (va a la F9)"}
        f7[rama] = r
    out["F7"] = f7
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datos", required=True, help="raíz de WAVE-DATA")
    ap.add_argument("--salida", help="carpeta de salida (por defecto <datos>/interim/auditoria)")
    a = ap.parse_args(argv)
    salida = Path(a.salida) if a.salida else Path(a.datos) / "interim" / "auditoria"
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / "recalculo.json"
    ruta.write_text(json.dumps(recalcular(Path(a.datos)), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"escrito {os.fspath(ruta)}")
    return ruta


if __name__ == "__main__":
    main()
