"""``ficha_sitio.md`` de la F1 (contrato §2): se compone solo con las salidas ya publicadas de la ficha.

Los textos fijos (fenómenos generadores, rango de C-06, limitaciones) citan solo referencias verificadas y
guardadas en ``WAVE-DATA/raw/documentos/`` (ADR-0019: nada de otro repositorio entra como evidencia).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

QUID = "CMEMS-GLO-QUID-001-032 v1.5 (31/05/2024)"
SEMEDO = "Semedo, A., Sušelj, K., Rutgersson, A. y Sterl, A. (2011). A global view on the wind sea and swell climate and variability from ERA-40. *Journal of Climate* 24, 1461–1479"
AGURTO = "Agurto Barragán, G., Collazo, S. y García-Herrera, R. (2026). Meteorological drivers of extreme swells on the Peruvian coast. *Weather and Climate Dynamics* 7, 1779–1796. doi 10.5194/wcd-7-1779-2026"
BARNARD = "Barnard, P. L. et al. (2015). Coastal vulnerability across the Pacific dominated by El Niño/Southern Oscillation. *Nature Geoscience*. doi 10.1038/ngeo2539 (publicado en línea el 21/09/2015)"
KNAPP = "Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J. y Neumann, C. J. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society* 91, 363–376"
RIBAL = "Ribal, A. y Young, I. R. (2019). 33 years of globally calibrated wave height and wind speed data based on altimeter observations. *Scientific Data* 6, 77. doi 10.1038/s41597-019-0083-9"

# C-06: lectura gráfica de la figura 1 del QUID (WAVERYS v1.1 frente a HY-2A, global, 2014-01-01 a 2018-12-26),
# hecha el 2026-11-15 sobre la página 14 renderizada a 600 ppp, antes de ver la serie. Ver §6 de la ficha.
C06_LECTURA = [  # (Hs del altímetro, m; WAVERYS − altímetro, m; relativo, %)
    ("6.5–7.0", -0.05, -0.7), ("7.0–7.5", -0.13, -1.8), ("7.5–8.0", -0.18, -2.3),
    ("8.0–8.5", -0.22, -2.7), ("8.5–9.0", -0.21, -2.4), ("≈ 14.5 (cruz aislada)", +0.18, +1.3),
]
C06_D = -2.7
C06_PRECISION_PP = 1.0


def _f(x, nd=1, vacio="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return vacio
    return f"{x:.{nd}f}"


def _tabla(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return cab + "\n".join("| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)) + "\n"


def componer(d: Path, ficha_nodo: dict, gate_serie: dict | None, deptho_m: float | None, version: str) -> str:
    r = json.loads((d / "resumen_ficha.json").read_text(encoding="utf-8"))
    sect = pd.read_csv(d / "exposicion.csv")
    obst = pd.read_csv(d / "obstaculos.csv")
    obs = pd.read_csv(d / "observaciones.csv")
    cob = pd.read_csv(d / "cobertura_observaciones.csv")
    cob_s = pd.read_csv(d / "cobertura_sensibilidad_periodo.csv")
    crono = pd.read_csv(d / "cronologia_waverys.csv")
    n = ficha_nodo["nodo_elegido"]
    vec = ficha_nodo.get("vecindario_3x3", [])
    L = []
    a = L.append

    a("# Ficha del sitio · marcona-200m\n")
    a(f"Generada por `peruvian_wave_analysis.sitio.ficha` (versión {version}) con el contrato de la F1 v3 (aceptado el "
      "2026-11-15). La primera corrida se hizo con el contrato en borrador v3, de igual texto. Las cifras salen de los "
      "archivos de esta carpeta; ninguna viene de otro proyecto (ADR-0019). **No se miró ningún valor de Hs ni de Tp.**\n")
    a(f"**Gate de la ficha:** {'cumple' if r['gate']['cumple'] else 'NO cumple'}"
      + ("" if r["gate"]["cumple"] else " — " + "; ".join(r["gate"]["fallas"])) + ".\n")

    # 1. Ubicación
    a("## 1. Ubicación y profundidad\n")
    filas = [
        ("Nodo de WAVERYS (ADR-0007)", f"{r['nodo']['lat']}, {r['nodo']['lon']}"),
        ("Respaldo", f"{r['respaldo']['lat']}, {r['respaldo']['lon']}"),
        ("Profundidad en el nodo, GEBCO_2026", f"{_f(n['prof_nodo_m'], 0)} m"),
        ("Profundidad del modelo (`deptho` de WAVERYS)",
         f"{_f(deptho_m, 0)} m" if deptho_m is not None else "no disponible (falta `raw/waverys/estatico/`)"),
        ("Celda del modelo (0.2°), GEBCO", f"mínima {_f(n['prof_min_celda_m'], 0)} m, media {_f(n['prof_media_celda_m'], 0)} m, "
                                           f"tierra {_f(100 * n['frac_tierra_celda'], 0)} %"),
        ("Distancia a la costa", f"{_f(r['dist_costa_gebco_km'])} km (GEBCO); "
                                 f"{_f(r['dist_costa_natural_earth_km']['principal'])} km (Natural Earth 1:10 m, la que usa la clase «sitio»)"),
        ("Distancia a la isóbata de 200 m", f"{_f(n['dist_iso_200m_km'], 2)} km"),
        ("Vecinas con dato", f"{n['vecinas_validas']} de {n['vecinas_presentes']}" + (" (bloque de 3 × 3 en `ficha_nodo.json`)" if vec else "")),
    ]
    a(_tabla(pd.DataFrame(filas, columns=["Campo", "Valor"])))
    a("\nLa diferencia entre 194 m y la profundidad del modelo es la limitación de ADR-0015 (§9).\n")

    # 2. Exposición
    a("## 2. Exposición por sectores\n")
    a("Dirección **desde donde viene el oleaje**. Rayos por gran círculo cada 1°, 23 por sector, hasta 1000 km con "
      "Natural Earth 1:10 m; alcance de cuenca hasta 12 000 km. Mide la geometría, no la llegada efectiva del swell.\n")
    t = pd.DataFrame({
        "Sector": sect["nombre_sector"], "Centro (°)": sect["centro"].map(lambda v: f"{v:.1f}"), "Clase": sect["clase"],
        "Rayos que llegan a 1000 km": sect["fraccion_rayos_1000_km"].map(lambda v: f"{100 * v:.0f} %"),
        "Primera tierra, Natural Earth (km)": sect["dist_min_tierra_km"].map(lambda v: _f(v, 0, "—")),
        "Primera tierra, máscara del modelo (km)": sect["dist_min_tierra_modelo_km"].map(lambda v: _f(v, 0, "—")),
        "Océano Austral": sect["abierto_oceano_austral"].map({True: "sí", False: "no"}),
        "Pacífico Norte": sect["abierto_pacifico_norte"].map({True: "sí", False: "no"}),
    })
    a(_tabla(t))
    ab = sect.loc[sect["clase"] == "abierto", "nombre_sector"].tolist()
    a(f"\n**Abiertos:** {', '.join(ab)}. **Al Océano Austral:** "
      f"{', '.join(sect.loc[sect['abierto_oceano_austral'], 'nombre_sector'])}. **Al Pacífico Norte:** "
      f"{', '.join(sect.loc[sect['abierto_pacifico_norte'], 'nombre_sector'])}. Un sector abierto que no llega a ninguna "
      "cuenca (p. ej. OSO) es geometría: su gran círculo no baja de unos 27° S ni cruza el ecuador.\n")
    a("«—» en la máscara del modelo: el rayo sale de la malla descargada sin tocar una celda sin dato.\n")

    # 3. Obstáculos
    a("## 3. Obstáculos (islas e islotes a menos de 50 km)\n")
    if len(obst):
        a(_tabla(pd.DataFrame({
            "Lat": obst["lat"].map(lambda v: f"{v:.3f}"), "Lon": obst["lon"].map(lambda v: f"{v:.3f}"),
            "Área (km²)": obst["area_km2"].map(lambda v: f"{v:.2f}"), "Distancia (km)": obst["dist_min_km"].map(lambda v: f"{v:.1f}"),
            "Rumbo (°)": obst["rumbo"].map(lambda v: f"{v:.0f}"), "Clase del sector": obst["clase_sector"],
            "Tierra en el modelo": obst["tierra_en_modelo"].map({True: "sí", False: "no"})})))
        a("\nNinguno está en un sector abierto: no hacen sombra a las direcciones abiertas.\n" if
          not obst["clase_sector"].eq("abierto").any() else "\n**Hay obstáculos en sectores abiertos.**\n")
    else:
        a("Ninguno.\n")
    a("Lo menor que un píxel de GEBCO de 15″ (≈ 460 m) no se detecta.\n")

    # 4. Fenómenos
    a("## 4. Fenómenos generadores (lista bibliográfica; contrato §3.3)\n")
    a("Solo bibliografía verificada en su texto (copias en `raw/documentos/articulos/`). La comprobación con la serie "
      "es de la F3 o la F6; la posible población mixta va a la F8–F11.\n")
    fen = [
        ("Swell del Océano Austral", "En invierno austral (JJA) el Hs del Pacífico oriental está dominado por swell del Océano Austral. "
         "Los eventos de swell extremo más fuertes en la costa peruana son del suroeste, en invierno austral, generados por ciclones extratropicales profundos (5 casos; viaje de 3–4 d)",
         "Semedo et al. (2011) pp. 1465–1466; Agurto et al. (2026) resumen", "verificada"),
        ("Swell del Pacífico Norte en el verano austral", "Seis eventos de swell extremo del noroeste en la costa peruana, en invierno boreal, generados por bajas profundas del Pacífico Norte central y occidental (viaje de 8–11 d)",
         "Agurto et al. (2026) resumen", "verificada"),
        ("Frentes y sistemas extratropicales", "Ciclón extratropical profundo, de estructura vertical apilada, con corredor persistente de vientos del oeste sobre la zona de generación, antes del pico de swell en la costa",
         "Agurto et al. (2026) resumen", "verificada"),
        ("Mar de viento local", "El jet costero de bajo nivel frente a Perú reduce la probabilidad de dominio del swell (firma regional del viento). "
         "Alisios del sureste y viento costero del sur: sin referencia verificada para el sitio",
         "Semedo et al. (2011) p. 1468", "verificada en parte"),
        ("El Niño y La Niña", "La erosión costera en el Pacífico varía sobre todo con ENSO y las anomalías de oleaje se correlacionan con índices climáticos, "
         "pero el estudio no incluye playas de Sudamérica: no dice nada del sitio",
         "Barnard et al. (2015) resumen", "sin referencia verificada para la costa peruana"),
        ("Ciclones tropicales", f"Sin ciclones tropicales a menos de {r['ciclones_tropicales']['radio_km']:.0f} km: "
         f"{r['ciclones_tropicales']['sistemas']} sistemas en {r['ciclones_tropicales']['total_registros_SP']} registros de la cuenca del Pacífico Sur de IBTrACS v04r01",
         "IBTrACS (Knapp et al., 2010), consulta propia", "no aplica"),
    ]
    a(_tabla(pd.DataFrame(fen, columns=["Fenómeno", "Qué se sabe", "Referencia", "Estado"])))
    a("\n**Lo que esta lista no dice.** La ausencia de ciclones tropicales solo descarta esa población. Sigue abierta la "
      "mezcla de swell del Océano Austral, swell del Pacífico Norte y mar de viento local. **Supuesto, sin comprobar con "
      "la serie:** que los extremos del sitio los domina el swell del sur. Es razonable por la exposición (§2) y por la "
      "bibliografía, pero lo comprueba la F3 o la F6.\n")

    # 5. Observaciones
    a("## 5. Observaciones de altimetría (contrato §3.4)\n")
    a("Fuente: Ribal y Young (2019), AODN. Clase «sitio»: ≤ 25 km del nodo, a la costa al menos tanto "
      "como el nodo (Natural Earth), profundidad GEBCO > 100 m, sector abierto o parcial. Clase «región»: ≤ 200 km, "
      "≥ 10 km de la costa, > 100 m, sector abierto o parcial. Independencia según la tabla 3 del QUID (§3.4).\n")
    g = r.get("gebco_observaciones", {})
    if g:
        sin = g.get("mediciones_200km_sin_gebco", {})
        a(f"Profundidad: `{g.get('archivo')}`; mediciones a ≤ 200 km sin GEBCO: "
          + ", ".join(f"{k} {v}" for k, v in sin.items()) + ".\n")
    a("\n**Regla del contrato** (pasadas de misiones no asimiladas):\n")
    tc = pd.DataFrame({
        "Nodo": cob["nodo"], "Régimen": cob["regimen"], "Pasadas «sitio» (N)": cob["pasadas_sitio_N"],
        "0.10·N (sobre P90)": cob["esperadas_sobre_P90"].map(lambda v: f"{v:.2f}"),
        "0.01·N (sobre P99)": cob["esperadas_sobre_P99"].map(lambda v: f"{v:.2f}"),
        "Cobertura útil (0.10·N ≥ 30)": cob["cobertura_util_sitio"].map({True: "sí", False: "no"}),
        "Pasadas «región»": cob["pasadas_region"], "0.10·N región": cob["esperadas_sobre_P90_region"].map(lambda v: f"{v:.1f}"),
    })
    a(_tabla(tc))
    util = bool(cob["cobertura_util_sitio"].any())
    a("\n**Resultado:** " + ("hay cobertura útil en algún régimen." if util else
      "ningún régimen tiene cobertura útil en el sitio, ni en el nodo principal ni en el respaldo. **El sesgo de la "
      "fuente no se puede medir en el sitio**; se aplica ADR-0004 punto 1 con el rango plausible de §6.") + "\n")
    a("La clase «región» se informa como evidencia regional, **no como validación del sitio**. El contrato no le fija "
      "umbral, así que sus cifras van sin juicio de «alcanza» o «no alcanza»; aplicar el 30 por analogía sería una "
      "regla nueva, posterior a ver los conteos.\n")
    hy = obs[(obs["nodo"] == "principal") & (obs["mision"] == "HY-2") & obs["util_para_sesgo"]]
    if len(hy):
        a(f"\n**Evidencia regional independiente desde 1993:** una sola misión, HY-2A ({str(hy['inicio'].iloc[0])[:7]} a "
          f"{str(hy['fin'].iloc[0])[:7]}). No dice nada de 1993–2010, y es la misma misión con la que el QUID validó "
          "WAVERYS: el rango de §6 y una comparación regional con HY-2A **no son dos evidencias independientes**.\n")
    nv = sorted(obs.loc[obs["asimilacion"] == "no verificable", "mision"].unique())
    if nv:
        a(f"\nMisiones «no verificables» (no figuran en el QUID; se tratan como asimiladas): {', '.join(nv)}.\n")
    a(f"\n**Regla del respaldo:** " + "; ".join(f"{k}: {v}" for k, v in r["regla_respaldo_por_regimen"].items()) + ".\n")
    a(f"\nTeselas sin archivo en AODN: {len(r.get('teselas_inexistentes', []))}. El código solo las marca así ante un "
      "404; dos de GFO en mar abierto (014S-283E, 015S-283E) se confirmaron el 2026-11-15 en el catálogo THREDDS, que no las lista.\n")
    a("\n**Sensibilidad, posterior a ver los conteos** (revisión de control del 2026-11-15; no es la regla del contrato): "
      "los tramos de misiones de la tabla 3 fuera de su período de asimilación (SARAL antes de 2016-03, GFO después de "
      "2006-08, CFOSAT antes de 2020-02) cuentan como no asimilados.\n\n")
    a(_tabla(pd.DataFrame({
        "Nodo": cob_s["nodo"], "Régimen": cob_s["regimen"], "Pasadas «sitio» (N)": cob_s["pasadas_sitio_N"],
        "0.10·N": cob_s["esperadas_sobre_P90"].map(lambda v: f"{v:.2f}"),
        "Cobertura útil": cob_s["cobertura_util_sitio"].map({True: "sí", False: "no"}),
        "Pasadas «región»": cob_s["pasadas_region"], "0.10·N región": cob_s["esperadas_sobre_P90_region"].map(lambda v: f"{v:.1f}")})))
    a("\nCon la sensibilidad la conclusión del sitio no cambia; la muestra regional crece.\n" if not cob_s["cobertura_util_sitio"].any()
      else "\n**Con la sensibilidad algún régimen alcanza cobertura útil: requiere decisión del usuario.**\n")

    # 6. C-06
    a("## 6. Rango plausible del sesgo de la fuente (C-06)\n")
    a(f"El {QUID} §IV.1.1 valida WAVERYS frente a HY-2A (no asimilada), global, del 2014-01-01 al 2018-12-26, con "
      "9 425 957 observaciones. No da cifras de los cuantiles 99 a 99.9, ni por región ni global (su figura 3 da solo "
      "sesgo e índice de dispersión por bandas de latitud). Por la alternativa del contrato, d se lee de las cruces "
      "rosadas (cuantiles 99 a 99.9) de la figura 1, panel de WAVERYS, frente a la diagonal. **Lectura gráfica** "
      "(página 14 renderizada a 600 ppp, centroides de los píxeles rosados; el 2026-11-15, antes de ver la serie):\n\n")
    a(_tabla(pd.DataFrame([(h, f"{dm:+.2f}", f"{dp:+.1f} %") for h, dm, dp in C06_LECTURA],
                          columns=["Hs del altímetro (m)", "WAVERYS − altímetro (m)", "Relativo"])))
    a(f"\nRegla del contrato: el de mayor valor absoluto, **d = {C06_D:+.1f} %**. **Rango plausible: "
      f"[{min(0, C06_D):+.1f} %, {max(0, C06_D):+.1f} %]**".replace("+0.0 %", "0 %").replace("-0.0 %", "0 %") + ", es decir, WAVERYS podría subestimar la cola hasta en "
      f"≈ {abs(C06_D):.1f} %. Precisión de la lectura ≈ ±{C06_PRECISION_PP:.0f} punto porcentual (figura en raster; las "
      "líneas tapan parte de las cruces). Es una cifra **global**, no del Pacífico sureste. La cruz aislada de ≈ 14.5 m "
      "queda por encima de la diagonal; no cambia d.\n")
    a("\nComparación con una boya: **no disponible en este proyecto**. La boya pública más cercana (NDBC 32012, clase "
      "«cuenca», a unos 1000 km) no se ha descargado ni analizado aquí; por ADR-0019 no se cita lo medido en otro "
      "proyecto. Si se quiere, la F5 la descarga y la rehace.\n")

    # 7. Cronología y serie
    a("## 7. Cronología de WAVERYS y serie descargada\n")
    a(_tabla(crono.rename(columns={"fecha": "Fecha", "tipo": "Tipo", "descripcion": "Descripción", "referencia": "Referencia"})))
    if gate_serie:
        c = gate_serie.get("catalogo", {})
        a(f"\n**Serie (ADR-0016):** `{c.get('dataset_id')}`, versión {', '.join(gate_serie.get('versiones', []))}, "
          f"{gate_serie['inicio']} a {gate_serie['fin']}, {gate_serie['pasos']} pasos de 3 h de {gate_serie['pasos_esperados']} "
          f"esperados, {gate_serie['duplicados']} duplicados, {gate_serie['faltantes']} faltantes, {gate_serie['archivos']} "
          f"archivos anuales, {gate_serie['celdas_con_dato']} de {gate_serie['celdas']} celdas del 5 × 5 con dato, "
          f"{len(c.get('variables', []))} variables sin faltantes en el nodo. Gate de la serie: "
          f"{'cumple' if gate_serie.get('cumple') else 'NO cumple'}.\n")
    else:
        a("\n**Serie:** no disponible (falta `serie_gate.json`).\n")

    # 8. Zonas
    a("## 8. Zonas y colisión (ADR-0012)\n")
    a(_tabla(pd.DataFrame([
        ("Zonas de exclusión que tocan el nodo", n.get("zonas_exclusion") or "ninguna"),
        ("Distancia a la zona de exclusión más cercana", f"{_f(n.get('dist_exclusion_km'))} km"),
        ("Zonas de colisión (se declaran, no excluyen)", n.get("zonas_colision") or "ninguna"),
        ("Capas no evaluadas (ADR-0017)", ", ".join(ficha_nodo.get("no_evaluado", [])) or "ninguna"),
    ], columns=["Campo", "Valor"])))

    # 9. Limitaciones
    a("\n## 9. Limitaciones\n")
    a(f"1. **Profundidad intermedia (ADR-0015): sesgo conservador no corregido; excepción a ADR-0004 punto 1, decidida en "
      f"ADR-0015.** El modelo ve {_f(deptho_m, 0)} m y el sitio tiene {_f(n['prof_nodo_m'], 0)} m. Por teoría lineal, el Hs "
      "del modelo puede exceder el que corresponde a 194 m en ≈ 3 % (Tp = 18 s) a ≈ 5 % (Tp = 20 s). La F3 informa la "
      "fracción de estados sobre el P99 con Tp ≥ 16 s y ≥ 18 s y la razón media Ks(194 m)/Ks(420 m).\n")
    a("2. **Capas no evaluadas (ADR-0017):** la franja de 5 millas no aplica (DS 002-2025-PRODUCE, artículo 63.1: «las "
      "primeras cinco millas marinas adyacentes a la costa»; el nodo está a 12.5 millas). Las demás capas sin evaluar son "
      "un límite del estudio, no un descarte.\n")
    a("3. **Un solo reanálisis (ADR-0018):** sin reanálisis de contraste.\n")
    a("4. **Validación de la cola solo regional:** sin cobertura útil en el sitio (§5); el sesgo de la fuente entra como "
      "rango plausible global (§6).\n")
    a("5. **Producto no homogéneo:** 1980–1992 sin asimilación ni corrientes; enero de 1993 es una mezcla lineal (§7). "
      "La F4 fija el período utilizable.\n")
    a("6. **Exposición geométrica:** los sectores miden la geometría, no la llegada efectiva del swell. Islotes menores "
      "de ≈ 460 m no se detectan.\n")

    # 10. Desvíos
    a("\n## 10. Desvíos y notas de proceso\n")
    a("- La descarga de la serie y la primera corrida de la ficha se hicieron con el contrato en borrador v3, de igual texto que el aceptado.\n")
    a("- La primera corrida tomó la profundidad SRTM30+ de los archivos de AODN fuera del recorte de GEBCO (desvío del "
      "contrato). Se corrigió con un recorte ampliado (`fuentes-de-datos.md` §5.1); con la regla del contrato, los conteos "
      "de misiones no asimiladas salieron iguales.\n")
    a("- La sensibilidad por período de asimilación (§5) es posterior a ver los conteos y va rotulada.\n")
    a("- El rango de C-06 es una lectura gráfica, como prevé el contrato cuando el QUID no da cifras.\n")

    # 11. Referencias
    a("\n## 11. Referencias (verificadas; copia en `raw/documentos/`)\n")
    for ref in [f"{QUID}, §II, §IV.1.1 (figuras 1 a 3), §VI.4 y tabla 3", "CMEMS-GLO-PUM-001-032 v1.6 (02/2026)",
                "GEBCO Compilation Group (2026). GEBCO 2026 Grid. doi 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa",
                RIBAL, SEMEDO, AGURTO, BARNARD, KNAPP,
                "DS 002-2025-PRODUCE (El Peruano, 27/02/2025), artículo 63.1 del Reglamento de la Ley General de Pesca"]:
        a(f"- {ref}\n")
    return "\n\n".join(s.strip("\n") for s in L) + "\n"


def deptho_en(ruta: Path | None, lat: float, lon: float) -> float | None:
    if ruta is None or not Path(ruta).exists():
        return None
    import xarray as xr

    with xr.open_dataset(ruta) as ds:
        nl = "latitude" if "latitude" in ds.coords else "lat"
        nn = "longitude" if "longitude" in ds.coords else "lon"
        v = ds["deptho"].sel({nl: lat, nn: lon}, method="nearest").values
    v = float(np.asarray(v).squeeze())
    return v if np.isfinite(v) else None
