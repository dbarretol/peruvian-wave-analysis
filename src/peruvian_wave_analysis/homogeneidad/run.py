"""Etapas publicadas de la F4: ``homogeneidad-altimetros`` y ``homogeneidad`` (contrato de la F4, v7, §2).

1. ``homogeneidad-altimetros``: lee las teselas de Ribal y Young (solo posición, hora y bandera; no Hs), verifica
   sus hashes contra el manifiesto de ``raw/`` (G2) y publica la exposición por paso de 3 h.
2. ``homogeneidad``: todo lo demás. **Orden verificable (G4):** calcula y escribe ``potencia.csv`` antes de
   calcular ningún estadístico observado; ``OrdenEscritura`` lo hace cumplir y el manifiesto registra el orden
   de escritura. (El contrato dice «marcas de tiempo»; se registra el orden porque el manifiesto no lleva marcas de
   tiempo, para que dos corridas sean idénticas byte a byte, G7.)

Uso::

    uv run homogeneidad-altimetros --datos C:/WORKSPACES/TESIS-M5-WAVE/WAVE-DATA
    uv run homogeneidad --datos C:/WORKSPACES/TESIS-M5-WAVE/WAVE-DATA
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import manifiesto, rutas
from ..comun import etapa, json_estricto, orquestador, semillas
from ..comun.errores import GateFallido
from ..datos import carga
from ..datos import detector as D
from ..datos.run import ruta_candidatos, ruta_serie
from ..extremos import decluster
from ..sitio import observaciones
from . import altimetros as A
from . import config as C
from . import cronologia, descriptivos, familias, figuras, relevante, tendencia
from . import permutacion as P

NOMBRE = "marcona-200m"
DIR_ETAPA = Path(__file__).resolve().parent
ETAPA_ALTIMETROS = orquestador.Etapa("homogeneidad-altimetros", DIR_ETAPA, Path("."))
ETAPA = orquestador.Etapa("homogeneidad", DIR_ETAPA, Path("."))
VENTANAS = {"c": (*C.VENTANA_CENTRADA, False), "r": (*C.VENTANA_ATRAS, True)}


def ruta_altimetros(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F04-homogeneidad" / "altimetros" / nombre


def ruta_homogeneidad(raiz: Path, nombre: str = NOMBRE) -> Path:
    return Path(raiz) / "processed" / "F04-homogeneidad" / "homogeneidad" / nombre


def _csv(df: pd.DataFrame, ruta: Path, index: bool = False) -> None:
    df.to_csv(ruta, index=index, lineterminator="\n", date_format="%Y-%m-%d %H:%M", float_format="%.6g")


# --- Etapa 1: exposición a altímetros ---------------------------------------------------------


def altimetros(raiz_datos=None, nombre: str = NOMBRE) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    raw = rutas.raw(raiz)
    entradas = manifiesto.leer(raw)
    rel = sorted(r for r in entradas if r.startswith("altimetria/ribal_young/") and r.endswith(".nc"))
    insumos = {f"alt:{r}": raw / r for r in rel}
    esperados = {f"alt:{r}": entradas[r]["sha256"] for r in rel}
    dir_s = ruta_serie(raiz, nombre)
    etapa.exigir_publicada(dir_s)
    insumos["serie"] = dir_s / "serie_nodo.parquet"
    etapa.exigir_insumos(insumos, esperados)  # G2: hash distinto del manifiesto de raw → error
    salida = ruta_altimetros(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA_ALTIMETROS)
    with etapa.staging_protegido(salida, raices) as st:
        rejilla = pd.read_parquet(insumos["serie"], columns=["valido"]).index
        med = observaciones.leer([insumos[k] for k in insumos if k.startswith("alt:")])
        pas = A.pasadas(med, *C.NODO, C.RADIO_KM, C.CORTE_PASADA_S)
        expo = exposicion_por_paso(rejilla, pas)
        locales = (
            pas.groupby("mision")
            .agg(inicio=("tiempo", "min"), fin=("tiempo", "max"), pasadas=("tiempo", "size"))
            .reset_index()
        )
        expo.to_parquet(st / "exposicion_altimetros.parquet")
        _csv(pas, st / "pasadas.csv")
        _csv(locales, st / "misiones_locales.csv")
        misiones_dir = sorted({r.split("/")[2] for r in rel})
        resumen = {
            "archivos": len(rel),
            "misiones_con_archivos": misiones_dir,
            "mediciones_validas": int(len(med)),
            "pasadas_200km": int(len(pas)),
            "ultimo_dato_local": str(pas["tiempo"].max()) if len(pas) else None,
            "ventanas_h": {k: list(v[:2]) for k, v in VENTANAS.items()},
        }
        json_estricto.escribir(st / "resumen_altimetros.json", resumen)
        motivos = [] if len(pas) else ["ninguna pasada a ≤ 200 km"]
        gate = etapa.Gate(not motivos, {"radio_km": C.RADIO_KM, "archivos": len(rel)}, motivos)
        return etapa.publicar(st, salida, gate, raices, insumos, raiz, None)


def exposicion_por_paso(rejilla: pd.DatetimeIndex, pas: pd.DataFrame) -> pd.DataFrame:
    expo = A.exposicion(rejilla, pas, VENTANAS)
    t = expo.index
    for v in VENTANAS:
        expo[f"asimilada:{v}"] = A.con_pasada_asimilada(expo, v, C.ASIMILADAS, C.NO_VERIFICABLES)
        expo[f"asimilada_sin_nv:{v}"] = A.con_pasada_asimilada(expo, v, C.ASIMILADAS, (), excluir=C.NO_VERIFICABLES)
    ultimo = pas["tiempo"].max() if len(pas) else t[0]
    expo["completa"] = A.completa(t, C.HUECOS_EXPOSICION, C.ASIMILADAS_SIN_DATOS_LOCALES, ultimo)
    return expo


# --- Etapa 2 --------------------------------------------------------------------------------------


class ErrorOrden(RuntimeError):
    """Se pidió un estadístico observado antes de escribir ``potencia.csv`` (G4)."""


class OrdenEscritura:
    def __init__(self, st: Path):
        self.st = Path(st)
        self.escritos: list[str] = []

    def registrar(self, nombre: str) -> None:
        self.escritos.append(nombre)

    def exigir_potencia(self) -> None:
        if not (self.st / "potencia.csv").is_file() or "potencia.csv" not in self.escritos:
            raise ErrorOrden("Estadístico observado pedido antes de escribir potencia.csv (G4)")


def _fin(b: str) -> pd.Timestamp:
    return pd.Timestamp(b) + pd.Timedelta(days=1)


def _rango(tramos_lado: tuple[str, ...]) -> tuple[str, str]:
    t = {n: (a, b) for n, a, b, _ in C.TRAMOS}
    return t[tramos_lado[0]][0], t[tramos_lado[-1]][1]


def _en(t: pd.DatetimeIndex, a: str, b: str) -> np.ndarray:
    return np.asarray((t >= pd.Timestamp(a)) & (t < _fin(b)))


def _rejilla_efectos(delta_rel: float | None, multiplos) -> list[float]:
    if delta_rel is None:
        return [0.01, 0.02, 0.05, 0.10]
    return [round(m * abs(delta_rel), 6) * (1 if delta_rel > 0 else -1) for m in multiplos]


def _mde(efectos, potencias, minimo) -> float | None:
    ok = [abs(e) for e, p in zip(efectos, potencias, strict=True) if p >= minimo]
    return min(ok) if ok else None


class Pruebas:
    """Datos de la etapa 2 y la parte previa a la potencia (no mira ningún estadístico observado)."""

    def __init__(self, serie, serie_sd, vecinas, expo, parametros):
        self.serie, self.serie_sd, self.vecinas, self.expo = serie, serie_sd, vecinas, expo
        self.par = parametros
        self.hs = serie["VHM0"]
        self.t = self.hs.index
        self.contrastes = {}
        self.lados_t = {}
        for nombre, c in C.CONTRASTES.items():
            l1, l2 = _rango(c["lado1"]), _rango(c["lado2"])
            self.lados_t[nombre] = (l1, l2)
            self.contrastes[nombre] = familias.preparar_contraste(nombre, self.hs, l1, l2, c["mes_inicio"])
        self.modelo = relevante.modelo_provisional(self.hs, C.CUANTIL_UMBRAL_PROVISIONAL, C.IET_H, C.PASO_H)
        self.deltas = relevante.rejilla(*C.REJILLA_DELTA)

    # δ_rel de A y pendiente relevante de B (§3.5)
    def delta_rel(self, nombre: str) -> relevante.DeltaRel:
        (a1, b1), (a2, b2) = self.lados_t[nombre]
        tp = self.modelo.t_picos
        lados = {"lado1": _en(tp, a1, b1), "lado2": _en(tp, a2, b2)}
        return relevante.delta_relevante(self.modelo, lados, C.PERIODOS_RETORNO, C.CAMBIO_RL, self.deltas)

    def pendiente_relevante(self) -> dict:
        a, b = _rango(("T1",))
        tp = self.modelo.t_picos
        m = _en(tp, a, b)
        frac = np.clip((tp - pd.Timestamp(a)) / (_fin(b) - pd.Timestamp(a)), 0, 1).to_numpy(float)

        def rl(d: float) -> np.ndarray:
            return relevante.niveles(self.modelo, np.where(m, 1.0 + d * frac, 1.0), C.PERIODOS_RETORNO)

        r = relevante.raiz_en_rejilla(rl, self.deltas, C.CAMBIO_RL)
        n = C.ANIOS_T1[1] - C.ANIOS_T1[0] + 1
        if r is None:
            return {"delta_acumulado": None, "pendiente_relativa_por_anio": None, "anios": n}
        return {
            "delta_acumulado": r[0],
            "pendiente_relativa_por_anio": r[0] / n,
            "periodo": C.PERIODOS_RETORNO[r[1]],
            "cambio": r[2],
            "anios": n,
        }

    # Familia D
    def eventos_tp(self) -> tuple[pd.DataFrame, familias.EventosTp]:
        u = float(np.nanquantile(self.hs.to_numpy(float), C.CUANTIL_UMBRAL_PROVISIONAL))
        df = self.serie[["VHM0", "VTPK"]].rename(columns={"VHM0": "Hs"})
        picos = decluster.decluster_rachas(df, u, C.IET_H, C.PASO_H, columna="Hs")
        l1, l2 = self.lados_t["C2"]
        return picos, familias.preparar_tp(picos, "VTPK", l1, l2, C.CONTRASTES["C2"]["mes_inicio"])

    # Altímetros: estratos y umbrales (§3.4)
    def preparar_altimetros(self) -> dict:
        hs = self.serie_sd["VHM0"]
        t = hs.index
        h = hs.to_numpy(float)
        prev = np.concatenate([[np.nan], h[:-1]])
        t0 = C.TRAMOS[0]
        c_t0 = A.umbral_c_T0(hs, t0[1], t0[2], C.CUANTIL_C_T0)
        niveles = tuple(float(np.nanquantile(h, q)) for q in C.NIVELES_HS)
        s = D.escala_s(h)
        hora = (t.hour.to_numpy() // 3).astype(int)
        principal = np.zeros(len(t), bool)
        for a, b in C.VENTANA_PRINCIPAL:
            principal |= _en(t, a, b)
        principal &= self.expo["completa"].to_numpy(bool)
        sw1 = self.serie_sd["VHM0_SW1"]
        return {
            "t": t,
            "hs": h,
            "nivel": A.nivel_hs(prev, niveles),
            "hora": hora,
            "c_T0": c_t0,
            "niveles_m": niveles,
            "s_m": s,
            "principal": principal,
            "epoca_principal": A.clase_por_intervalos(t, C.EPOCAS_PRINCIPAL),
            "epoca_despues": A.clase_por_intervalos(t, C.EPOCAS_DESPUES),
            "t0": _en(t, t0[1], t0[2]),
            "sw1": sw1.to_numpy(float),
            "c_T0_sw1": A.umbral_c_T0(sw1, t0[1], t0[2], C.CUANTIL_C_T0),
            "niveles_sw1": tuple(float(np.nanquantile(sw1.to_numpy(float), q)) for q in C.NIVELES_HS),
        }

    @staticmethod
    def evento_y_validez(x: np.ndarray, salto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ini, eleg = A.inicios_de_evento(salto)
        prev = np.concatenate([[np.nan], x[:-1]])
        ok = eleg & np.isfinite(x) & np.isfinite(prev)
        return ini, ok

    def tasa_base_t0(self, alt: dict) -> np.ndarray:
        """Tasa de inicios de evento por paso en T0, por nivel de Hs (umbral relativo c_T0). T0 no asimila nada."""
        ev, ok = self.evento_y_validez(alt["hs"], A.saltos_relativos(alt["hs"], alt["c_T0"]))
        tasas = np.zeros(3)
        for k in range(3):
            m = ok & alt["t0"] & (alt["nivel"] == k)
            tasas[k] = ev[m].mean() if m.any() else np.nan
        return tasas

    def placebos(self, alt: dict) -> list[dict]:
        """Exposición y pasos de cada placebo de la combinación principal (§3.4 punto 8 y 10)."""
        expo = self.expo
        t = alt["t"]
        sin_asim = ~expo["asimilada:c"].to_numpy(bool)
        out = []
        for nombre, mision, a, b in C.PLACEBOS:
            col = f"c:{mision}"
            x = expo[col].to_numpy(bool) if col in expo else np.zeros(len(t), bool)
            pasos = alt["principal"] & _en(t, a, b) & sin_asim
            out.append({"nombre": nombre, "expuesto": x, "pasos": pasos, "epoca": alt["epoca_principal"]})
        nombre, a, b, anios = C.PLACEBO_T0
        x = A.calendario_desplazado(expo["asimilada:c"], a, b, anios, t)
        out.append({"nombre": nombre, "expuesto": x, "pasos": alt["t0"].copy(), "epoca": np.zeros(len(t), int)})
        return out

    def estrato(self, alt: dict, epoca: np.ndarray, pasos: np.ndarray, ok: np.ndarray, nivel=None) -> np.ndarray:
        return A.codigos_estrato(alt["hora"], epoca, alt["nivel"] if nivel is None else nivel, validos=pasos & ok)


def homogeneidad(
    raiz_datos=None,
    nombre: str = NOMBRE,
    procesos_n: int | None = None,
    B: int = C.B,
    replicas: int = C.REPLICAS,
    multiplos=C.MULTIPLOS_EFECTO,
    B_ic: int = C.B_IC,
) -> Path:
    raiz = rutas.directorio_datos(raiz_datos)
    procesos_n = procesos_n or os.cpu_count() or 1
    dir_s, dir_c, dir_a = ruta_serie(raiz, nombre), ruta_candidatos(raiz, nombre), ruta_altimetros(raiz, nombre)
    for d in (dir_s, dir_c, dir_a):
        etapa.exigir_publicada(d)
    f1 = raiz / "processed" / "F01-sitio-y-fuentes" / "ficha-sitio" / nombre / "cronologia_waverys.csv"
    insumos = {
        "serie": dir_s / "serie_nodo.parquet",
        "serie_sin_descartar": dir_c / "serie_decodificada.parquet",
        "vecinas": dir_s / "vecinas_hs.parquet",
        "excursiones_k_rev": dir_c / "excursiones_k_rev.csv",
        "candidatos": dir_c / "candidatos.csv",
        "marcas_por_anio": dir_s / "marcas_por_anio.csv",
        "marcas_por_regla": dir_s / "marcas_por_regla.csv",
        "cronologia_f1": f1,
        "exposicion": dir_a / "exposicion_altimetros.parquet",
        "misiones_locales": dir_a / "misiones_locales.csv",
    }
    etapa.exigir_insumos(insumos)
    salida = ruta_homogeneidad(raiz, nombre)
    raices = orquestador.rutas_codigo(ETAPA)
    with etapa.staging_protegido(salida, raices) as st:
        orden = OrdenEscritura(st)
        serie = carga.cargar_serie(raiz, nombre)
        serie_sd = carga.cargar_serie(raiz, nombre, sin_descartar=True)
        expo = pd.read_parquet(insumos["exposicion"]).reindex(serie_sd.index)
        expo = expo.fillna(False).astype(bool)
        locales = pd.read_csv(insumos["misiones_locales"], parse_dates=["inicio", "fin"])
        vecinas = pd.read_parquet(insumos["vecinas"])
        pr = Pruebas(serie, serie_sd, vecinas, expo, {})

        # --- 1. Cronología, tramos y cobertura (no son estadísticos observados) ------------------
        crono = cronologia.cronologia(pd.read_csv(f1))
        _csv(crono, st / "cronologia_f04.csv")
        orden.registrar("cronologia_f04.csv")
        _csv(cronologia.tramos(C.TRAMOS, C.SUBTRAMOS_DESCRIPTIVOS, C.ESTACIONES), st / "tramos.csv")
        orden.registrar("tramos.csv")
        cob = cronologia.cobertura_altimetria(
            locales,
            C.ASIMILADAS,
            C.ASIMILADAS_SIN_DATOS_LOCALES,
            C.NO_ASIMILADAS,
            C.NO_VERIFICABLES,
            C.PLACEBOS + (C.PLACEBO_SENSIBILIDAD,),
            C.HUECOS_EXPOSICION,
        )
        _csv(cob, st / "cobertura_altimetria.csv")
        orden.registrar("cobertura_altimetria.csv")

        # --- 2. Potencia (§3.6), antes de cualquier estadístico observado -------------------------
        filas_pot: list[dict] = []
        previo: dict = {}
        dr = {n: pr.delta_rel(n) for n in C.CONTRASTES}
        previo["delta_rel"] = {n: asdict(d) for n, d in dr.items()}

        # cronometraje y B de las réplicas (§3.6)
        c1 = pr.contrastes["C1"]
        t_ini = time.perf_counter()
        familias._tarea_A((c1, "anios", B, 0.0, 1, False, C.RHO_AR1, ("f04", "cronometro"), 0, 1))
        seg = time.perf_counter() - t_ini
        n_sets = 3 + len(multiplos)
        proyectado_h = seg * replicas * n_sets * len(C.CONTRASTES) / procesos_n / 3600.0
        B_pot = B if proyectado_h <= C.HORAS_MAX_POTENCIA else min(B, C.B_REDUCIDO)
        # el cronometraje varía entre corridas: va a la consola, no a las salidas (G7)
        print(
            f"Cronómetro: {seg:.2f} s por réplica; potencia proyectada {proyectado_h:.2f} h con {procesos_n} "
            f"procesos; B de las réplicas = {B_pot}",
            flush=True,
        )
        previo["B_replicas"] = {
            "B": B_pot,
            "reducido": B_pot < B,
            "regla": "§3.6: si la potencia proyectada pasa de 8 h, B = 999 en las réplicas (no en la prueba observada)",
        }

        reglas_A: dict[str, familias.ReglaDependencia] = {}
        pot_A: dict[str, dict] = {}
        for n, c in pr.contrastes.items():
            d = dr[n]
            lado_iny = 0 if d.lado == "lado1" else 1

            def nula(esquema: str, etiqueta: str, c=c, n=n) -> np.ndarray:
                return familias.simular_A(
                    c, esquema, B_pot, replicas, 0.0, 1, True, C.RHO_AR1, ("f04", "A", n, etiqueta, esquema), procesos_n
                )

            regla = familias.aplicar_regla(nula, C.ALFA, C.TIPO_I_MAX)
            reglas_A[n] = regla
            for esq, err in regla.tipo_I.items():
                filas_pot.append(
                    {
                        "prueba": "A",
                        "contraste": n,
                        "efecto": "tipo I AR(1)",
                        "valor": 0.0,
                        "tasa": err,
                        "replicas": replicas,
                        "B": B_pot,
                        "esquema": esq,
                        "critico": C.ALFA,
                        "dependencia": True,
                    }
                )
            if regla.tipo_I_tras_recalibrar is not None:
                filas_pot.append(
                    {
                        "prueba": "A",
                        "contraste": n,
                        "efecto": "tipo I AR(1) tras recalibrar",
                        "valor": 0.0,
                        "tasa": regla.tipo_I_tras_recalibrar,
                        "replicas": replicas,
                        "B": B_pot,
                        "esquema": regla.esquema,
                        "critico": regla.critico,
                        "dependencia": True,
                    }
                )
            efectos = _rejilla_efectos(d.delta, multiplos)
            pots = []
            for e in efectos:
                p = familias.simular_A(
                    c,
                    regla.esquema,
                    B_pot,
                    replicas,
                    e,
                    lado_iny,
                    regla.dependencia_en_potencia,
                    C.RHO_AR1,
                    ("f04", "A", n, "potencia", f"{e:.6f}"),
                    procesos_n,
                )
                pots.append(float(np.mean(p <= regla.critico)))
                filas_pot.append(
                    {
                        "prueba": "A",
                        "contraste": n,
                        "efecto": "delta",
                        "valor": e,
                        "tasa": pots[-1],
                        "replicas": replicas,
                        "B": B_pot,
                        "esquema": regla.esquema,
                        "critico": regla.critico,
                        "dependencia": regla.dependencia_en_potencia,
                    }
                )
            pot_rel = pots[list(multiplos).index(1.0)] if (d.delta is not None and 1.0 in multiplos) else None
            pot_A[n] = {
                "efectos": efectos,
                "potencias": pots,
                "potencia_delta_rel": pot_rel,
                "mde_80": _mde(efectos, pots, C.POTENCIA_MIN),
            }

        # Familia B: pendiente relevante y potencia con n = 23
        pend = pr.pendiente_relevante()
        previo["pendiente_relevante"] = pend
        pot_B = {"efectos": [], "potencias": [], "potencia_relevante": None, "mde_80": None}
        if pend["pendiente_relativa_por_anio"] is not None:
            hs_t1 = pr.hs[_en(pr.t, f"{C.ANIOS_T1[0]}-01-01", f"{C.ANIOS_T1[1]}-12-31")]
            # valores anuales de T1 reordenados al azar en cada réplica: su orden (el resultado) no se usa
            base = descriptivos.anuales_hs(hs_t1)[["hs_p99", "hs_max"]].to_numpy(float)
            efs = [m * pend["pendiente_relativa_por_anio"] for m in multiplos]
            for e in efs:
                rng = semillas.generador("f04", "B", "potencia", f"{e:.8f}")
                rech = 0
                for _ in range(replicas):
                    y = base[rng.permutation(base.shape[0])]
                    y = y * (1.0 + e * np.arange(y.shape[0]))[:, None]
                    ps = [tendencia.mann_kendall(y[:, j]).p for j in range(2)]
                    rech += P.holm(ps).min() <= C.ALFA
                pot_B["efectos"].append(e)
                pot_B["potencias"].append(rech / replicas)
                filas_pot.append(
                    {
                        "prueba": "B",
                        "contraste": "T1",
                        "efecto": "pendiente relativa por año",
                        "valor": e,
                        "tasa": rech / replicas,
                        "replicas": replicas,
                        "B": 0,
                        "esquema": "Mann-Kendall con Hamed-Rao, Holm",
                        "critico": C.ALFA,
                        "dependencia": False,
                    }
                )
            pot_B["potencia_relevante"] = pot_B["potencias"][list(multiplos).index(1.0)] if 1.0 in multiplos else None
            pot_B["mde_80"] = _mde(pot_B["efectos"], pot_B["potencias"], C.POTENCIA_MIN)

        # Familia D: mismas unidades que A en C2; error de tipo I AR(1) y potencia frente a 0.3 s
        picos, ev = pr.eventos_tp()
        esq_D = reglas_A["C2"].esquema

        def nula_D(etiqueta: str) -> np.ndarray:
            return familias.simular_D(
                ev, esq_D, B_pot, replicas, 0.0, True, C.RHO_AR1, ("f04", "D", etiqueta), procesos_n
            )

        p_nula_D = nula_D("tipo-I")
        tipoI_D = float(np.mean(p_nula_D <= C.ALFA))
        critico_D, dep_D, tipoI_D_v = C.ALFA, esq_D != "anios", None
        if tipoI_D > C.TIPO_I_MAX:
            critico_D = P.valor_critico_recalibrado(p_nula_D, C.ALFA)
            tipoI_D_v = float(np.mean(nula_D("verificacion") <= critico_D))
            dep_D = True
        filas_pot.append(
            {
                "prueba": "D",
                "contraste": "C2",
                "efecto": "tipo I AR(1)",
                "valor": 0.0,
                "tasa": tipoI_D,
                "replicas": replicas,
                "B": B_pot,
                "esquema": esq_D,
                "critico": C.ALFA,
                "dependencia": True,
            }
        )
        pot_D = {"efectos": [], "potencias": []}
        for m in multiplos:
            e = m * C.EFECTO_TP_S
            p = familias.simular_D(
                ev, esq_D, B_pot, replicas, e, dep_D, C.RHO_AR1, ("f04", "D", "potencia", f"{e}"), procesos_n
            )
            pot_D["efectos"].append(e)
            pot_D["potencias"].append(float(np.mean(p <= critico_D)))
            filas_pot.append(
                {
                    "prueba": "D",
                    "contraste": "C2",
                    "efecto": "desplazamiento de Tp (s)",
                    "valor": e,
                    "tasa": pot_D["potencias"][-1],
                    "replicas": replicas,
                    "B": B_pot,
                    "esquema": esq_D,
                    "critico": critico_D,
                    "dependencia": dep_D,
                }
            )
        pot_D["potencia_relevante"] = pot_D["potencias"][list(multiplos).index(1.0)] if 1.0 in multiplos else None
        pot_D["mde_80"] = _mde(pot_D["efectos"], pot_D["potencias"], C.POTENCIA_MIN)

        # Altímetros: potencia de la combinación principal y de los placebos (clases y m)
        alt = pr.preparar_altimetros()
        tasas = pr.tasa_base_t0(alt)
        tasa_paso = tasas[np.clip(alt["nivel"], 0, 2)]
        ev_p, ok_p = pr.evento_y_validez(alt["hs"], A.saltos_relativos(alt["hs"], alt["c_T0"]))
        est_p = pr.estrato(alt, alt["epoca_principal"], alt["principal"], ok_p)
        x_p = expo["asimilada:c"].to_numpy(bool)
        pot_alt = A.potencia_mh(
            x_p,
            est_p,
            tasa_paso,
            C.COCIENTE_ALTERNATIVO,
            C.FRACCION_OCULTA,
            1 - C.ALFA,
            replicas,
            semillas.generador("f04", "altimetros", "principal"),
        )
        filas_pot.append(
            {
                "prueba": "altimetros",
                "contraste": "principal",
                "efecto": "cociente",
                "valor": 1.5,
                "tasa": pot_alt,
                "replicas": replicas,
                "B": 0,
                "esquema": "Mantel-Haenszel",
                "critico": C.ALFA,
                "dependencia": False,
            }
        )
        placebos = pr.placebos(alt)
        for pl in placebos:
            pl["n_pasos"] = int((pl["pasos"] & ok_p).sum())
            pl["n_con_pasada"] = int((pl["pasos"] & ok_p & pl["expuesto"]).sum())
        con_pasos = [pl["nombre"] for pl in placebos if 0 < pl["n_con_pasada"] < pl["n_pasos"]]
        por_nombre = {pl["nombre"]: pl for pl in placebos}

        def potencia_placebo(nombre_pl: str, m: int) -> float:
            pl = por_nombre[nombre_pl]
            est = pr.estrato(alt, pl["epoca"], pl["pasos"], ok_p)
            return A.potencia_mh(
                pl["expuesto"],
                est,
                tasa_paso,
                C.COCIENTE_ALTERNATIVO,
                0.0,
                1 - C.ALFA / m,
                replicas,
                semillas.generador("f04", "altimetros", nombre_pl, str(m)),
            )

        clases = A.clasificar_placebos(
            [pl["nombre"] for pl in placebos], con_pasos, potencia_placebo, C.ALFA, C.POTENCIA_MIN
        )
        cuentan = clases.cuentan
        for pl in placebos:
            pl["clase"] = clases.clase[pl["nombre"]]
            pl["potencia"] = clases.potencia.get(pl["nombre"])
            pl["nivel"] = 1 - C.ALFA / max(clases.m, 1)
            filas_pot.append(
                {
                    "prueba": "altimetros",
                    "contraste": f"placebo: {pl['nombre']}",
                    "efecto": "cociente",
                    "valor": 1.5,
                    "tasa": np.nan if pl["potencia"] is None else pl["potencia"],
                    "replicas": replicas,
                    "B": 0,
                    "esquema": "Mantel-Haenszel",
                    "critico": 1 - pl["nivel"],
                    "dependencia": False,
                }
            )
        pot = pd.DataFrame(filas_pot)
        _csv(pot, st / "potencia.csv")
        orden.registrar("potencia.csv")

        # --- 3. Estadísticos observados -----------------------------------------------------------
        orden.exigir_potencia()
        res_A = {}
        for n, c in pr.contrastes.items():
            regla = reglas_A[n]
            r = familias.prueba_A(c, regla.esquema, B, "f04", "A", n, "observada")
            rechaza = r.p_min <= regla.critico
            d = dr[n]
            cat = familias.resultado_adr0005(rechaza, pot_A[n]["potencia_delta_rel"], C.POTENCIA_MIN, d.sin_raiz)
            res_A[n] = {
                "estadisticos": list(P.NOMBRES_A),
                "log_cociente": r.T.tolist(),
                "cociente": np.exp(r.T).tolist(),
                "z": r.z.tolist(),
                "p_crudo": r.p_crudo.tolist(),
                "p_ajustado_maxT": r.p_ajustado.tolist(),
                "B": B,
                "esquema": regla.esquema,
                "valor_critico": regla.critico,
                "tipo_I_AR1": regla.tipo_I,
                "tipo_I_tras_recalibrar": regla.tipo_I_tras_recalibrar,
                "recalibrable": regla.recalibrable,
                "potencia_dependiente": regla.dependencia_en_potencia,
                "rechaza": bool(rechaza),
                "delta_rel": asdict(d),
                "potencia": pot_A[n],
                "resultado": cat,
                "cambia_periodo": cat in (C.ESTABLECIDO, C.NO_CONCLUYENTE) and not d.sin_raiz,
            }

        # Cuerpo (equivalencia de P50) por contraste
        rng_ic = semillas.generador("f04", "ic")
        for n in C.CONTRASTES:
            (a1, b1), (a2, b2) = pr.lados_t[n]
            q, lo, hi = descriptivos.cociente_p50(pr.hs[_en(pr.t, a1, b1)], pr.hs[_en(pr.t, a2, b2)], B_ic, rng_ic)
            eq = bool(lo >= 1 - C.EQUIVALENCIA_P50 and hi <= 1 + C.EQUIVALENCIA_P50)
            res_A[n]["cuerpo_p50"] = {
                "cociente": q,
                "ic": [lo, hi],
                "equivalente_2pct": eq,
                "cuerpo_estable_y_cola_cambia": bool(eq and res_A[n]["resultado"] == C.ESTABLECIDO),
            }

        # Familia B
        orden.exigir_potencia()
        an_all = descriptivos.anuales_hs(pr.hs)
        res_B = {}
        for etiqueta, desde in (("principal", C.ANIOS_T1[0]), ("sin_1993_1997", C.SIN_PRIMEROS[1] + 1)):
            g = an_all.loc[desde : C.ANIOS_T1[1]]
            r99, rmx = tendencia.mann_kendall(g["hs_p99"].to_numpy()), tendencia.mann_kendall(g["hs_max"].to_numpy())
            ph = P.holm([r99.p, rmx.p])
            res_B[etiqueta] = {
                "anios": [int(g.index.min()), int(g.index.max())],
                "P99": {
                    **asdict(r99),
                    "pendiente_relativa_por_anio": r99.pendiente / g["hs_p99"].mean(),
                    "p_holm": float(ph[0]),
                },
                "maximo_anual": {
                    **asdict(rmx),
                    "pendiente_relativa_por_anio": rmx.pendiente / g["hs_max"].mean(),
                    "p_holm": float(ph[1]),
                },
                "rechaza": bool(ph.min() <= C.ALFA),
            }
        sin_raiz_B = pend["pendiente_relativa_por_anio"] is None
        res_B["pendiente_relevante"] = pend
        res_B["potencia"] = pot_B
        res_B["resultado"] = familias.resultado_adr0005(
            res_B["principal"]["rechaza"], pot_B["potencia_relevante"], C.POTENCIA_MIN, sin_raiz_B
        )
        res_B["advertencia"] = (
            "Mann-Kendall con Hamed-Rao no mantiene el nivel con n = 23 bajo AR(1) "
            "(prueba sintética de §5); ver metodología."
        )

        # Familia C
        completos = an_all[an_all["meses"] == 12]
        anios_crono = sorted({pd.Timestamp(f).year for f in crono["fecha"] if f > "1980-01-01"})
        res_C = {}
        for col in ("hs_p50", "hs_p99", "hs_max"):
            y = completos[col].to_numpy(float)
            for metodo, fn in (("pettitt", tendencia.pettitt), ("snht", tendencia.snht)):
                pc = fn(y)
                anio = int(completos.index[pc.posicion])
                co, azar = descriptivos.coincidencias([anio], anios_crono, completos.index, C.VENTANA_COINCIDE_ANIOS)
                res_C[f"{col}:{metodo}"] = {
                    "primer_anio_despues": anio,
                    "estadistico": pc.estadistico,
                    "coincide": co[0],
                    "probabilidad_coincidir_por_azar": azar,
                    "a_la_F8": not co[0],
                }

        # Familia D
        r_D = familias.prueba_D(
            ev, ev.tp, np.arange(ev.anios.size), esq_D, B, semillas.generador("f04", "D", "observada")
        )
        res_D = {
            "diferencia_mediana_tp_s": r_D.diferencia,
            "p": r_D.p,
            "B": B,
            "esquema": esq_D,
            "valor_critico": critico_D,
            "tipo_I_AR1": tipoI_D,
            "tipo_I_tras_recalibrar": tipoI_D_v,
            "eventos_por_lado": list(r_D.n_eventos),
            "estados_por_lado": [
                int(
                    (
                        (pr.hs >= np.nanquantile(pr.hs, C.CUANTIL_UMBRAL_PROVISIONAL)) & _en(pr.t, *pr.lados_t["C2"][k])
                    ).sum()
                )
                for k in (0, 1)
            ],
            "potencia": pot_D,
        }
        res_D["resultado"] = familias.resultado_adr0005(r_D.p <= critico_D, pot_D["potencia_relevante"], C.POTENCIA_MIN)

        # Altímetros
        def mh(evento, ok, pasos, expuesto, epoca, nivel=None, nivel_conf=0.95):
            est = pr.estrato(alt, epoca, pasos, ok, nivel)
            return A.mantel_haenszel(evento, expuesto, est, nivel_conf)

        principal = mh(ev_p, ok_p, alt["principal"], x_p, alt["epoca_principal"])
        lect = C.SENAL if principal.excluye_1() else C.DESCARTADO if pot_alt >= C.POTENCIA_MIN else C.NO_CONCLUYENTE
        m_cuentan = len(cuentan)
        res_pl = []
        invalida = False
        for pl in placebos:
            r = mh(ev_p, ok_p, pl["pasos"], pl["expuesto"], pl["epoca"])
            nivel = 1 - C.ALFA / max(m_cuentan, 1)
            r_aj = mh(ev_p, ok_p, pl["pasos"], pl["expuesto"], pl["epoca"], nivel_conf=nivel)
            fila = {
                "nombre": pl["nombre"],
                "clase": pl["clase"],
                "pasos": pl["n_pasos"],
                "pasos_con_pasada": pl["n_con_pasada"],
                "potencia": pl.get("potencia"),
                "mh": asdict(r),
                "ic_ajustado": [r_aj.lo, r_aj.hi],
                "nivel_ajustado": nivel,
            }
            if np.isfinite(principal.lo) and np.isfinite(r.lo) and r.cociente > 0:
                se = lambda x: (np.log(x.hi) - np.log(x.lo)) / (2 * 1.959964)  # noqa: E731
                lr = np.log(principal.cociente) - np.log(r.cociente)
                s_ = np.hypot(se(principal), se(r))
                fila["cociente_de_cocientes"] = [
                    float(np.exp(lr)),
                    float(np.exp(lr - 1.959964 * s_)),
                    float(np.exp(lr + 1.959964 * s_)),
                ]
            if pl["clase"] == "cuenta" and r_aj.excluye_1():
                invalida = True
            res_pl.append(fila)
        control = "sin control de placebo" if m_cuentan == 0 else ("inválida" if invalida else "válida")

        sens = {}
        x_r = expo["asimilada:r"].to_numpy(bool)
        sens["ventana_hacia_atras"] = asdict(mh(ev_p, ok_p, alt["principal"], x_r, alt["epoca_principal"]))
        for k in C.K_A:
            e_k, o_k = pr.evento_y_validez(alt["hs"], A.saltos_absolutos(alt["hs"], k * alt["s_m"]))
            sens[f"k_a={k}"] = asdict(mh(e_k, o_k, alt["principal"], x_p, alt["epoca_principal"]))
        e_10, o_10 = pr.evento_y_validez(alt["hs"], A.saltos_relativos(alt["hs"], C.UMBRAL_RELATIVO_POSTERIOR))
        sens["relativo_0.10_posterior_a_ver_datos"] = asdict(
            mh(e_10, o_10, alt["principal"], x_p, alt["epoca_principal"])
        )
        ultimo = locales["fin"].max()
        despues = _en(alt["t"], C.INICIO_SAR, str(pd.Timestamp(ultimo).date()))
        sens["despues_de_2016_03"] = asdict(mh(ev_p, ok_p, despues, x_p, alt["epoca_despues"]))
        sens["despues_de_2016_03_sin_no_verificables"] = asdict(
            mh(ev_p, ok_p, despues, expo["asimilada_sin_nv:c"].to_numpy(bool), alt["epoca_despues"])
        )
        niv_sw1 = A.nivel_hs(np.concatenate([[np.nan], alt["sw1"][:-1]]), alt["niveles_sw1"])
        e_s, o_s = pr.evento_y_validez(alt["sw1"], A.saltos_relativos(alt["sw1"], alt["c_T0_sw1"]))
        sens["swell_1_umbral_c_T0_propio"] = asdict(
            mh(e_s, o_s, alt["principal"], x_p, alt["epoca_principal"], nivel=niv_sw1)
        )
        nombre_s, mision_s, a_s, b_s = C.PLACEBO_SENSIBILIDAD
        col_s = f"c:{mision_s}"
        if col_s in expo:
            pasos_s = despues & _en(alt["t"], a_s, b_s) & ~x_p
            sens[f"placebo {nombre_s}"] = asdict(
                mh(ev_p, ok_p, pasos_s, expo[col_s].to_numpy(bool), alt["epoca_despues"])
            )
        res_alt = {
            "umbral_c_T0": alt["c_T0"],
            "umbral_c_T0_swell_1": alt["c_T0_sw1"],
            "niveles_hs_m": list(alt["niveles_m"]),
            "s_m": alt["s_m"],
            "tasa_base_T0_por_nivel": tasas.tolist(),
            "principal": asdict(principal),
            "potencia_principal": pot_alt,
            "lectura": lect,
            "placebos": res_pl,
            "placebos_que_cuentan": m_cuentan,
            "control_de_placebo": control,
            "resultado": "inválida" if invalida else lect,
            "sensibilidades": sens,
            "nota_sensibilidades": "Descriptivas. Después de 2016-03 la exposición es incompleta: solo una «señal» "
            "es interpretable.",
        }

        # Descriptivos: cuantiles por tramo, indicadores, figuras
        grupos = {n: (a, b) for n, a, b, _ in C.TRAMOS + C.SUBTRAMOS_DESCRIPTIVOS}
        for n in C.CONTRASTES:
            (a1, b1), (a2, b2) = pr.lados_t[n]
            grupos[f"{n}:lado1"], grupos[f"{n}:lado2"] = (a1, b1), (a2, b2)
        variables = {v: serie[v] for v in C.VARIABLES}
        variables["Hs_vecinas_mediana"] = descriptivos.mediana_vecinas(vecinas, C.NODO_VECINAS)
        estaciones = {
            "todo": (None, 1),
            "invierno": (C.ESTACIONES["invierno"], 5),
            "verano": (C.ESTACIONES["verano"], 11),
        }
        filas = []
        for v, s in variables.items():
            for g, (a, b) in grupos.items():
                sg = s[_en(s.index, a, b)]
                for est, (meses, mi) in estaciones.items():
                    r = descriptivos.cuantiles_de_grupo(sg, mi, meses, C.CUANTILES, B_ic, rng_ic)
                    for k, (val, lo, hi, na) in r.items():
                        filas.append(
                            {
                                "variable": v,
                                "grupo": g,
                                "estacion": est,
                                "estadistico": k,
                                "valor": val,
                                "ic_lo": lo,
                                "ic_hi": hi,
                                "anios_completos": na,
                            }
                        )
        u99 = float(np.nanquantile(pr.hs, C.CUANTIL_TP_HS_ALTO))
        for g, (a, b) in grupos.items():
            r = descriptivos.tp_hs_alto(serie[_en(serie.index, a, b)], u99, B_ic, rng_ic)
            for k, (val, lo, hi, na) in r.items():
                filas.append(
                    {
                        "variable": "VTPK|Hs>=P99",
                        "grupo": g,
                        "estacion": "todo",
                        "estadistico": k,
                        "valor": val,
                        "ic_lo": lo,
                        "ic_hi": hi,
                        "anios_completos": na,
                    }
                )
        cpt = pd.DataFrame(filas)
        _csv(cpt, st / "cuantiles_por_tramo.csv")
        orden.registrar("cuantiles_por_tramo.csv")
        ind = descriptivos.indicadores_anuales(
            serie,
            pd.read_csv(insumos["excursiones_k_rev"]),
            pd.read_csv(insumos["candidatos"]),
            pd.read_csv(insumos["marcas_por_anio"], index_col="anio"),
        )
        _csv(ind, st / "indicadores_anuales.csv", index=True)
        orden.registrar("indicadores_anuales.csv")
        fechas = sorted({pd.Timestamp(f) for f, tr in zip(crono["fecha"], crono["tramo"], strict=True) if tr})
        figuras.cuantiles_por_tramo(cpt, [n for n, *_ in C.TRAMOS], st / "figuras" / "cuantiles_por_tramo.png")
        figuras.indicadores(ind, fechas, st / "figuras" / "indicadores_anuales.png")
        figuras.cuantiles_anuales(an_all, fechas, st / "figuras" / "cuantiles_anuales.png")
        figuras.exposicion(expo, st / "figuras" / "exposicion_altimetros.png")
        orden.registrar("figuras")

        # --- 4. Decisión (§3.7) --------------------------------------------------------------------
        resumen = decision(res_A, res_B, res_C, res_D, res_alt)
        json_estricto.escribir(
            st / "pruebas_cambio.json", {"A": res_A, "B": res_B, "C": res_C, "D": res_D, "previo": previo}
        )
        orden.registrar("pruebas_cambio.json")
        json_estricto.escribir(st / "prueba_altimetros.json", res_alt)
        orden.registrar("prueba_altimetros.json")
        json_estricto.escribir(st / "resumen_f04.json", resumen)
        orden.registrar("resumen_f04.json")

        motivos = gate_f04(crono, cob, res_A, res_B, res_D, res_alt, resumen)
        gate = etapa.Gate(
            not motivos,
            {"alfa": C.ALFA, "potencia_min": C.POTENCIA_MIN, "B": B, "replicas": replicas, "B_replicas": B_pot},
            motivos,
        )
        return etapa.publicar(
            st,
            salida,
            gate,
            raices,
            insumos,
            raiz,
            semillas.semilla("f04"),
            extra={"orden_de_escritura": orden.escritos},
        )


def decision(res_A, res_B, res_C, res_D, res_alt) -> dict:
    """Escenarios y lo que heredan la F5, la F8 y la F9 (§3.7)."""
    a1, a2 = res_A["C1"], res_A["C2"]
    inicio = C.TRAMOS[0][1]
    fin = C.TRAMOS[-1][2]
    escenarios = [{"nombre": "E0", "definicion": "serie completa", "periodo": [inicio, fin]}]
    e1 = e2 = None
    if a1["cambia_periodo"]:
        cociente = a1["cociente"][P.NOMBRES_A.index("media_maximos")]  # asimilado / T0
        if cociente > 1:
            e1 = {
                "nombre": "E1",
                "forma": "recorte",
                "definicion": "serie de 1993-02 a 2026-08",
                "periodo": [C.TRAMOS[1][1], fin],
                "cociente_media_maximos_asimilado_T0": cociente,
            }
        else:
            e1 = {
                "nombre": "E1",
                "forma": "escalon_al_nivel_T0",
                "definicion": "serie completa con el escalón de C1 como covariable, al nivel de T0",
                "periodo": [inicio, fin],
                "fecha_escalon": C.TRAMOS[1][1],
                "cociente_media_maximos_asimilado_T0": cociente,
            }
        escenarios.append(e1)
    if a2["cambia_periodo"]:
        cociente = a2["cociente"][P.NOMBRES_A.index("media_maximos")]
        nivel = "T2-T4" if cociente > 1 else "T1"
        e2 = {
            "nombre": "E2",
            "definicion": f"serie completa con el escalón de C2 como covariable, al nivel de {nivel}",
            "periodo": [inicio, fin],
            "fecha_escalon": C.INICIO_SAR,
            "nivel": nivel,
        }
        escenarios.append(e2)
    variantes_tp = res_D["resultado"] in (C.ESTABLECIDO, C.NO_CONCLUYENTE)
    contornos = []
    for e in escenarios:
        if variantes_tp and e["nombre"] in ("E0", "E2"):
            contornos += [f"{e['nombre']}:Tp sin escalón", f"{e['nombre']}:Tp con escalón de C2"]
        else:
            contornos.append(e["nombre"])
    f8 = []
    if a1["cambia_periodo"]:
        f8.append({"fecha": C.TRAMOS[1][1], "origen": "C1", "resultado": a1["resultado"]})
    if a2["cambia_periodo"]:
        f8.append({"fecha": C.INICIO_SAR, "origen": "C2", "resultado": a2["resultado"]})
    if res_B["resultado"] in (C.ESTABLECIDO, C.NO_CONCLUYENTE):
        f8.append(
            {
                "deriva_en_T1": res_B["resultado"],
                "lectura": "probada" if res_B["resultado"] == C.ESTABLECIDO else "hipótesis sin probar",
            }
        )
    hip = [
        {"serie": k, "primer_anio_despues": v["primer_anio_despues"], "rotulo": "exploratoria, sin p-valor"}
        for k, v in res_C.items()
        if v["a_la_F8"]
    ]
    return {
        "periodo_base": [inicio, fin],
        "escenarios_hs": escenarios,
        "n_escenarios_hs": len(escenarios),
        "variantes_tp": variantes_tp,
        "contornos": contornos,
        "n_contornos": len(contornos),
        "maximo_contornos_contrato": 5,
        "regla_de_diseno": "mismo rango; el diseño comprueba el más desfavorable de los activos hasta que la F5 "
        "atribuya (ADR-0005; ADR-0004 punto 5). Las variantes de Tp las comprueba el cálculo "
        "estructural.",
        "resultados": {
            "A_C1": a1["resultado"],
            "A_C2": a2["resultado"],
            "B": res_B["resultado"],
            "D": res_D["resultado"],
            "altimetros": res_alt["resultado"],
        },
        "cuerpo_estable_y_cola_cambia": {n: res_A[n]["cuerpo_p50"]["cuerpo_estable_y_cola_cambia"] for n in res_A},
        "a_la_F8": {"fechas_de_cambio": f8, "hipotesis_rotuladas": hip},
        "a_la_F5": {
            "C1": {
                "pregunta": "¿El producto se aleja de las observaciones de forma distinta en T0 y en el tramo "
                "asimilado, en la cola?",
                "observaciones": {"T0": "GEOSAT (1985-03-31 a 1989-08-29)", "asimilado": ["HY-2", "HY-2B"]},
                "activa": a1["cambia_periodo"],
            },
            "C2": {
                "pregunta": "La misma, antes y después de 2016-03.",
                "observaciones": {"principal": "HY-2 (HY-2A), T1 frente a T2, 2011-10 a 2018-06", "T3-T4": "HY-2B"},
                "activa": a2["cambia_periodo"],
            },
            "otras_misiones": "Sentinel-3B, Sentinel-6A, SWOT y el nadir de CFOSAT solo si se verifica su "
            "independencia.",
            "metodo": "lo fija el contrato de la F5",
            "altimetros": "validar los registros con salto coincidente con pasada"
            if res_alt["resultado"] == C.SENAL
            else None,
        },
        "a_la_F9": {"sensibilidad_sin_saltos_coincidentes": res_alt["resultado"] == C.SENAL},
        "G8": "pendiente de la aceptación del usuario",
    }


def gate_f04(crono, cob, res_A, res_B, res_D, res_alt, resumen) -> list[str]:
    motivos = []
    malos = crono[~crono["estado"].isin(["verificada", "inferida", "no disponible"]) | crono["referencia"].isna()]
    if len(malos):
        motivos.append(f"G3: {len(malos)} filas de la cronología sin referencia ni estado")
    for n, r in res_A.items():
        if r["resultado"] not in (C.ESTABLECIDO, C.DESCARTADO, C.NO_CONCLUYENTE, C.SIN_RAIZ):
            motivos.append(f"G5: familia A {n} sin resultado")
        sin_pot = r["potencia"]["potencia_delta_rel"] is None
        if not r["rechaza"] and sin_pot and r["delta_rel"]["delta"] is not None:
            motivos.append(f"G5: familia A {n} no rechaza sin potencia al lado")
    for nombre, r in (("B", res_B), ("D", res_D)):
        if r["resultado"] not in (C.ESTABLECIDO, C.DESCARTADO, C.NO_CONCLUYENTE, C.SIN_RAIZ):
            motivos.append(f"G5: familia {nombre} sin resultado")
    if any(p["clase"] not in ("cuenta", "no informativo", "sin pasos") for p in res_alt["placebos"]):
        motivos.append("G6: placebo sin clase")
    if resumen["n_contornos"] > resumen["maximo_contornos_contrato"]:
        motivos.append("G5: más contornos que el máximo del contrato")
    return motivos


# --- CLI ----------------------------------------------------------------------------------------


def main_altimetros(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="homogeneidad-altimetros")
    ap.add_argument("--datos")
    ap.add_argument("--nombre", default=NOMBRE)
    a = ap.parse_args(argv)
    try:
        print(f"Publicado en {altimetros(a.datos, a.nombre)}")
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="homogeneidad")
    ap.add_argument("--datos")
    ap.add_argument("--nombre", default=NOMBRE)
    ap.add_argument("--procesos", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        print(f"Publicado en {homogeneidad(a.datos, a.nombre, a.procesos)}")
    except GateFallido as exc:
        print(f"GATE FALLIDO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
