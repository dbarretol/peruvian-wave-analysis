"""Avance guardado y registro de progreso de las simulaciones largas de la F4 (potencia y error de tipo I).

**Qué se guarda.** Cada lote de réplicas terminado se escribe en ``interim/F04-homogeneidad/avance/<clave>/`` como
un ``.npy`` con los p de sus réplicas. Al relanzar la etapa, los lotes ya guardados se leen y no se vuelven a
calcular. Solo se guardan **simulaciones** (réplicas con datos reasignados al azar), nunca un estadístico
observado: el orden de G4 no cambia.

**Cuándo se reutiliza.** La clave del directorio combina la huella del código de la etapa y los hashes de sus
insumos; la de cada lote, todos los parámetros de la tarea (esquema, B, efecto, lado, dependencia, ρ, nombres de la
semilla y réplicas). Si cambia el código, un insumo o un parámetro, la clave cambia y no se reutiliza nada. Como
cada réplica deriva su semilla de nombres, un lote guardado es idéntico al que se calcularía: el resultado no
depende de si la corrida se interrumpió.

**Escritura atómica:** se escribe a un temporal y se renombra; un corte a mitad de escritura no deja un lote roto.

**Progreso:** tras cada tanda se imprime la simulación, las réplicas hechas y el tiempo restante estimado con la
velocidad medida en esa simulación (en paralelo, no con una réplica sola).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ..comun import procesos


def _clave(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Avance:
    def __init__(self, directorio: Path | None, clave_base: str, registro: Callable[[str], None] = print):
        self.dir = None if directorio is None else Path(directorio) / clave_base[:24]
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.registro = registro
        self.inicio = time.perf_counter()
        self.reutilizados = 0
        self.calculados = 0

    def _ruta(self, firma) -> Path | None:
        return None if self.dir is None else self.dir / f"{_clave(firma)}.npy"

    def _leer(self, firma) -> list[float] | None:
        r = self._ruta(firma)
        if r is None or not r.is_file():
            return None
        return np.load(r).tolist()

    def _guardar(self, firma, valores: list[float]) -> None:
        r = self._ruta(firma)
        if r is None:
            return
        tmp = r.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, np.asarray(valores, float))
        os.replace(tmp, r)

    def ejecutar(
        self,
        funcion: Callable,
        tareas: Sequence,
        firmas: Sequence,
        replicas_por_tarea: Sequence[int],
        procesos_n: int,
        etiqueta: str,
    ) -> list[list[float]]:
        """Resultado de cada tarea, en orden: los guardados se leen; los demás se calculan por tandas de
        2·procesos tareas, se guardan al terminar cada tanda y se informa el progreso."""
        res: list = [self._leer(f) for f in firmas]
        faltan = [i for i, r in enumerate(res) if r is None]
        total = int(sum(replicas_por_tarea))
        hechas = total - int(sum(replicas_por_tarea[i] for i in faltan))
        self.reutilizados += len(tareas) - len(faltan)
        if not faltan:
            self.registro(f"[{etiqueta}] {total}/{total} réplicas (todas guardadas)")
            return res
        if hechas:
            self.registro(f"[{etiqueta}] retoma: {hechas}/{total} réplicas ya guardadas")
        tanda = max(1, 2 * procesos_n)
        t0 = time.perf_counter()
        calc = 0
        for k in range(0, len(faltan), tanda):
            idx = faltan[k : k + tanda]
            sub = [tareas[i] for i in idx]
            out = [funcion(t) for t in sub] if procesos_n <= 1 else procesos.mapear(funcion, sub, procesos=procesos_n)
            for i, r in zip(idx, out, strict=True):
                self._guardar(firmas[i], r)
                res[i] = r
                calc += replicas_por_tarea[i]
            self.calculados += len(idx)
            hechas_ahora = hechas + calc
            vel = calc / max(time.perf_counter() - t0, 1e-9)
            resta = (total - hechas_ahora) / vel if vel > 0 else float("nan")
            self.registro(
                f"[{etiqueta}] {hechas_ahora}/{total} réplicas; {vel * 3600:.0f} réplicas/h; "
                f"faltan {resta / 60:.1f} min en esta simulación; {self.transcurrido()} desde el inicio"
            )
        return res

    def transcurrido(self) -> str:
        s = int(time.perf_counter() - self.inicio)
        return f"{s // 3600} h {s % 3600 // 60:02d} min"


def procesos_por_defecto() -> int:
    """Núcleos físicos, suponiendo dos hilos por núcleo (hyperthreading): en cálculo numérico, más procesos que
    núcleos no acelera y baja la frecuencia."""
    return max(1, (os.cpu_count() or 2) // 2)
