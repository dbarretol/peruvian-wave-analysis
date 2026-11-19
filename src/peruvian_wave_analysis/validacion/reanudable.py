"""Cálculo por tareas que se puede reanudar si la corrida se interrumpe (corte de luz, cierre de la ventana, Ctrl+C).

Cada tarea guarda su resultado en un archivo propio en cuanto termina, con escritura atómica (temporal y renombrado).
Al relanzar, las tareas con archivo se cargan y solo se calculan las que faltan. El directorio de avance lleva una
**clave** que depende de la huella del código y de los hashes de los insumos: si cambia cualquiera de los dos, la clave
cambia y no se reutiliza nada. Los resultados son los mismos con o sin interrupción, porque cada réplica tiene su
propia semilla derivada de su nombre.

Los archivos son ``pickle`` escritos por esta misma etapa en ``interim/``; no se leen archivos de otro origen.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..comun import procesos


def clave(*partes: Any) -> str:
    """Hash corto y estable de las partes (serializadas como JSON ordenado)."""
    texto = json.dumps(partes, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def _guardar(ruta: Path, valor: Any) -> None:
    tmp = ruta.with_name(ruta.name + ".parcial")
    with open(tmp, "wb") as f:
        pickle.dump(valor, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ruta)  # atómico: o está entero o no está


def _cargar(ruta: Path) -> Any:
    with open(ruta, "rb") as f:
        return pickle.load(f)  # noqa: S301 (archivo propio de esta etapa, en interim/)


def _tarea(args) -> bool:
    """Ejecuta una tarea en un proceso hijo y guarda su resultado. Devuelve True al terminar."""
    funcion, argumentos, ruta = args
    _guardar(Path(ruta), funcion(argumentos))
    return True


def ejecutar(
    funcion: Callable[[Any], Any],
    tareas: Sequence[tuple[str, Any]],
    directorio: Path,
    procesos_: int,
    avisar: Callable[[str], None] | None = None,
) -> list[Any]:
    """Resultados de ``funcion(args)`` para cada ``(nombre, args)``, en orden, reanudando lo ya guardado.

    ``procesos_ = 0`` ejecuta en este mismo proceso (pruebas); si no, en procesos hijos (``procesos.mapear``).
    ``funcion`` debe poder importarse desde su módulo. Con ``avisar``, informa el avance unas 20 veces (tareas
    hechas, minutos transcurridos y una estimación de lo que falta).
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    for resto in directorio.glob("*.parcial"):
        resto.unlink()  # restos de una escritura cortada
    rutas = [directorio / f"{nombre}.pkl" for nombre, _ in tareas]
    pendientes = [(funcion, args, str(r)) for (_, args), r in zip(tareas, rutas, strict=True) if not r.exists()]
    if avisar:
        avisar(f"{len(tareas) - len(pendientes)} de {len(tareas)} tareas ya guardadas; faltan {len(pendientes)}")
    if pendientes:
        t0 = time.time()
        total = len(pendientes)
        paso = max(1, total // 20)  # unas 20 líneas de avance, y siempre la última

        def informar(hechas: int) -> None:
            if avisar and (hechas % paso == 0 or hechas == total):
                seg = time.time() - t0
                resto = seg / hechas * (total - hechas)
                avisar(f"{hechas}/{total} tareas, {seg / 60:.1f} min; faltan unos {resto / 60:.1f} min")

        if procesos_ == 0:
            for k, p in enumerate(pendientes, 1):
                _tarea(p)
                informar(k)
        else:
            for k, _ in enumerate(procesos.mapear_a_medida(_tarea, pendientes, procesos_), 1):
                informar(k)
    return [_cargar(r) for r in rutas]


def limpiar(directorio: Path) -> None:
    """Borra el avance una vez publicada la etapa."""
    shutil.rmtree(directorio, ignore_errors=True)
