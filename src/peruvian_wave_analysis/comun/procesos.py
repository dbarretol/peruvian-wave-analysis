"""Cálculo repartido en procesos con un hilo de BLAS por proceso (contrato de la F2, §3.9).

Las bibliotecas de álgebra lineal leen ``OMP_NUM_THREADS`` y afines **al cargarse**. Fijar la variable en el proceso
que ya importó numpy llega tarde y nada lo avisa. Por eso las tareas corren **siempre** en procesos hijos nuevos
(``spawn``), también con un solo proceso: cada hijo es un intérprete nuevo que hereda la variable antes de importar
numpy. Así 1 y N procesos ejecutan la misma configuración, sin depender del orden de importación de quien llama.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

VARIABLES_HILOS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def mapear(funcion: Callable[..., Any], *iterables: Iterable, procesos: int, chunksize: int = 1) -> list:
    """``list(map(funcion, *iterables))`` en ``procesos`` hijos nuevos con un hilo de BLAS cada uno, en orden.

    ``funcion`` debe poder importarse desde su módulo (los hijos se crean con ``spawn``).
    """
    if procesos < 1:
        raise ValueError(f"procesos debe ser ≥ 1: {procesos}")
    antes = {v: os.environ.get(v) for v in VARIABLES_HILOS}
    os.environ.update({v: "1" for v in VARIABLES_HILOS})  # los hijos heredan el entorno al crearse
    try:
        with ProcessPoolExecutor(max_workers=procesos, mp_context=multiprocessing.get_context("spawn")) as ex:
            return list(ex.map(funcion, *iterables, chunksize=max(1, chunksize)))
    finally:
        for v, valor in antes.items():
            if valor is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = valor


def mapear_a_medida(funcion: Callable[[Any], Any], argumentos: Iterable, procesos: int) -> Iterator[tuple[int, Any]]:
    """Como ``mapear`` con un argumento, pero entrega ``(índice, resultado)`` a medida que cada tarea termina.

    Sirve para informar el avance de corridas largas; el orden de llegada no es el de las tareas.
    """
    if procesos < 1:
        raise ValueError(f"procesos debe ser ≥ 1: {procesos}")
    antes = {v: os.environ.get(v) for v in VARIABLES_HILOS}
    os.environ.update({v: "1" for v in VARIABLES_HILOS})
    try:
        with ProcessPoolExecutor(max_workers=procesos, mp_context=multiprocessing.get_context("spawn")) as ex:
            futuros = {ex.submit(funcion, a): i for i, a in enumerate(argumentos)}
            for fu in as_completed(futuros):
                yield futuros[fu], fu.result()
    finally:
        for v, valor in antes.items():
            if valor is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = valor


def entorno_del_proceso(_: object = None) -> dict[str, Any]:
    """Diagnóstico: identificador del proceso y variables de hilos que ve (para pruebas y registros)."""
    return {"pid": os.getpid(), **{v: os.environ.get(v) for v in VARIABLES_HILOS}}
