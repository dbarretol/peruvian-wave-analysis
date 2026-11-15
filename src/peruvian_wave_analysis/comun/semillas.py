"""Semillas derivadas de nombres, no del orden de ejecución.

Una semilla raíz del proyecto; cada (etapa, propósito, ...) deriva la suya con ``SeedSequence`` a partir de un hash
estable de los nombres. Ninguna función usa el estado global del generador.
"""

from __future__ import annotations

import hashlib

import numpy as np

SEMILLA_RAIZ = 20261115


def _entropia(nombres: tuple[str, ...]) -> list[int]:
    return [int.from_bytes(hashlib.sha256(n.encode("utf-8")).digest()[:8], "little") for n in nombres]


def secuencia(*nombres: str, raiz: int = SEMILLA_RAIZ) -> np.random.SeedSequence:
    if not nombres:
        raise ValueError("Hace falta al menos un nombre para derivar la semilla")
    return np.random.SeedSequence([raiz, *_entropia(nombres)])


def semilla(*nombres: str, raiz: int = SEMILLA_RAIZ) -> int:
    """Entero de 32 bits derivado de los nombres (para registrar en un manifiesto)."""
    return int(secuencia(*nombres, raiz=raiz).generate_state(1)[0])


def generador(*nombres: str, raiz: int = SEMILLA_RAIZ) -> np.random.Generator:
    return np.random.default_rng(secuencia(*nombres, raiz=raiz))
