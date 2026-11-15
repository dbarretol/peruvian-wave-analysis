"""Publicación de una etapa: insumos con hash → staging → gate → salida publicada con manifiesto.

Ciclo::

    insumos = {"serie": ruta, ...}
    hashes = exigir_insumos(insumos, esperados={...})     # faltante o alterado → error
    with staging_protegido(salida, raices_codigo) as st:  # cualquier excepción antes de publicar borra la salida
        ... escribir salidas y figuras en st ...
        publicar(st, salida, Gate(...), raices_codigo, insumos, raiz_datos, semilla=...)

Reglas:

- Si el gate pasa, la salida publicada se reemplaza entera y lleva ``MANIFEST.json`` **sin marcas de tiempo**: dos
  corridas iguales dan bytes iguales. El manifiesto registra cada insumo con su ruta relativa a ``WAVE-DATA`` y su
  hash, cada salida con su hash, la huella del código, la semilla y los criterios del gate.
- Si el gate falla, o hay una excepción antes de publicar, **se borra la salida publicada anterior** y queda solo
  ``GATE_FAILED.json`` con el motivo. Ninguna etapa siguiente puede leer resultados viejos.
- La huella del código se calcula al abrir el staging y otra vez al publicar; si difieren, el gate falla: la
  salida no correspondería al código que la produjo. Por eso ``publicar`` solo acepta un staging abierto con
  ``abrir_staging`` (o ``staging_protegido``) en el mismo proceso.
- Leer una salida publicada (``exigir_publicada``) verifica el hash de cada archivo.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import manifiesto
from . import huella, json_estricto
from .errores import GateFallido, HashDistinto, InsumoFaltante

MANIFIESTO = "MANIFEST.json"
GATE_FALLIDO = "GATE_FAILED.json"

# huella del código al abrir cada staging (ruta resuelta del staging → huella); la consume ``publicar``
_HUELLAS_AL_ABRIR: dict[Path, str] = {}


@dataclass
class Gate:
    aprobado: bool
    criterios: dict[str, Any]
    motivos: list[str] = field(default_factory=list)


def ruta_relativa(ruta: Path, raiz_datos: Path) -> str:
    """Ruta POSIX relativa a WAVE-DATA. Un insumo fuera de WAVE-DATA es un error (proyecto autocontenido)."""
    try:
        return Path(ruta).resolve().relative_to(Path(raiz_datos).resolve()).as_posix()
    except ValueError as e:
        raise ValueError(f"El insumo {ruta} no está dentro del directorio de datos {raiz_datos}") from e


def exigir_insumos(insumos: Mapping[str, Path], esperados: Mapping[str, str] | None = None) -> dict[str, str]:
    """sha256 de cada insumo. Falta uno → InsumoFaltante; hash distinto del esperado → HashDistinto."""
    out = {}
    for nombre, p in insumos.items():
        p = Path(p)
        if not p.is_file():
            raise InsumoFaltante(f"Falta el insumo «{nombre}»: {p}")
        h = manifiesto.sha256(p)
        if esperados and nombre in esperados and esperados[nombre] != h:
            raise HashDistinto(f"sha256 de «{nombre}» ({p}) = {h}, se esperaba {esperados[nombre]}")
        out[nombre] = h
    return out


def ruta_staging(salida: Path) -> Path:
    salida = Path(salida)
    return salida.with_name(salida.name + ".staging")


def abrir_staging(salida: Path, raices_codigo: Sequence[Path], base_codigo: Path = huella.DIR_PROYECTO) -> Path:
    """Crea vacío el directorio de staging de una salida y registra la huella del código en ese momento."""
    st = ruta_staging(salida)
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    _HUELLAS_AL_ABRIR[st.resolve()] = huella.huella_codigo(raices_codigo, base_codigo)
    return st


def fallar(
    staging: Path,
    salida: Path,
    motivos: Sequence[str],
    raices_codigo: Sequence[Path],
    criterios: Mapping[str, Any] | None = None,
    base_codigo: Path = huella.DIR_PROYECTO,
) -> Path:
    """Borra la salida publicada y el staging y deja solo GATE_FAILED.json. No lanza la excepción."""
    staging, salida = Path(staging), Path(salida)
    shutil.rmtree(staging, ignore_errors=True)
    if salida.exists():
        shutil.rmtree(salida)
    salida.mkdir(parents=True)
    json_estricto.escribir(
        salida / GATE_FALLIDO,
        {
            "motivos": list(motivos),
            "criterios": dict(criterios or {}),
            "huella_codigo": huella.huella_codigo(raices_codigo, base_codigo),
            "huella_version": huella.VERSION,
        },
    )
    return salida


def publicar(
    staging: Path,
    salida: Path,
    gate: Gate,
    raices_codigo: Sequence[Path],
    insumos: Mapping[str, Path],
    raiz_datos: Path,
    semilla: int | None,
    base_codigo: Path = huella.DIR_PROYECTO,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Publica el staging como salida si el gate pasa; si no, aplica ``fallar`` y lanza GateFallido.

    También falla el gate si la huella del código cambió entre la apertura del staging y la publicación.
    """
    staging, salida = Path(staging), Path(salida)
    al_abrir = _HUELLAS_AL_ABRIR.pop(staging.resolve(), None)
    if al_abrir is None:
        raise ValueError(
            f"El staging {staging} no se abrió con abrir_staging en este proceso: falta la huella del código al empezar"
        )
    huella_final = huella.huella_codigo(raices_codigo, base_codigo)
    motivos = [] if gate.aprobado else list(gate.motivos or ["sin motivo"])
    if huella_final != al_abrir:
        motivos.append(
            f"El código cambió durante la corrida: huella {al_abrir} al abrir el staging y {huella_final} al publicar"
        )
    if motivos:
        fallar(staging, salida, motivos, raices_codigo, gate.criterios, base_codigo)
        raise GateFallido(f"Gate fallido en {salida.name}: {'; '.join(motivos)}")
    hashes = exigir_insumos(insumos)
    salidas = {
        p.relative_to(staging).as_posix(): manifiesto.sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()
    }
    if MANIFIESTO in salidas or GATE_FALLIDO in salidas:
        raise ValueError(f"El staging no puede contener {MANIFIESTO} ni {GATE_FALLIDO}")
    contenido: dict[str, Any] = {
        "gate": {"aprobado": True, "criterios": gate.criterios},
        "huella_codigo": huella_final,
        "huella_version": huella.VERSION,
        "insumos": {
            k: {"ruta": ruta_relativa(Path(v), raiz_datos), "sha256": hashes[k]} for k, v in sorted(insumos.items())
        },
        "salidas": salidas,
        "semilla": semilla,
    }
    if extra:
        contenido["extra"] = dict(extra)
    texto = json_estricto.dumps(contenido)  # un tipo no serializable falla aquí, antes de tocar la salida
    if salida.exists():
        shutil.rmtree(salida)
    (staging / MANIFIESTO).write_text(texto, encoding="utf-8", newline="\n")
    salida.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(salida)
    return salida


def exigir_publicada(salida: Path) -> dict[str, Any]:
    """Manifiesto de una salida publicada, tras verificar el hash de cada archivo."""
    salida = Path(salida)
    if (salida / GATE_FALLIDO).exists():
        raise GateFallido(f"La salida {salida} tiene el gate fallido; no se puede usar")
    if not (salida / MANIFIESTO).is_file():
        raise InsumoFaltante(f"No hay salida publicada en {salida}")
    man = json_estricto.leer(salida / MANIFIESTO)
    presentes = {p.relative_to(salida).as_posix() for p in salida.rglob("*") if p.is_file()} - {MANIFIESTO}
    sobrantes = presentes - set(man["salidas"])
    if sobrantes:
        raise HashDistinto(f"La salida {salida} tiene archivos que no están en su manifiesto: {sorted(sobrantes)}")
    for rel, h in man["salidas"].items():
        f = salida / rel
        if not f.is_file():
            raise InsumoFaltante(f"Falta la salida publicada {f}")
        if manifiesto.sha256(f) != h:
            raise HashDistinto(f"La salida {f} cambió después de publicarse")
    return man


@contextmanager
def staging_protegido(
    salida: Path, raices_codigo: Sequence[Path], base_codigo: Path = huella.DIR_PROYECTO
) -> Iterator[Path]:
    """Abre el staging; si algo falla antes de publicar, borra la salida publicada y relanza.

    ``publicar`` consume el staging (lo renombra o lo borra): si sigue existiendo al salir con error, el fallo
    ocurrió antes de publicar.
    """
    st = abrir_staging(salida, raices_codigo, base_codigo)
    try:
        yield st
    except Exception as exc:
        if st.exists():
            fallar(st, salida, [f"{type(exc).__name__}: {exc}"], raices_codigo, base_codigo=base_codigo)
        raise
