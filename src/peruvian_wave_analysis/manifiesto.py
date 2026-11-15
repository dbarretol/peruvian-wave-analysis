"""Manifiesto de datos crudos: hash sha256 y metadatos de origen de cada archivo.

Reglas (roadmap §8):
- Los datos crudos son de solo lectura: un archivo registrado no se sobrescribe.
- Un insumo sin registrar o con hash distinto al registrado es un error.

El manifiesto vive en la raíz de ``raw/`` en dos archivos:
- ``MANIFEST.sha256``: una línea ``<sha256>  <ruta relativa>`` por archivo (verificable con ``sha256sum -c``);
- ``MANIFEST.json``: los mismos archivos con su origen, parámetros de la consulta y fecha de acceso.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NOMBRE_SHA = "MANIFEST.sha256"
NOMBRE_JSON = "MANIFEST.json"


class ErrorManifiesto(RuntimeError):
    """Insumo crudo ausente del manifiesto o con hash que no coincide."""


def sha256(ruta: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def _relativa(raiz: Path, ruta: Path) -> str:
    return Path(ruta).resolve().relative_to(Path(raiz).resolve()).as_posix()


def leer(raiz: Path) -> dict[str, dict[str, Any]]:
    p = Path(raiz) / NOMBRE_JSON
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _escribir(raiz: Path, entradas: dict[str, dict[str, Any]]) -> None:
    raiz = Path(raiz)
    ordenadas = dict(sorted(entradas.items()))
    (raiz / NOMBRE_JSON).write_text(json.dumps(ordenadas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lineas = [f"{e['sha256']}  {rel}" for rel, e in ordenadas.items()]
    (raiz / NOMBRE_SHA).write_text("\n".join(lineas) + "\n", encoding="utf-8")


def registrar(raiz: Path, ruta: Path, origen: dict[str, Any]) -> dict[str, Any]:
    """Registra un archivo recién descargado. Si ya estaba registrado con otro hash, es un error."""
    rel = _relativa(raiz, ruta)
    entradas = leer(raiz)
    h = sha256(ruta)
    if rel in entradas and entradas[rel]["sha256"] != h:
        raise ErrorManifiesto(f"{rel} ya está registrado con otro hash; los datos crudos no se sobrescriben.")
    entrada = {
        "sha256": h,
        "bytes": Path(ruta).stat().st_size,
        "fecha_acceso_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **origen,
    }
    if rel in entradas:  # mismo archivo, mismo hash: se conserva la fecha de acceso original
        entrada["fecha_acceso_utc"] = entradas[rel]["fecha_acceso_utc"]
    entradas[rel] = entrada
    _escribir(raiz, entradas)
    return entrada


def verificar(raiz: Path, ruta: Path) -> dict[str, Any]:
    """Comprueba que un insumo existente está registrado y que su hash coincide. Devuelve su entrada."""
    rel = _relativa(raiz, ruta)
    entradas = leer(raiz)
    if rel not in entradas:
        raise ErrorManifiesto(f"{rel} no está en el manifiesto de {raiz}: insumo sin origen registrado.")
    h = sha256(ruta)
    if entradas[rel]["sha256"] != h:
        raise ErrorManifiesto(f"El hash de {rel} no coincide con el manifiesto: el archivo cambió.")
    return entradas[rel]
