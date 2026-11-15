"""Huella semántica del código y la configuración.

Un ``.py`` cuenta por su árbol sintáctico sin docstrings (los comentarios no llegan al árbol); un YAML, por su
contenido cargado con sus tipos. Documentar, comentar o reformatear no cambia la huella; un cambio de lógica sí.
Cualquier otro archivo (por ejemplo ``uv.lock``) cuenta por sus bytes, con los fines de línea normalizados.

La huella lleva su versión y la de Python, porque el volcado del árbol sintáctico puede cambiar entre versiones.
El sha256 de un archivo es el de ``manifiesto.sha256``: no se duplica.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

VERSION = 1
DIR_PROYECTO = Path(__file__).resolve().parents[3]  # raíz del repositorio (src/peruvian_wave_analysis/comun → raíz)
SUFIJOS = (".py", ".yaml", ".yml")
_ETIQUETA = f"huella-v{VERSION}-ast-py{sys.version_info.major}.{sys.version_info.minor}".encode()


def _sin_docstrings(arbol: ast.AST) -> ast.AST:
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            cuerpo = nodo.body
            if (cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant)
                    and isinstance(cuerpo[0].value.value, str)):
                nodo.body = cuerpo[1:] or [ast.Pass()]
    return arbol


def _canonico(x: object) -> object:
    """Valor YAML con su tipo explícito: ``1`` ≠ ``"1"``, ``true`` ≠ ``"true"``. Conserva el orden de las claves."""
    if isinstance(x, dict):
        return ["map", [[_canonico(k), _canonico(v)] for k, v in x.items()]]
    if isinstance(x, list):
        return ["seq", [_canonico(v) for v in x]]
    if isinstance(x, (set, frozenset)):
        return ["set", sorted((_canonico(v) for v in x), key=lambda v: json.dumps(v, ensure_ascii=False))]
    if x is None:
        return ["null"]
    return [type(x).__name__, repr(x)]


def contenido_semantico(nombre: str, datos: bytes) -> bytes:
    """Forma canónica de un archivo para la huella."""
    datos = datos.replace(b"\r\n", b"\n")
    if nombre.endswith(".py"):
        try:
            arbol = ast.parse(datos, filename=nombre)
        except SyntaxError:
            return b"RAW:" + datos
        return b"PY:" + ast.dump(_sin_docstrings(arbol), include_attributes=False).encode("utf-8")
    if nombre.endswith((".yaml", ".yml")):
        try:
            contenido = yaml.safe_load(datos)
        except yaml.YAMLError:
            return b"RAW:" + datos
        return b"YAML:" + json.dumps(_canonico(contenido), ensure_ascii=False).encode("utf-8")
    return b"RAW:" + datos


def _archivos(raiz: Path) -> list[tuple[str, Path]]:
    if raiz.is_file():
        return [(raiz.name, raiz)]
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe la raíz de código {raiz}")
    return [(p.relative_to(raiz).as_posix(), p) for p in sorted(raiz.rglob("*"))
            if p.is_file() and p.suffix in SUFIJOS and "__pycache__" not in p.parts]


def huella_codigo(raices: Iterable[Path], base: Path | None = DIR_PROYECTO) -> str:
    """Huella semántica de los archivos bajo cada raíz (directorio entero o archivo), en orden estable.

    Cada archivo se identifica por su ruta relativa a ``base`` (si cae dentro) o a su raíz, nunca por la ruta
    absoluta: mover el proyecto de carpeta no cambia la huella.
    """
    h = hashlib.sha256(_ETIQUETA + b"\n")
    for raiz in raices:
        raiz = Path(raiz)
        for rel, p in _archivos(raiz):
            if base is not None:
                try:
                    rel = p.resolve().relative_to(Path(base).resolve()).as_posix()
                except ValueError:
                    pass
            h.update(b"FILE:" + rel.encode("utf-8") + b"\n")
            h.update(contenido_semantico(p.name, p.read_bytes()) + b"\n")
    return h.hexdigest()
