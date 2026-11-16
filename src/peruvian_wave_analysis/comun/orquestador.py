"""Registro de etapas, huella de cada etapa y estado: qué hay que volver a correr.

**Huella de una etapa.** Cubre su código propio (un paquete o un módulo, más el código extra declarado), el cierre
de sus importaciones dentro de ``peruvian_wave_analysis`` (también las hechas dentro de funciones), los YAML junto a
los módulos importados y ``uv.lock``: un cambio de versión de una dependencia puede cambiar los resultados. La
huella es semántica (``huella.huella_codigo``): comentarios, docstrings y formato no cuentan. La que registra
``etapa.publicar`` y la que compara el orquestador salen de la misma función, ``rutas_codigo``.

**Estado de una etapa** (``diagnostico``):

- *desactualizada*: sin salida publicada (o con el gate fallido), huella distinta de la publicada, o algún insumo
  registrado que cambió o ya no existe;
- *pendiente*: su código y sus insumos están al día, pero depende de una etapa que no lo está y cuyos archivos
  registra como insumos. Si esa etapa vuelve a publicar salidas idénticas, la pendiente queda al día sin correr;
- *al día*: lo demás.

Las dependencias son las declaradas más las deducidas del manifiesto: una etapa que registra un insumo dentro de
la salida de otra depende de ella aunque no lo declare.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import manifiesto
from . import huella, json_estricto
from .etapa import GATE_FALLIDO, MANIFIESTO

PAQUETE_RAIZ = "peruvian_wave_analysis"
DIR_PAQUETE = Path(__file__).resolve().parents[1]  # src/peruvian_wave_analysis
DIR_PROYECTO = huella.DIR_PROYECTO
LOCK = "uv.lock"

AL_DIA, DESACTUALIZADA, PENDIENTE = "al_dia", "desactualizada", "pendiente"


@dataclass(frozen=True)
class Etapa:
    nombre: str
    codigo: Path  # paquete o módulo propio de la etapa
    salida: Path  # salida publicada (dentro de WAVE-DATA)
    depende_de: tuple[str, ...] = field(default_factory=tuple)
    codigo_extra: tuple[Path, ...] = field(default_factory=tuple)


# --- Cierre de importaciones -------------------------------------------------------------------


def _archivos_de_modulo(nombre: str, raiz: Path) -> tuple[list[Path], bool]:
    """Archivos que ejecuta ``import <nombre>`` dentro de ``raiz`` (el paquete) y si el último es un paquete."""
    partes = nombre.split(".")
    if partes[0] != PAQUETE_RAIZ:
        return [], False
    out, actual, es_paquete = [], raiz, True
    if (raiz / "__init__.py").is_file():
        out.append(raiz / "__init__.py")
    for p in partes[1:]:
        actual = actual / p
        if (actual / "__init__.py").is_file():
            out.append(actual / "__init__.py")
            es_paquete = True
        elif actual.with_suffix(".py").is_file():
            out.append(actual.with_suffix(".py"))
            es_paquete = False
            break
        else:
            break
    return out, es_paquete


def _modulo_de_archivo(archivo: Path, raiz: Path) -> list[str]:
    partes = list(archivo.relative_to(raiz.parent).with_suffix("").parts)
    return partes[:-1] if partes[-1] == "__init__" else partes


def importaciones(archivo: Path, raiz: Path = DIR_PAQUETE) -> set[Path]:
    """Archivos del paquete que importa ``archivo``, en cualquier lugar del código."""
    arbol = ast.parse(archivo.read_bytes(), filename=str(archivo))
    out: set[Path] = set()

    def anadir(nombre: str, paquete_entero: bool) -> None:
        archivos, es_paquete = _archivos_de_modulo(nombre, raiz)
        out.update(archivos)
        if paquete_entero and es_paquete and archivos:
            out.update(p for p in archivos[-1].parent.rglob("*.py") if "__pycache__" not in p.parts)

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                anadir(alias.name, paquete_entero=True)
        elif isinstance(nodo, ast.ImportFrom):
            if nodo.level:
                base = _modulo_de_archivo(archivo, raiz)
                if archivo.name != "__init__.py":
                    base = base[:-1]
                base = base[: len(base) - (nodo.level - 1)]
                modulo = ".".join(base + ([nodo.module] if nodo.module else []))
            else:
                modulo = nodo.module or ""
            estrella = any(a.name == "*" for a in nodo.names)
            anadir(modulo, paquete_entero=estrella)
            for alias in nodo.names:
                if alias.name != "*":
                    anadir(f"{modulo}.{alias.name}", paquete_entero=True)
    return out


def _dentro(p: Path, bases: Sequence[Path]) -> bool:
    return any(p == b or (b.is_dir() and p.is_relative_to(b)) for b in bases)


def modulos_importados(et: Etapa, raiz: Path = DIR_PAQUETE) -> list[Path]:
    """Cierre de importaciones fuera del código propio de la etapa, con los YAML junto a cada módulo."""
    raiz = Path(raiz).resolve()
    propios = [Path(et.codigo).resolve(), *(Path(e).resolve() for e in et.codigo_extra)]
    pendientes = [
        p
        for b in propios
        for p in ([b] if b.is_file() else sorted(b.rglob("*.py")))
        if p.suffix == ".py" and "__pycache__" not in p.parts
    ]
    vistos: set[Path] = set()
    while pendientes:
        f = pendientes.pop()
        if f in vistos:
            continue
        vistos.add(f)
        pendientes.extend(g.resolve() for g in importaciones(f, raiz) if g.resolve() not in vistos)
    fuera = {f for f in vistos if not _dentro(f, propios)}
    yamls = {y for f in fuera for y in f.parent.iterdir() if y.suffix in (".yaml", ".yml") and not _dentro(y, propios)}
    return sorted(fuera | yamls, key=lambda p: p.relative_to(raiz).as_posix())


def rutas_codigo(et: Etapa, raiz: Path = DIR_PAQUETE, proyecto: Path = DIR_PROYECTO) -> list[Path]:
    """Rutas cuya huella registra la publicación y compara el orquestador. Único punto de verdad."""
    lock = Path(proyecto) / LOCK
    return [
        Path(et.codigo),
        *map(Path, et.codigo_extra),
        *modulos_importados(et, raiz),
        *([lock] if lock.is_file() else []),
    ]


def huella_etapa(et: Etapa, raiz: Path = DIR_PAQUETE, proyecto: Path = DIR_PROYECTO) -> str:
    return huella.huella_codigo(rutas_codigo(et, raiz, proyecto), base=proyecto)


# --- Orden y estado -----------------------------------------------------------------------------


def orden_topologico(etapas: Sequence[Etapa]) -> list[str]:
    por_nombre = {e.nombre: e for e in etapas}
    for e in etapas:
        for d in e.depende_de:
            if d not in por_nombre:
                raise ValueError(f"La etapa {e.nombre} depende de una etapa desconocida: {d}")
    orden: list[str] = []
    marca: dict[str, int] = {}

    def visitar(n: str) -> None:
        if marca.get(n) == 2:
            return
        if marca.get(n) == 1:
            raise ValueError(f"Hay un ciclo de dependencias que pasa por {n}")
        marca[n] = 1
        for d in sorted(por_nombre[n].depende_de):
            visitar(d)
        marca[n] = 2
        orden.append(n)

    for n in sorted(por_nombre):
        visitar(n)
    return orden


def _manifiesto(salida: Path) -> dict | None:
    p = Path(salida) / MANIFIESTO
    return json_estricto.leer(p) if p.is_file() else None


def insumos_cambiados(salida: Path, raiz_datos: Path) -> list[str]:
    """Insumos registrados en el manifiesto cuyo archivo cambió o ya no existe."""
    m = _manifiesto(salida) or {}
    out = []
    for nombre, ins in m.get("insumos", {}).items():
        p = Path(raiz_datos) / ins["ruta"]
        if not p.is_file() or manifiesto.sha256(p) != ins["sha256"]:
            out.append(nombre)
    return sorted(out)


def _registra_insumos_de(salida: Path, dep_salida: Path, raiz_datos: Path) -> bool:
    m = _manifiesto(salida) or {}
    dep = Path(dep_salida).resolve()
    return any((Path(raiz_datos) / ins["ruta"]).resolve().is_relative_to(dep) for ins in m.get("insumos", {}).values())


def dependencias_deducidas(etapas: Sequence[Etapa], raiz_datos: Path) -> list[Etapa]:
    out = []
    for e in etapas:
        deps = list(e.depende_de)
        for o in etapas:
            if o.nombre != e.nombre and o.nombre not in deps and _registra_insumos_de(e.salida, o.salida, raiz_datos):
                deps.append(o.nombre)
        out.append(replace(e, depende_de=tuple(deps)))
    return out


@dataclass(frozen=True)
class Estado:
    etapa: str
    estado: str  # AL_DIA, DESACTUALIZADA o PENDIENTE
    motivos: tuple[str, ...] = ()


def diagnostico(
    etapas: Sequence[Etapa], raiz_datos: Path, raiz: Path = DIR_PAQUETE, proyecto: Path = DIR_PROYECTO
) -> dict[str, Estado]:
    """Estado de cada etapa leyendo sus salidas publicadas, en orden topológico."""
    etapas = dependencias_deducidas(etapas, raiz_datos)
    por_nombre = {e.nombre: e for e in etapas}
    out: dict[str, Estado] = {}
    for n in orden_topologico(etapas):
        et = por_nombre[n]
        m = _manifiesto(et.salida)
        if m is None:
            motivo = "gate fallido" if (Path(et.salida) / GATE_FALLIDO).is_file() else "sin salida publicada"
            out[n] = Estado(n, DESACTUALIZADA, (motivo,))
            continue
        motivos = []
        if m.get("huella_version") != huella.VERSION or m.get("huella_codigo") != huella_etapa(et, raiz, proyecto):
            motivos.append("código: la huella actual no coincide con la publicada")
        if cambiados := insumos_cambiados(et.salida, raiz_datos):
            motivos.append("insumos cambiados: " + ", ".join(cambiados))
        if motivos:
            out[n] = Estado(n, DESACTUALIZADA, tuple(motivos))
            continue
        estado, motivos = AL_DIA, []
        for d in et.depende_de:
            if out[d].estado == AL_DIA:
                continue
            if _registra_insumos_de(et.salida, por_nombre[d].salida, raiz_datos):
                motivos.append(f"depende de {d}: se reevalúa cuando {d} se publique")
                estado = PENDIENTE if estado == AL_DIA else estado
            else:
                motivos.append(f"depende de {d} y no registra insumos suyos: no se puede cortar la cascada")
                estado = DESACTUALIZADA
        out[n] = Estado(n, estado, tuple(motivos))
    return out


def plan(etapas: Sequence[Etapa], raiz_datos: Path, incluir_pendientes: bool = True, **kw) -> list[str]:
    """Etapas que no están al día, en orden topológico."""
    validos = {DESACTUALIZADA, PENDIENTE} if incluir_pendientes else {DESACTUALIZADA}
    return [n for n, e in diagnostico(etapas, raiz_datos, **kw).items() if e.estado in validos]
