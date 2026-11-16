"""Errores que detienen una etapa. Un insumo faltante o alterado es un error, nunca una advertencia."""

from __future__ import annotations


class ErrorGate(Exception):
    """Base de todo fallo que debe detener una etapa."""


class InsumoFaltante(ErrorGate):
    """Falta un insumo declarado, o una salida publicada que otra etapa necesita."""


class HashDistinto(ErrorGate):
    """Un insumo existe pero su sha256 no coincide con el esperado, o una salida cambió después de publicarse."""


class GateFallido(ErrorGate):
    """El gate cuantitativo no se cumplió, o se intenta leer una salida con el gate fallido."""
