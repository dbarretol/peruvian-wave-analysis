"""F5 · Validación externa (contrato v6 de la F5, aceptado el 2026-11-17).

Por ahora, solo las descargas: boyas NDBC (``ndbc``) y WAVERYS en la región y en las boyas (``waverys``).

**G4 también en los mensajes.** Nada de este subpaquete imprime, registra ni pone en un mensaje de error un valor
observado: ni el repr de un Dataset ni un ``head()`` de una boya. Los mensajes llevan solo nombres de archivo,
dimensiones, nombres de variable, número de registros y fechas extremas.
"""
