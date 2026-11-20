# peruvian-wave-analysis

Análisis de oleaje extremo y contornos ambientales Hs–Tp a partir del reanálisis WAVERYS de Copernicus Marine.

## Estado

Fase 1 (sitio y fuentes) completa: mapa de selección de sitio, serie de WAVERYS y ficha del sitio.

Fase 2 (biblioteca común) completa:
- `comun/`: publicación de etapas con staging, gate y `MANIFEST.json`; huella semántica del código, comprobada al abrir el staging y al publicar; orquestador; semillas por nombre; procesos con un hilo de BLAS.
- `extremos/`: ajuste GPD, niveles de retorno, IC por perfil de verosimilitud, estratos, declustering, tasas, prueba de Poisson por reescalado de tiempo y bootstrap por años.
- `verificacion-f2`: estudio de verificación por simulación de todo lo anterior.

Fase 3 (datos: ingesta, control de calidad e integridad) cerrada, en dos etapas:
- `datos-candidatos`: ingesta contra el gate de la serie, particiones ausentes, centinelas y acumulaciones, período saturado, límites físicos, peralte, coherencia, calibración de k con series sintéticas, detector bilateral de picos, dipolo, los 20 mayores máximos independientes y una figura por máximo y por candidato.
- `datos-serie`: aplica las resoluciones versionadas en `config/f03_resoluciones.yaml` y publica la serie del nodo con una columna por regla y `valido`, la cobertura, las marcas por año y por regla, Tp largo con la razón de asomeramiento y la lista de registros dudosos para la sensibilidad de los niveles de retorno.

Fase 4 (homogeneidad de la fuente) cerrada, en dos etapas (`homogeneidad/`):
- `homogeneidad-altimetros`: exposición del nodo a pasadas de altímetros por misión, pasadas y misiones con datos locales.
- `homogeneidad`: cronología del producto, tramos, cuantiles por tramo, pruebas confirmatorias de cambio con su potencia escrita antes que cualquier estadístico observado, prueba de saltos frente a pasadas de altímetros y escenarios de Hs.

Fase 5 (validación externa) cerrada, en cuatro órdenes (`validacion/` y `validacion_obs/`):
- `validacion-descarga`: descarga, con hash, las boyas NDBC 32302 y 32012, WAVERYS en sus nodos y WAVERYS regional.
- `validacion-modelo`: control de calidad regional del modelo y su coherencia con la serie del nodo.
- `validacion-potencia`: colocaciones con la altimetría y las boyas, escalón del modelo en cada fuente y potencia de cada prueba, sin leer valores observados. Simulación con SIMEX, bootstrap por bloques mensuales y max-T; réplicas reanudables.
- `validacion`: lee los valores observados solo después de comprobar el manifiesto y los hashes de la potencia. Mide el ruido de 1 Hz, el sesgo por cuantil y las pruebas de atribución y de sesgo, y publica los escenarios activos, las variantes de corrección, la serie corregida y la fila de incertidumbre de la fuente.

Fase 6 (clima de oleaje) cerrada, en `clima/` y `extremos/`:
- `extremos/dependencia.py`: índice extremal de Ferro y Segers dentro del invierno, declustering automático, rachas y corte natural de los huecos, extremograma con referencia estacional y escalas de decorrelación.
- `extremos/gev.py`: GEV de máximos anuales con validación de soporte, nivel de retorno de ADR-0022 y de Coles e IC de perfil.
- `clima/clase_fisica.py`: clase física por particiones (swell, mixto, mar de viento).
- `clima/estratificacion.py`: mezclas de dos estratos GPD, frontera del efecto de 2 % en el RL100, Anderson–Darling, razón de verosimilitudes y elección del α de familia.
- `clima-etapa0`: etapa 0 sintética, factor de la regla de concordancia (publicada).
- `clima-simulacion-previa`: reproducción de la simulación previa del contrato (§5 y §5.4), sin datos del sitio (publicada).
- `clima-etapa1`: estratificación sin leer excesos: τ_e y separación provisional, picos provisionales (fechas y direcciones), bimodalidad de `VPED`, conteos por estrato en las cuatro ramas, fronteras para la fracción real, potencia de Anderson–Darling, α de familia y G7b.
- `clima` (`clima/run.py`, con `escalas.py`, `estratos_sitio.py`, `gev_ramas.py`, `tablas.py` y `figuras.py`): etapa principal. Antes de leer excesos verifica que la etapa 1 esté publicada, sin cambios y con la misma huella de código e insumos (G7). Después: prueba de Anderson–Darling con el α congelado y escenarios, dependencia con IC por bloques (B = 999), descriptivos, GEV por rama, figuras y gate. **No se puede cambiar ningún archivo que use la etapa 1** (su huella cubre `picos.py`, `insumos.py`, `descriptivos.py`, `estratificacion.py`, `extremos/dependencia.py`, `estilo.py`, `uv.lock` y otros) sin repetirla.

Primera herramienta: **mapa de selección de sitio**, en dos pasos.

1. **Mapa exploratorio** (`mapa-sitio`): dada una coordenada de referencia, descarga la batimetría (GMRT), la malla de nodos de WAVERYS y las zonas protegidas y de hidrocarburos, y dibuja las isóbatas elegidas (por defecto 200, 100 y 50 m), la malla con el recuadro de cada celda, las zonas de exclusión y de colisión de usos, y hasta 3 nodos candidatos.
2. **Mapa definitivo** (`mapa-sitio-final`): con el nodo ya elegido y un recorte de GEBCO descargado a mano, rehace el mapa **sin descargar nada**, verificando el hash de cada insumo, y comprueba que el nodo cumple los criterios.

Después:

3. **Serie de WAVERYS**, en dos comandos: `serie-waverys` consulta el catálogo y descarga el registro completo del bloque de 5 × 5 nodos alrededor del nodo, un archivo por año, con la versión del dataset fija, reanudación y hash por archivo; `serie-waverys-gate` verifica la serie sin red y publica su gate.
4. **Ficha del sitio** (`ficha-sitio`): exposición por sectores (Natural Earth), obstáculos (GEBCO), inventario de observaciones de altimetría (Ribal y Young, 2019, en AODN) con su independencia respecto de WAVERYS, ciclones tropicales (IBTrACS), cronología del producto, gate de la fase y `ficha_sitio.md`. No lee valores de Hs ni de Tp.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12 o 3.13.

```bash
uv sync
uv run copernicusmarine login   # una sola vez: credenciales de Copernicus Marine
```

Los datos no se guardan en el repositorio. Indicar la carpeta de datos con la variable de entorno `WAVE_DATA_DIR` (o `directorio_datos` en la configuración):

```powershell
$env:WAVE_DATA_DIR = "C:\ruta\a\WAVE-DATA"
```

## Uso

### 1. Mapa exploratorio

```bash
uv run mapa-sitio --nombre marcona --lat -15.4 --lon -75.3
uv run mapa-sitio --nombre marcona --lat -15.4 --lon -75.3 --isobatas 300 200 100 --semiancho 1.5
uv run mapa-sitio --config config/mapa_sitio.ejemplo.yaml
uv run mapa-sitio --nombre marcona --lat -15.4 --lon -75.3 --punto "Parque eólico (aprox.)" "15°23′20″S" "75°04′15″O"
```

Al terminar indica el recuadro de GEBCO que hay que descargar, y lo deja escrito en `siguiente_paso.md`.

### 2. Descargar y registrar GEBCO

Descargar el recorte en <https://download.gebco.net/> (formato NetCDF) con el recuadro indicado y registrarlo:

```bash
uv run registrar-gebco --archivo C:/Descargas/gebco_2025_recorte.nc --version GEBCO_2025
```

### 3. Mapa definitivo

```bash
uv run mapa-sitio-final \
  --exploratorio $WAVE_DATA_DIR/interim/F01-sitio-y-fuentes/mapa-sitio/marcona/metadatos.json \
  --gebco $WAVE_DATA_DIR/raw/batimetria/gebco/gebco_2025_recorte.nc \
  --nodo-lat -15.6 --nodo-lon -75.2
```

Si el nodo no cumple los criterios con GEBCO, termina con `GATE FALLIDO` y borra la salida definitiva anterior.

### 4. Serie de WAVERYS

```bash
uv run serie-waverys                 # descarga (con red): registro completo, solo escribe en raw/
uv run serie-waverys --anio 2020     # descarga de un año
uv run serie-waverys-gate            # gate (sin red): verifica y publica serie_gate.json
```

La descarga no publica nada: si el catálogo de hoy difiere del anotado al descargar, lo dice en la consola. El gate usa la cobertura del catálogo anotada en el manifiesto el día de la descarga, exige todos los años de esa cobertura y registra el hash de cada archivo. Si falla, termina con `GATE FALLIDO` y borra el informe anterior.

### 5. Ficha del sitio

```bash
uv run ficha-sitio              # todo (≈ 8 min; descarga Natural Earth, teselas de AODN e IBTrACS si faltan)
uv run ficha-sitio --solo-md    # rehace solo ficha_sitio.md desde las salidas ya publicadas
```

Requiere el mapa definitivo (`ficha_nodo.json`), el gate publicado de la serie y un recorte de GEBCO que cubra 200 km alrededor del nodo y del respaldo. Registra cada insumo con su hash. Si el gate de la fase no cumple, termina con código 1 y borra la ficha anterior.

### 6. Estudio de verificación de la biblioteca de extremos

```bash
uv run verificacion-f2 --procesos 2   # ≈ 2 h con 2 procesos (medido: 1 h 56 min); el resultado no depende del número de procesos
```

Simula escenarios de respuesta conocida y comprueba la cobertura de los intervalos por perfil y el tamaño de la prueba de Poisson contra bandas fijadas antes de correr. Si una banda falla, repite ese escenario una vez con 1600 réplicas.

### 7. Datos: control de calidad de la serie del nodo

```bash
uv run datos-candidatos   # ≈ 6 min: calibración de k con 10 000 años sintéticos por generador
uv run datos-serie        # segundos; exige una resolución por candidato manual y por máximo
```

`datos-candidatos` no exige resoluciones. Se detiene con «revisión requerida» si se supera un tope: 47 descartes automáticos, 5 en un año, 100 candidatos de revisión en la cola o 100 registros de una marca que exige resolución. `datos-serie` falla sin tocar la salida de la primera si falta una resolución, si una entrada no corresponde a un candidato ni a un máximo, si dos entradas se contradicen o si se descarta uno de los 20 máximos sin `confirmado_por_usuario`. Un descarte afecta a todas las variables del registro y a toda la excursión del candidato. La biblioteca carga la serie con `datos.carga.cargar_serie`, que aplica `valido`.

### 8. Homogeneidad de la fuente

```bash
uv run homogeneidad-altimetros
uv run homogeneidad --procesos 4   # por defecto, los núcleos físicos; con puntos de control
```

### 9. Validación externa

```bash
uv run validacion-descarga todo              # con red: ndbc, waverys-boyas o waverys-regional; solo escribe en raw/
uv run validacion-modelo
uv run validacion-potencia --previo          # geometría, nulidad y cronometraje, sin rejilla
uv run validacion-potencia --procesos 8      # ≈ 7 h con 8 procesos (rejilla y validación); se reanuda si se corta
uv run validacion --previo                   # valores sintéticos en interim/, para medir el costo
uv run validacion --procesos 8               # ≈ 10 min; lee los valores observados
```

`validacion` falla sin leer ningún valor observado si la potencia no está publicada, si su hash no coincide con el manifiesto o si la huella del código de la potencia cambió. Los tiempos de cada paso se escriben en `interim/`, fuera de las salidas publicadas.

### 10. Clima de oleaje

```bash
uv run clima-etapa0 --previo                 # cronometra, no publica
uv run clima-etapa0 --procesos 6             # unos 2 min; factor de concordancia
uv run clima-simulacion-previa --previo
uv run clima-simulacion-previa --procesos 4  # unos 45 min; se reanuda si se corta y muestra el avance
uv run clima-etapa1 --previo                 # lee la serie; cronometra, no publica
uv run clima-etapa1 --procesos 6             # con datos del sitio, sin leer excesos
uv run clima --previo                        # verifica la etapa 1 y cronometra; no lee excesos
uv run clima --procesos 6                    # etapa principal; se reanuda si se corta y muestra el avance
```

`clima` lee, si existe, el registro de la revisión visual de las figuras de eventos (`interim/F06-clima-de-oleaje/revision-visual/<nombre>/revision_visual_declustering.csv`, con quién clasificó, el criterio y la razón de cada figura) y lo vuelca en `declustering_dudoso.csv`, rotulado «posterior a ver las figuras». Sin el registro, los candidatos salen con «revisión visual pendiente».

`clima-etapa0` y `clima-simulacion-previa` no leen datos del sitio. El avance va a `interim/F06-clima-de-oleaje/` y se borra al publicar.

### Cuaderno

`notebooks/01_mapa_seleccion_sitio.ipynb`: cambiar la celda de parámetros y ejecutar todo. Incluye el paso al mapa definitivo.

### Desde Python

```python
from peruvian_wave_analysis.sitio import ConfigMapa, generar_mapa

r = generar_mapa(ConfigMapa(nombre="marcona", latitud=-15.4, longitud=-75.3, isobatas=[200, 100, 50]))
r.tabla.head()
```

## Parámetros del mapa exploratorio

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `nombre` | — | Nombre corto del sitio (minúsculas); nombra la carpeta de salida. |
| `latitud`, `longitud` | — | Coordenada de referencia en grados (negativas al sur y al oeste). |
| `isobatas` | `[200, 100, 50]` | Profundidades de las isóbatas (m). |
| `isobata_principal` | la primera | Isóbata con la que se ordenan los candidatos. |
| `semiancho_grados` | `1.0` | Semiancho del recuadro alrededor de la referencia. |
| `n_candidatos` | `3` | Candidatos numerados como máximo. |
| `resguardo_km` | `1.0` | Distancia de resguardo alrededor de las zonas de exclusión. |
| `min_vecinas` | `0` | Mínimo de vecinas con dato para ser candidato (0 = no descarta). |
| `max_tierra` | `1.0` | Fracción de tierra tolerada en la celda (1 = no descarta). |
| `zonas` | `true` | `false` omite las zonas (el mapa lo declara). |
| `puntos` | `[]` | Puntos que solo se dibujan (nombre, latitud, longitud), en grados decimales o sexagesimales (`15°23′20″S`, `75°04′15″O`). No intervienen en los criterios. Opcionales: `etiqueta` (texto corto en el mapa; la leyenda lleva el nombre), `marcador` (símbolo de matplotlib, por defecto `D`) y `desplazamiento` de la etiqueta en puntos (`[7, -3]`). En la línea de comandos: `--punto NOMBRE LAT LON`, repetible. |
| `batimetria.fuente` | `gmrt` | `gmrt` (descarga) o `archivo` (NetCDF local). |
| `batimetria.resolucion_gmrt` | `med` | `low`, `med`, `high` o `max`. |
| `waverys.fuente` | `copernicus` | `copernicus` (descarga) o `archivo`. |
| `waverys.fecha` | `2020-07-15T12:00:00` | Paso de tiempo descargado para leer la malla y la máscara de tierra. |

## Criterios de selección

La elección final del nodo es visual. La herramienta solo descarta un nodo si:

1. no tiene dato en WAVERYS (es tierra para el modelo), o
2. su celda toca una zona de exclusión ampliada en el resguardo.

La fracción de tierra en la celda y las vecinas con dato se informan pero no descartan (salvo que se fijen `max_tierra` o `min_vecinas`). Las zonas de colisión de usos tampoco descartan: se listan por nombre. Los candidatos se ordenan por distancia a la isóbata principal.

| Zonas de exclusión | Zonas de colisión de usos |
|---|---|
| Parques nacionales, santuarios nacionales e históricos | Reservas nacionales y demás ANP de uso directo |
| Zonas de protección estricta (zonificación de ANP y ACR) | Áreas de conservación regional y privada |
| Zonas reservadas | Zonas de amortiguamiento |
| Pozos petroleros | Áreas de alto valor para la conservación marina |
| | Propuestas de ANP |
| | Lotes de hidrocarburos con contrato |

En el mapa, la malla de WAVERYS muestra el borde de las celdas que tienen algo de agua según la batimetría y los nodos con dato; las celdas enteramente en tierra y los nodos sin dato no se dibujan (siguen en `candidatos.csv`).

Una capa que no se puede descargar queda como **no evaluada** en el mapa y en los metadatos; nunca se trata como «sin zonas». No hay fuente en línea para zonas militares, tráfico marítimo, fondeaderos, cables, franja de 5 millas, caladeros ni acuicultura: el mapa las lista como no evaluadas.

## Salidas

En `WAVE_DATA_DIR`:

- `raw/`: insumos descargados, de solo lectura, registrados en `raw/MANIFEST.sha256` y `raw/MANIFEST.json` (hash, origen, parámetros, fecha de acceso). Un insumo existente se reutiliza tras verificar su hash; las zonas, solo si son del mismo día.
- `interim/F01-sitio-y-fuentes/mapa-sitio/<nombre>/` (exploratorio): `mapa.png`, `candidatos.csv`, `metadatos.json`, `siguiente_paso.md`.
- `processed/F01-sitio-y-fuentes/mapa-sitio-final/<nombre>/` (definitivo): `mapa.png` (contexto regional con el nodo elegido, sin candidatos), `detalle_nodo.png` (la celda del nodo y sus ocho vecinas con la batimetría a resolución completa, líneas de profundidad, profundidad y tierra de cada celda), `nodos.csv` (todos los nodos recalculados con GEBCO, sin numeración), `ficha_nodo.json` (nodo elegido y vecindario de 3 × 3) y `MANIFEST.json`.
- `raw/waverys/serie/`: la serie de WAVERYS, un NetCDF por año.
- `processed/F01-sitio-y-fuentes/serie-waverys/<nombre>/`: `serie_gate.json` (gate de la serie) y `MANIFEST.json`.
- `processed/F01-sitio-y-fuentes/ficha-sitio/<nombre>/`: `ficha_sitio.md`, `exposicion.csv` (por sector) y `exposicion_rayos.csv` (por rayo), `obstaculos.csv`, `observaciones.csv` y `cobertura_observaciones.csv` (por misión y por régimen), `observaciones_sensibilidad_periodo.csv` y `cobertura_sensibilidad_periodo.csv` (sensibilidad rotulada), `cronologia_waverys.csv`, `misiones_asimiladas_waverys.csv`, `resumen_ficha.json` (con el gate de la fase) y `MANIFEST.json`.
- `processed/F02-biblioteca-comun/verificacion/`: `resumen.json` (bandas y métricas por escenario), `replicas/<escenario>.csv`, `bandas.png` y `MANIFEST.json`.
- `processed/F03-datos/candidatos/<nombre>/`: `serie_decodificada.parquet`, `hs_celdas.parquet`, `marcas.parquet`, `candidatos.csv` (con excursión, salto y vecinas usadas), `excursiones_k_rev.csv` (todas las excursiones del umbral de revisión, con sensibilidades de vecinas), `maximos.csv`, `figuras/` (máximos, candidatos e histogramas de acumulaciones), `resumen_candidatos.json` (escala, k, falsas alarmas sintéticas con IC, potencia, topes, particiones ausentes, períodos bajo resolución) y `MANIFEST.json`.
- `processed/F03-datos/serie/<nombre>/`: `serie_nodo.parquet`, `vecinas_hs.parquet`, `cobertura.csv` (Hs y Tp, por año y mes), `cobertura_anual.csv`, `cobertura_mensual.csv`, `marcas_por_anio.csv`, `marcas_por_regla.csv`, `periodos_largos.json`, `resumen_f03.json` (con la lista de sensibilidad y sus dos variantes) y `MANIFEST.json`.

- `processed/F04-homogeneidad/altimetros/<nombre>/`: `exposicion_altimetros.parquet`, `pasadas.csv`, `misiones_locales.csv`, `resumen_altimetros.json` y `MANIFEST.json`.
- `processed/F04-homogeneidad/homogeneidad/<nombre>/`: `cronologia_f04.csv`, `tramos.csv`, `cuantiles_por_tramo.csv`, `indicadores_anuales.csv`, `potencia.csv`, `pruebas_cambio.json`, `prueba_altimetros.json`, `altimetros_por_mision_anio.csv`, `cobertura_altimetria.csv`, `figuras/`, `resumen_f04.json` (con los escenarios de Hs) y `MANIFEST.json`.
- `processed/F05-validacion-externa/modelo/<nombre>/`: `modelo_region.parquet`, `marcas_region.parquet`, `celdas_region.csv`, `modelo_boyas.parquet`, `celdas_boyas.csv`, `resumen_modelo.json` y `MANIFEST.json`.
- `processed/F05-validacion-externa/potencia/<nombre>/`: `colocaciones_geometria.parquet`, `qc_geometria.csv`, `escalon_modelo.csv`, `potencia.csv`, `potencia.json`, `efecto_minimo.csv`, `replicas_potencia.parquet` y `MANIFEST.json`.
- `processed/F05-validacion-externa/validacion/<nombre>/`: `colocaciones.parquet`, `qc_observaciones.csv`, `observaciones.csv`, `sigma_1hz.csv`, `representatividad.json`, `sesgo_por_cuantil.csv`, `pruebas_validacion.json`, `correccion.json`, `serie_corregida.parquet`, `periodos.csv`, `saltos_validados.csv`, `figuras/`, `resumen_f05.json` (escenarios, variantes, fila de la fuente y entregas) y `MANIFEST.json`.
- `processed/F06-clima-de-oleaje/etapa0-sintetica/`: `etapa0.json` (factor de concordancia, resumen por proceso y señal de variación de θ(u)), `replicas_etapa0.csv` y `MANIFEST.json`.
- `processed/F06-clima-de-oleaje/etapa1/<nombre>/`: `estratos_etapa1.json` (conteos por estrato y rama, bimodalidad, fronteras, potencia, α de familia, G7b; `excesos_leidos: false`) y `MANIFEST.json`.
- `processed/F06-clima-de-oleaje/<nombre>/`: salidas de §2 del contrato (`resumen_f06.json`, `estratos.json`, `dependencia.json`, `clima_mensual.csv`, `clase_fisica.parquet`, `dominancia.csv`, `picos_provisionales.csv`, `origen_mayores.csv`, `rachas.csv`, `declustering_dudoso.csv`, `tp_condicional.csv`, `peralte.csv`, `gev_bloques.csv`, `figuras/`) y `MANIFEST.json`.
- `processed/F06-clima-de-oleaje/simulacion-previa/`: `simulacion_previa.json` (fronteras con sus rejillas, potencia, sesgo y ancho de los modelos, barrido de umbral y mínimo del modelo con ξ común) y `MANIFEST.json`.

Cada salida publicada lleva un `MANIFEST.json` sin marcas de tiempo: insumos con su ruta relativa a `WAVE_DATA_DIR` y su hash, salidas con su hash, huella del código, semilla y criterios del gate. Un gate fallido borra la salida anterior y deja solo `GATE_FAILED.json`.

### Columnas de `candidatos.csv`

| Columna | Significado |
|---|---|
| `candidato` | Número en el mapa (1 a 3); vacío si el nodo no cumple o queda fuera de los 3 primeros. |
| `lat`, `lon` | Coordenadas exactas del nodo de WAVERYS. |
| `valido` | El nodo tiene dato (no es tierra en la máscara del modelo). |
| `prof_nodo_m` | Profundidad de la batimetría en el nodo. |
| `prof_min_celda_m`, `prof_media_celda_m` | Profundidad mínima y media (en agua) dentro de la celda. Mínima negativa = hay tierra. |
| `frac_tierra_celda` | Fracción de la celda que es tierra según la batimetría. |
| `vecinas_validas`, `vecinas_presentes` | De las ocho vecinas, cuántas tienen dato y cuántas se descargaron. |
| `excluido_por_zona`, `zonas_exclusion` | Si la celda toca una zona de exclusión ampliada, y cuáles. |
| `zonas_colision` | Zonas de colisión de usos que toca la celda. |
| `dist_exclusion_km` | Distancia del nodo a la zona de exclusión más cercana. |
| `dist_costa_km`, `dist_iso_<d>m_km` | Distancia a la costa y a cada isóbata. |
| `dist_referencia_km` | Distancia a la coordenada de referencia. |
| `zonas_evaluadas` | Si se evaluaron las zonas. |
| `motivo_descarte` | Primer criterio que el nodo no cumple; vacío si cumple todos. |
| `cumple_criterios` | Cumple los criterios. |

## Pruebas

```bash
uv run pytest
```

Las pruebas usan datos sintéticos en directorios temporales y **no acceden a la red**. Se comprueban con `uv run ruff check src tests` (ruff fijado en las dependencias de desarrollo).

## Fuentes de datos

- **WAVERYS**: Copernicus Marine Service, producto `GLOBAL_MULTIYEAR_WAV_001_032`, dataset `cmems_mod_glo_wav_my_0.2deg_PT3H-i`. DOI: 10.48670/moi-00022.
- **Batimetría**: GMRT, Global Multi-Resolution Topography (Ryan et al., 2009), para el mapa exploratorio; GEBCO para el definitivo.
- **Zonas**: SERNANP (geoservicios.sernanp.gob.pe) y PERUPETRO vía GEOCATMIN-INGEMMET (geocatmin.ingemmet.gob.pe).
- **Costa**: Natural Earth 1:10 m (`ne_10m_land`, `ne_10m_minor_islands`), versión 5.1.2.
- **Altimetría**: Ribal y Young (2019), altimetría calibrada multimisión, IMOS/AODN (thredds.aodn.org.au). doi 10.1038/s41597-019-0083-9.
- **Ciclones tropicales**: IBTrACS v04r01 (NOAA NCEI), cuenca del Pacífico Sur.
- **Boyas**: NOAA NDBC, archivos históricos `stdmet` de las estaciones 32302 (1986–1995) y 32012 (2007–2018).

Ver las condiciones de uso de cada servicio antes de redistribuir datos derivados. Este mapa es un filtro exploratorio para una tesis; no sustituye la consulta a las autoridades.
