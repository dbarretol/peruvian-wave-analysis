# peruvian-wave-analysis

Análisis de oleaje extremo y contornos ambientales Hs–Tp a partir del reanálisis WAVERYS de Copernicus Marine.

## Estado

Primera herramienta: **mapa de selección de sitio**, en dos pasos.

1. **Mapa exploratorio** (`mapa-sitio`): dada una coordenada de referencia, descarga la batimetría (GMRT), la malla de nodos de WAVERYS y las zonas protegidas y de hidrocarburos, y dibuja las isóbatas elegidas (por defecto 200, 100 y 50 m), la malla con el recuadro de cada celda, las zonas de exclusión y de colisión de usos, y hasta 3 nodos candidatos.
2. **Mapa definitivo** (`mapa-sitio-final`): con el nodo ya elegido y un recorte de GEBCO descargado a mano, rehace el mapa **sin descargar nada**, verificando el hash de cada insumo, y comprueba que el nodo cumple los criterios.

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
- `processed/F01-sitio-y-fuentes/mapa-sitio-final/<nombre>/` (definitivo): `mapa.png`, `candidatos.csv`, `ficha_nodo.json`, `MANIFEST.json`.

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

Las pruebas usan datos sintéticos en directorios temporales y **no acceden a la red**.

## Fuentes de datos

- **WAVERYS**: Copernicus Marine Service, producto `GLOBAL_MULTIYEAR_WAV_001_032`, dataset `cmems_mod_glo_wav_my_0.2deg_PT3H-i`. DOI: 10.48670/moi-00022.
- **Batimetría**: GMRT, Global Multi-Resolution Topography (Ryan et al., 2009), para el mapa exploratorio; GEBCO para el definitivo.
- **Zonas**: SERNANP (geoservicios.sernanp.gob.pe) y PERUPETRO vía GEOCATMIN-INGEMMET (geocatmin.ingemmet.gob.pe).

Ver las condiciones de uso de cada servicio antes de redistribuir datos derivados. Este mapa es un filtro exploratorio para una tesis; no sustituye la consulta a las autoridades.
