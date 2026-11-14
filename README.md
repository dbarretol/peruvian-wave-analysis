# peruvian-wave-analysis

Análisis de oleaje extremo y contornos ambientales Hs–Tp a partir del reanálisis WAVERYS de Copernicus Marine.

## Estado

Primera herramienta: **mapa de selección de sitio**. Dada una coordenada de referencia, dibuja la batimetría, las isóbatas elegidas (por defecto 200, 100 y 50 m) y la malla de nodos de WAVERYS con el recuadro de cada celda, y arma una tabla de nodos candidatos.

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

### Línea de comandos

```bash
uv run mapa-sitio --nombre marcona --lat -15.4 --lon -75.3
uv run mapa-sitio --nombre marcona --lat -15.4 --lon -75.3 --isobatas 300 200 100 --semiancho 1.5
uv run mapa-sitio --config config/mapa_sitio.ejemplo.yaml
```

### Cuaderno

`notebooks/01_mapa_seleccion_sitio.ipynb`: cambiar la celda de parámetros y ejecutar todo.

### Desde Python

```python
from peruvian_wave_analysis.sitio import ConfigMapa, generar_mapa

r = generar_mapa(ConfigMapa(nombre="marcona", latitud=-15.4, longitud=-75.3, isobatas=[200, 100, 50]))
r.tabla.head()
```

## Parámetros

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `nombre` | — | Nombre corto del sitio (minúsculas); nombra la carpeta de salida. |
| `latitud`, `longitud` | — | Coordenada de referencia en grados (negativas al sur y al oeste). |
| `isobatas` | `[200, 100, 50]` | Profundidades de las isóbatas (m). |
| `isobata_principal` | la primera | Isóbata con la que se ordenan los candidatos. |
| `semiancho_grados` | `1.0` | Semiancho del recuadro alrededor de la referencia. |
| `n_candidatos` | `10` | Candidatos numerados en el mapa. |
| `batimetria.fuente` | `gmrt` | `gmrt` (descarga del GridServer de GMRT) o `archivo` (NetCDF local, p. ej. un recorte de GEBCO). |
| `batimetria.resolucion_gmrt` | `med` | `low`, `med`, `high` o `max`. |
| `waverys.fuente` | `copernicus` | `copernicus` (descarga con `copernicusmarine`) o `archivo`. |
| `waverys.fecha` | `2020-07-15T12:00:00` | Paso de tiempo descargado para leer la malla y la máscara de tierra. |

## Salidas

En `WAVE_DATA_DIR`:

- `raw/batimetria/gmrt/…nc`, `raw/waverys/malla/…nc`: insumos descargados, de solo lectura, registrados en `raw/MANIFEST.sha256` y `raw/MANIFEST.json` (hash, origen, parámetros, fecha de acceso). Si un insumo ya existe se reutiliza tras verificar su hash.
- `interim/F01-sitio-y-fuentes/mapa-sitio/<nombre>/`:
  - `mapa.png`: el mapa;
  - `candidatos.csv`: una fila por nodo del recuadro;
  - `metadatos.json`: configuración, versión del código e insumos con su hash.

### Columnas de `candidatos.csv`

| Columna | Significado |
|---|---|
| `candidato` | Número en el mapa; vacío si el nodo no cumple los criterios básicos. |
| `lat`, `lon` | Coordenadas exactas del nodo de WAVERYS. |
| `valido` | El nodo tiene dato (no es tierra en la máscara del modelo). |
| `prof_nodo_m` | Profundidad de la batimetría en el nodo. |
| `prof_min_celda_m`, `prof_media_celda_m` | Profundidad mínima y media (en agua) dentro de la celda de 0.2°. Mínima negativa = hay tierra. |
| `frac_tierra_celda` | Fracción de la celda que es tierra según la batimetría. |
| `vecinas_validas`, `vecinas_presentes` | De las ocho vecinas, cuántas tienen dato y cuántas se descargaron. |
| `dist_costa_km`, `dist_iso_<d>m_km` | Distancia a la costa y a cada isóbata. |
| `dist_referencia_km` | Distancia a la coordenada de referencia. |
| `cumple_criterios_basicos` | Nodo válido, celda sin tierra y ocho vecinas válidas. |

Los candidatos se ordenan por distancia a la isóbata principal. La herramienta **no evalúa** zonas de exclusión ni de colisión de usos: el mapa lo dice en su leyenda.

## Pruebas

```bash
uv run pytest
```

Las pruebas usan datos sintéticos en directorios temporales y no acceden a la red.

## Fuentes de datos

- **WAVERYS**: Copernicus Marine Service, producto `GLOBAL_MULTIYEAR_WAV_001_032`, dataset `cmems_mod_glo_wav_my_0.2deg_PT3H-i`. Doi: 10.48670/moi-00022.
- **Batimetría**: GMRT, Global Multi-Resolution Topography (Ryan et al., 2009), o GEBCO si se usa un archivo local.

Ver las condiciones de uso de cada servicio antes de redistribuir datos derivados.
