# Sensor virtual de vibraciones para máquina CNC

Proyecto de aprendizaje automático que **infiere variables agregadas de vibración** (`ACCEL_PEAK`, `ACCEL_RMS`, `ACCEL_RMS_FREQ`) a partir de las **señales internas de una máquina CNC**, mediante **regresión supervisada tabular**. El objetivo es estimar el estado vibratorio aprovechando la sensórica que el autómata de la máquina ya produce, sin instrumentación física adicional.

El dataset base son muestras a 5 Hz registradas en junio de 2020 sobre la herramienta de identificador `1018`. El pipeline reproducible evalúa cinco familias de modelos (RandomForest, XGBoost, LightGBM, CatBoost, MLPRegressor) en variantes single-output y multi-output, sobre tres condiciones de carga del husillo (`all`, `load`, `no_load`).

## Resultados clave

- Mejor R² sobre el conjunto de prueba: **0,8547** (LightGBM, `ACCEL_RMS`, condición `all`, single-output).
- 90 combinaciones evaluadas (5 modelos × 3 targets × 3 condiciones × 2 tipos de salida).
- Tabla maestra de métricas: [`reports/results/01_models_v2_results.csv`](reports/results/01_models_v2_results.csv).
- Mejor modelo por (target × condición): [`reports/results/02_best_by_target_load.csv`](reports/results/02_best_by_target_load.csv).
- Top-10 global por R²: [`reports/results/03_top10_r2.csv`](reports/results/03_top10_r2.csv).
- Pivote de R² por modelo × (target, carga, salida): [`reports/results/04_pivot_r2.csv`](reports/results/04_pivot_r2.csv).
- Figuras de resultados: 6 PNG en [`reports/figures/`](reports/figures/).

## Requisitos

- Python **3.11**.
- Linux probado (Ubuntu/derivados); Windows compatible con la misma `environment.yml`.
- ~3 GB de disco para el entorno Conda.
- Dataset fuente `Data/upload_5hz_2020_6.h5` (no incluido; ver sección «Datos»).

## Instalación

### Opción A — Conda (recomendada)

```bash
conda env create -f environment.yml
conda activate cnc-vsensor
```

El entorno está fijado a `python=3.11` y depende únicamente del canal `conda-forge`; produce el baseline reproducible bit a bit.

### Opción B — pip (alternativa best-effort)

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lista las dependencias directas alineadas con `environment.yml`. **No garantiza paridad bit a bit** (los wheels de PyPI pueden diferir de los binarios de `conda-forge`): es válido para verificar el pipeline; para reproducir métricas exactas, usar Conda.

## Estructura del repositorio

```
.
├── src/                          módulos Python reutilizables
│   ├── config.py                 constantes del pipeline
│   ├── preprocessing.py          helpers del pipeline de datos
│   └── evaluation.py             registry de modelos + métricas
├── notebooks/                    pipeline reproducible
│   ├── read_data_v2.ipynb        Data/*.h5 → df_train.csv, df_test.csv
│   ├── models_v2.ipynb           df_*.csv → métricas de los modelos
│   ├── results_figures.ipynb     → reports/figures/*, reports/results/*
│   └── legacy/                   notebooks previos al refactor (congelados)
├── reports/
│   ├── figures/                  6 PNG de resultados
│   └── results/                  5 CSV de métricas
├── docs/figures/                 diagramas conceptuales del pipeline
├── environment.yml               entorno Conda
├── requirements.txt              entorno pip alternativo
└── filter_metadata.json          parámetros del filtrado (artefacto del pipeline)
```

El dataset fuente y los artefactos intermedios (`df_train.csv`, `df_test.csv`, etc.) se excluyen del control de versiones (`.gitignore`).

## Ejecutar el pipeline

Con el entorno `cnc-vsensor` activo, desde la raíz del repositorio:

```bash
# 1. Pipeline de datos (genera df_train.csv, df_test.csv, filter_metadata.json)
jupyter nbconvert --execute notebooks/read_data_v2.ipynb \
    --to notebook --output /tmp/read_data_v2_run.ipynb \
    --ExecutePreprocessor.timeout=600

# 2. Modelos + métricas
jupyter nbconvert --execute notebooks/models_v2.ipynb \
    --to notebook --output /tmp/models_v2_run.ipynb \
    --ExecutePreprocessor.timeout=1800

# 3. Figuras y tablas finales
jupyter nbconvert --execute notebooks/results_figures.ipynb \
    --to notebook --output /tmp/results_figures_run.ipynb \
    --ExecutePreprocessor.timeout=300
```

Los notebooks resuelven `REPO_ROOT` automáticamente y se ejecutan desde la raíz o desde `notebooks/`. Se usa `--output /tmp/...` (nunca `--inplace`) para no modificar los notebooks versionados.

## Reproducibilidad

- `df_train.csv` y `df_test.csv` se regeneran de forma determinista al ejecutar `notebooks/read_data_v2.ipynb`.
- Las métricas de calidad (MAE/MSE/RMSE/R²) se reproducen fila a fila entre plataformas; la columna `fit_s` depende del hardware y se acepta como variable.
- `filter_metadata.json` puede variar cosméticamente en el orden de `correlation_dropped` (no determinismo del orden de iteración de `set` en Python); el contenido lógico es idéntico.

## Datos

El dataset fuente `Data/upload_5hz_2020_6.h5` **no está en el repositorio** (excluido por `.gitignore`) por motivos de confidencialidad de la organización que lo aportó. Se asume disponible localmente antes de ejecutar el pipeline; sin él, `notebooks/read_data_v2.ipynb` no puede cargar los datos.

## CPU vs GPU

- **Baseline declarado: CPU.** En `notebooks/models_v2.ipynb` los registries se construyen con `use_gpu=False`. Es el flujo soportado y reproducible.
- **GPU opcional.** XGBoost y CatBoost pueden activar aceleración GPU; LightGBM-GPU requiere OpenCL. Cambiar el backend puede alterar marginalmente las métricas y rompería la paridad bit a bit, por lo que la GPU se trata como experimento separado.

## Notebooks legacy

`notebooks/legacy/` conserva cinco notebooks previos al refactor como **evidencia histórica**. No son ejecutables desde su ubicación actual (sus rutas internas apuntan a la estructura del proyecto original) y contienen defectos metodológicos documentados (split aleatorio sobre serie temporal, transformaciones pre-split, leakage en el escalado). El pipeline reproducible es el de `notebooks/`. Más contexto en [`notebooks/legacy/README.md`](notebooks/legacy/README.md).

## Notas metodológicas

- El pipeline usa **partición temporal por días**: las últimas N fechas del dataset filtrado se asignan a prueba, sin solapamiento con entrenamiento.
- Las transformaciones que pueden generar fuga de información (umbrales IQR de outliers, One-Hot encoding, poda por correlación) se **ajustan solo sobre entrenamiento** y se aplican luego a prueba.
- El filtro IQR sobre los targets `ACCEL_*` es una decisión sensible: descarta filas con valores extremos antes de modelar, lo que acota el rango sobre el que se mide la calidad del sensor virtual.
- Las métricas reportadas corresponden al **dataset filtrado**, no a la señal cruda completa; cualquier comparación con la literatura debe tener este recorte en cuenta.

## Licencia y uso

Proyecto de investigación aplicada con fines educativos. El dataset fuente no se redistribuye.
