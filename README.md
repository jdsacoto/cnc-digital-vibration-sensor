# Sensor virtual de vibraciones para máquina CNC

Repositorio de código y resultados del Trabajo de Fin de Máster
**«Desarrollo de un sensor virtual de vibraciones basado en señales internas de una máquina CNC»**.

- **Autor:** José Daniel Sacoto Peralta
- **Universidad:** Universidad Internacional de La Rioja (UNIR)
- **Programa:** Máster Universitario en Inteligencia Artificial
- **Escuela:** Escuela Superior de Ingeniería y Tecnología
- **Directora académica:** PhD. Carriba Perez Amaya

---

Proyecto de aprendizaje automático que **infiere variables agregadas de vibración** (`ACCEL_PEAK`, `ACCEL_RMS`, `ACCEL_RMS_FREQ`) a partir de las **señales internas de una máquina CNC**, mediante **regresión supervisada tabular**. El objetivo es estimar el estado vibratorio aprovechando la sensórica que el autómata de la máquina ya produce, sin instrumentación física adicional.

El dataset base son muestras a 5 Hz registradas en junio de 2020 sobre la herramienta de identificador `1018`. El pipeline reproducible evalúa cinco familias de modelos (RandomForest, XGBoost, LightGBM, CatBoost, MLPRegressor) en variantes single-output y multi-output, sobre tres condiciones de carga del husillo (`all`, `load`, `no_load`).

## Resultados clave

El escenario principal del trabajo es **`cnc_available_all`**: las 28 columnas disponibles del
autómata (todas menos `DATE` y los tres targets) como entrada. Se evalúa sobre **dos** conjuntos
de prueba: `filtered_test` (in-distribution) y `raw_target_test` (test temporal conservando los
picos de vibración).

- Mejor R² in-distribution: **0,8547** (LightGBM, `ACCEL_RMS`, condición `all`, single-output).
- Conservando los picos: **0,8214** (XGBoost) y **0,8204** (LightGBM) sobre el mismo target.
- **180 filas** de métricas: 90 combinaciones base (5 modelos × 3 targets × 3 condiciones ×
  2 estrategias de salida) × 2 evaluaciones.

### Registro principal (v3, escenario `cnc_available_all`)

| Artefacto | Ruta | Contenido |
|---|---|---|
| Registro maestro | [`reports/results/01_models_v3_cnc_available_all_results.csv`](reports/results/01_models_v3_cnc_available_all_results.csv) | 180 filas × 17 columnas |
| Mejor modelo por target × carga | [`reports/results/02_models_v3_cnc_available_all_best_by_target_load.csv`](reports/results/02_models_v3_cnc_available_all_best_by_target_load.csv) | |
| Top-10 en `filtered_test` | [`reports/results/03_models_v3_cnc_available_all_top10_filtered.csv`](reports/results/03_models_v3_cnc_available_all_top10_filtered.csv) | |
| Top-10 en `raw_target_test` | [`reports/results/04_models_v3_cnc_available_all_top10_raw_target.csv`](reports/results/04_models_v3_cnc_available_all_top10_raw_target.csv) | |
| Pivote de R² | [`reports/results/05_models_v3_cnc_available_all_r2_pivot.csv`](reports/results/05_models_v3_cnc_available_all_r2_pivot.csv) | |
| Comparación filtered vs raw | [`reports/results/06_models_v3_cnc_available_all_filtered_vs_raw_comparison.csv`](reports/results/06_models_v3_cnc_available_all_filtered_vs_raw_comparison.csv) | |
| Análisis de sensibilidad | [`reports/results/ablation_feature_sensitivity.csv`](reports/results/ablation_feature_sensitivity.csv) | escenarios A / B / D |
| Índice de figuras v3 | [`reports/figures/figures_v3_cnc_available_all_index.md`](reports/figures/figures_v3_cnc_available_all_index.md) | fuente e interpretación de cada figura |

### Registro antecedente (v2, solo in-distribution)

Se conserva como antecedente reproducible: 90 filas en
[`reports/results/01_models_v2_results.csv`](reports/results/01_models_v2_results.csv), con sus
tablas derivadas `02`–`05` y las 6 figuras `01`–`06` de [`reports/figures/`](reports/figures/).
Sus métricas coinciden con las filas `filtered_test` del registro v3.

En total: 14 CSV y 4 resúmenes en `reports/results/`, y 13 PNG más el índice en `reports/figures/`.

## Requisitos

- Python **3.11**.
- Linux probado (Ubuntu/derivados); Windows compatible con la misma `environment.yml`.
- ~3 GB de disco para el entorno Conda.
- Dataset fuente `Data/upload_5hz_2020_6.h5` (no incluido; ver sección «Datos»).

## Instalación

### Opción A — Conda (recomendada)

```bash
conda env create -f environment.yml
conda activate tfm-cnc
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
│   ├── evaluation.py             registry de modelos + métricas
│   └── experiments/              scripts del registro v3
│       ├── models_v3_cnc_available_all.py   genera el registro maestro v3
│       ├── evaluate_raw_target_test.py      evaluación con picos conservados
│       ├── generate_figures_v3.py           genera las figuras v3
│       └── ablation_feature_leakage.py      corrida base del análisis de sensibilidad
├── notebooks/                    pipeline reproducible
│   ├── read_data_v2.ipynb        Data/*.h5 → df_train.csv, df_test.csv
│   ├── models_v2.ipynb           df_*.csv → métricas de los modelos
│   ├── results_figures.ipynb     → reports/figures/*, reports/results/*
│   ├── experiments/              experimentos complementarios (ejecutados, con salidas)
│   │   ├── neural_mlp_v5_executed.ipynb   red densa sobre ACCEL_RMS
│   │   └── rnn_v1_executed.ipynb          modelos secuenciales LSTM/GRU
│   └── legacy/                   notebooks previos al refactor (congelados)
├── reports/
│   ├── figures/                  13 PNG (6 de v2, 7 de v3) + índice de figuras v3
│   └── results/                  14 CSV + 4 resúmenes en Markdown
├── environment.yml               entorno Conda
├── requirements.txt              entorno pip alternativo
└── filter_metadata.json          parámetros del filtrado (artefacto del pipeline)
```

El dataset fuente y los artefactos intermedios (`df_train.csv`, `df_test.csv`, etc.) se excluyen del control de versiones (`.gitignore`).

## Ejecutar el pipeline

Con el entorno `tfm-cnc` activo, desde la raíz del repositorio:

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

### Registro v3 (escenario principal)

Los notebooks anteriores producen el registro antecedente v2. El registro principal del trabajo
—180 filas sobre `filtered_test` y `raw_target_test`— se regenera con los scripts de
`src/experiments/`, en este orden y desde la raíz del repositorio:

```bash
python -m src.experiments.models_v3_cnc_available_all   # registro maestro v3 (180 filas)
python -m src.experiments.ablation_feature_leakage      # corrida base de sensibilidad
python -m src.experiments.generate_figures_v3           # figuras v3 (solo lee CSV)
```

`generate_figures_v3.py` no reentrena: únicamente lee los CSV ya generados.

> **Nota sobre la nomenclatura de los archivos de ablación.** Durante el desarrollo, la
> contribución de `SPEED_RMS` y `TEMPERATURE` se investigó bajo la hipótesis de una posible fuga
> de información, y por eso el script y sus salidas conservan el nombre histórico
> `ablation_feature_leakage`. Esa hipótesis **no se confirmó**: el diccionario de variables es
> ambiguo sobre el origen de ambas señales, pero no demuestra que deriven del sensor de
> vibraciones. En consecuencia se reencuadró como **análisis de sensibilidad**, y los archivos
> `ablation_feature_sensitivity.*` recogen esa lectura corregida sobre los mismos datos, sin
> reentrenar. Ambos conjuntos se publican para dejar trazable la revisión.

### Experimentos complementarios con redes neuronales

`notebooks/experiments/` contiene dos notebooks ya ejecutados, con sus salidas y figuras
guardadas, que contrastan el bloque de *boosting* frente a familias de modelos distintas:

- `neural_mlp_v5_executed.ipynb` — red densa de mayor capacidad sobre `ACCEL_RMS`.
- `rnn_v1_executed.ipynb` — modelos secuenciales con celdas LSTM y GRU.

En ambos casos las alternativas neuronales **no superan** al bloque LightGBM/XGBoost sobre las
variables disponibles. Se conservan como material complementario y línea de trabajo futuro.

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
