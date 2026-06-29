"""Constantes metodológicas compartidas del pipeline del proyecto.

Fuente única de verdad para los valores que aparecen tanto en
``read_data_v2.ipynb`` como en ``models_v2.ipynb``. Importadas por
:mod:`src.preprocessing` y :mod:`src.evaluation`.

**Las rutas no viven aquí.** ``H5_PATH``, ``DATA_DIR``, ``OUT_DIR`` y
similares dependen del lugar desde donde se ejecuta cada notebook y se
mantienen literales en cada notebook. Decisión documentada en
``PHASE_3_2_EXTRACTION_MAP.md`` §1.
"""

# --- Targets del sensor virtual -------------------------------------------
TARGET_VARS = ['ACCEL_PEAK', 'ACCEL_RMS', 'ACCEL_RMS_FREQ']

# --- Semilla global -------------------------------------------------------
RANDOM_STATE = 42

# --- Selección de herramienta (CNC tool ID) -------------------------------
TOOL_VALIDATION = 1018

# --- Split temporal -------------------------------------------------------
N_TEST_DAYS = 4

# --- Filtros pre-split ----------------------------------------------------
HIGH_CARDINALITY_THRESHOLD = 15
CORR_THRESHOLD = 0.95

# --- Estrategias de outliers por columna ----------------------------------
# Cada entrada define método y parámetros. La aplicación efectiva la hacen
# ``src.preprocessing.fit_outlier_thresholds`` y
# ``src.preprocessing.apply_outlier_thresholds``.
OUTLIER_STRATEGIES = {
    'ACCEL_PEAK':           {'method': 'iqr',       'factor': 4.0},
    'ACCEL_RMS':            {'method': 'iqr',       'factor': 4.0},
    'ACCEL_RMS_FREQ':       {'method': 'iqr',       'factor': 3.0},
    'B':                    {'method': 'domain',    'min_val': 0,   'max_val': 360},
    'CURR_B':               {'method': 'winsorize', 'lower_p': 0.005,  'upper_p': 0.995},
    'CURR_X':               {'method': 'winsorize', 'lower_p': 0.005,  'upper_p': 0.995},
    'CURR_Y':               {'method': 'winsorize', 'lower_p': 0.005,  'upper_p': 0.995},
    'CURR_Z':               {'method': 'winsorize', 'lower_p': 0.005,  'upper_p': 0.995},
    'SPEED_RMS':            {'method': 'winsorize', 'lower_p': 0.01,   'upper_p': 0.99},
    'SPINDLE_ACTUAL_FEED':  {'method': 'percentile','lower_p': 0.005,  'upper_p': 0.995},
    'SPINDLE_ACTUAL_SPEED': {'method': 'domain',    'min_val': 0,   'max_val': 15000},
    'SPINDLE_LOAD':         {'method': 'winsorize', 'lower_p': 0.0025, 'upper_p': 0.9975},
    'TEMPERATURE':          {'method': 'percentile','lower_p': 0.01,   'upper_p': 0.99},
    'X':                    {'method': 'winsorize', 'lower_p': 0.0025, 'upper_p': 0.9975},
    'Y':                    {'method': 'winsorize', 'lower_p': 0.0025, 'upper_p': 0.9975},
    'Z':                    {'method': 'winsorize', 'lower_p': 0.0025, 'upper_p': 0.9975},
}

# --- Variables categóricas a OneHotEncoding -------------------------------
KNOWN_CATEGORICALS = ['JOG_OVERRIDE', 'SPINDLE_OVERRIDE_X105', 'X105']

# --- Umbral SPINDLE_LOAD para "con carga" vs "sin carga" ------------------
SPINDLE_LOAD_THRESHOLD = 5.0

# --- Condiciones de carga a iterar en experimentos ------------------------
LOAD_CONDITIONS = ['all', 'no_load', 'load']
