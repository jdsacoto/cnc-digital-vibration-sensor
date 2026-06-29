"""Helpers de evaluación de modelos del proyecto "Sensor virtual de vibraciones CNC".

Funciones extraídas de ``models_v2.ipynb`` (Sub-paso 3.2 de Fase 3). Las
funciones que dependían de variables globales del notebook (``df_train``,
``df_test``, ``FEATURE_COLS``, ``TARGET_VARS``) ahora las reciben como
parámetros — esa es la única adaptación de firma. La lógica interna es
idéntica al notebook.

Los registries de modelos se exponen como funciones ``make_*_registry`` con
parámetro ``use_gpu=False`` por defecto. Con ``use_gpu=False`` el registry
es literalmente igual al del notebook (CPU). Con ``use_gpu=True`` XGBoost y
CatBoost activan GPU; LightGBM se queda en CPU porque su build de
conda-forge exige OpenCL no disponible en este sistema (ver
``GPU_SANITY_CHECK.md`` §3).

Notas sobre validación:

- :func:`evaluate_single` y :func:`evaluate_multi` NO se ejecutan con
  ``model.fit(...)`` en Sub-paso 3.2 (mapa §7.3 B5+B6). En Sub-paso 3.2
  solo se valida que la firma acepte los parámetros esperados vía
  :func:`inspect.signature`. La validación end-to-end real ocurre en
  Sub-paso 3.3 mediante smoke test de ``models_v2.ipynb`` adaptado a
  imports.
- :func:`make_single_registry` y :func:`make_multi_registry` solo se
  validan en estructura (claves, ``factory`` callable, ``scale`` bool) sin
  invocar ``cfg['factory']()`` (eso instanciaría modelos, fuera de
  Sub-paso 3.2).
"""
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from .config import (
    TARGET_VARS,
    RANDOM_STATE,
    SPINDLE_LOAD_THRESHOLD,
    LOAD_CONDITIONS,
)


def filter_load(df, kind, threshold=SPINDLE_LOAD_THRESHOLD):
    """Filtrar un DataFrame por condición de carga del husillo.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columna ``SPINDLE_LOAD``.
    kind : {'all', 'no_load', 'load'}
        Condición de carga:

        - ``'all'``: no filtra.
        - ``'no_load'``: conserva filas con ``SPINDLE_LOAD <= threshold``.
        - ``'load'``: conserva filas con ``SPINDLE_LOAD > threshold``.
    threshold : float, default ``SPINDLE_LOAD_THRESHOLD``
        Umbral de carga del husillo. El default viene de :mod:`src.config`.

    Returns
    -------
    pandas.DataFrame
        Subconjunto filtrado de ``df``.

    Raises
    ------
    ValueError
        Si ``kind`` no es uno de los valores admitidos.

    Notes
    -----
    Adaptación ligera de ``models_v2.ipynb`` cell 7 (Sub-paso 3.2):
    ``threshold`` se expone como parámetro (antes era global del notebook).
    """
    if kind == 'all':
        return df
    if kind == 'no_load':
        return df[df['SPINDLE_LOAD'] <= threshold]
    if kind == 'load':
        return df[df['SPINDLE_LOAD'] > threshold]
    raise ValueError(f'kind desconocido: {kind}')


def evaluate_single(model_factory, model_name, target, load_kind,
                    df_train, df_test, feature_cols, scale=False):
    """Entrenar y evaluar un modelo single-output. Devuelve dict de métricas.

    Parameters
    ----------
    model_factory : callable
        Llamable sin argumentos que devuelve una instancia nueva del modelo.
    model_name : str
        Nombre del modelo (para el dict de retorno).
    target : str
        Nombre de la columna objetivo.
    load_kind : {'all', 'no_load', 'load'}
        Condición de carga del husillo, ver :func:`filter_load`.
    df_train, df_test : pandas.DataFrame
        Datos de entrenamiento y test.
    feature_cols : list[str]
        Columnas a usar como features (sin incluir targets ni ``DATE``).
    scale : bool, default False
        Si ``True``, ajusta un :class:`~sklearn.preprocessing.StandardScaler`
        solo sobre train y lo aplica a test. Usar con modelos sensibles a
        escala (MLPRegressor); innecesario con árboles.

    Returns
    -------
    dict
        Métricas y metadatos del experimento: ``model``, ``output``,
        ``target``, ``load``, ``n_train``, ``n_test``, ``MAE``, ``MSE``,
        ``RMSE``, ``R2``, ``fit_s``.

    Notes
    -----
    Adaptación obligatoria de firma de ``models_v2.ipynb`` cell 7
    (Sub-paso 3.2): ``df_train``, ``df_test``, ``feature_cols`` ahora son
    parámetros (antes eran globales del notebook). Justificación en
    ``PHASE_3_REFACTOR_PLAN.md`` §4 y ``PHASE_3_2_EXTRACTION_MAP.md`` §6.
    """
    tr = filter_load(df_train, load_kind)
    te = filter_load(df_test, load_kind)

    X_tr, y_tr = tr[feature_cols].values, tr[target].values
    X_te, y_te = te[feature_cols].values, te[target].values

    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    t0 = time.time()
    model = model_factory()
    model.fit(X_tr, y_tr)
    fit_secs = time.time() - t0
    y_pred = model.predict(X_te)

    return {
        'model': model_name,
        'output': 'single',
        'target': target,
        'load': load_kind,
        'n_train': len(tr),
        'n_test': len(te),
        'MAE': mean_absolute_error(y_te, y_pred),
        'MSE': mean_squared_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred)),
        'R2': r2_score(y_te, y_pred),
        'fit_s': fit_secs,
    }


def evaluate_multi(model_factory, model_name, load_kind,
                   df_train, df_test, feature_cols, target_vars, scale=False):
    """Entrenar y evaluar un modelo multi-output. Devuelve lista de dicts por target.

    Parameters
    ----------
    model_factory : callable
        Llamable sin argumentos que devuelve una instancia nueva del modelo
        multi-output.
    model_name : str
        Nombre del modelo (para los dicts de retorno).
    load_kind : {'all', 'no_load', 'load'}
        Condición de carga del husillo.
    df_train, df_test : pandas.DataFrame
        Datos de entrenamiento y test.
    feature_cols : list[str]
        Columnas a usar como features.
    target_vars : list[str]
        Lista de columnas objetivo (típicamente las tres ``ACCEL_*``).
    scale : bool, default False
        Si ``True``, escalado fit-on-train análogo a :func:`evaluate_single`.

    Returns
    -------
    list[dict]
        Una entrada por target con las mismas claves que
        :func:`evaluate_single`, más ``output='multi'``.

    Notes
    -----
    Adaptación obligatoria de firma de ``models_v2.ipynb`` cell 7
    (Sub-paso 3.2): ``df_train``, ``df_test``, ``feature_cols``,
    ``target_vars`` ahora son parámetros. Lógica interna idéntica al
    notebook.
    """
    tr = filter_load(df_train, load_kind)
    te = filter_load(df_test, load_kind)

    X_tr, Y_tr = tr[feature_cols].values, tr[target_vars].values
    X_te, Y_te = te[feature_cols].values, te[target_vars].values

    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    t0 = time.time()
    model = model_factory()
    model.fit(X_tr, Y_tr)
    fit_secs = time.time() - t0
    Y_pred = model.predict(X_te)

    rows = []
    for i, t in enumerate(target_vars):
        rows.append({
            'model': model_name,
            'output': 'multi',
            'target': t,
            'load': load_kind,
            'n_train': len(tr),
            'n_test': len(te),
            'MAE': mean_absolute_error(Y_te[:, i], Y_pred[:, i]),
            'MSE': mean_squared_error(Y_te[:, i], Y_pred[:, i]),
            'RMSE': np.sqrt(mean_squared_error(Y_te[:, i], Y_pred[:, i])),
            'R2': r2_score(Y_te[:, i], Y_pred[:, i]),
            'fit_s': fit_secs,
        })
    return rows


def make_single_registry(use_gpu=False, random_state=RANDOM_STATE):
    """Construir el registro de modelos single-output.

    Con ``use_gpu=False`` (default) el registry es literalmente igual al
    definido en ``models_v2.ipynb`` cell 9. Con ``use_gpu=True`` XGBoost
    activa ``device='cuda'`` y CatBoost activa ``task_type='GPU'``.
    LightGBM se queda en CPU en ambos casos por falta de OpenCL en el
    entorno actual (ver ``GPU_SANITY_CHECK.md`` §3).

    Parameters
    ----------
    use_gpu : bool, default False
        Si ``True``, activa GPU en los modelos que la soportan en este
        entorno. **El default es ``False`` para preservar reproducibilidad
        bit-a-bit del baseline CPU** (regla operativa de Fase 3).
    random_state : int, default ``RANDOM_STATE``
        Semilla para los modelos.

    Returns
    -------
    dict[str, dict]
        Registro ``{model_name: {'factory': callable, 'scale': bool}}`` con
        los cinco modelos del baseline (RandomForest, XGBoost, LightGBM,
        CatBoost, MLPRegressor).

    Notes
    -----
    Adaptación de literal a factory de ``models_v2.ipynb`` cell 9
    (Sub-paso 3.2).
    """
    if use_gpu:
        xgb_factory = lambda: xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            n_jobs=-1, random_state=random_state, verbosity=0,
            device='cuda', tree_method='hist',
        )
        catboost_factory = lambda: CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=8,
            random_state=random_state, verbose=0,
            allow_writing_files=False,
            task_type='GPU', devices='0',
        )
    else:
        xgb_factory = lambda: xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            n_jobs=-1, random_state=random_state, verbosity=0,
        )
        catboost_factory = lambda: CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=8,
            random_state=random_state, verbose=0,
            allow_writing_files=False,
        )

    return {
        'RandomForest': {
            'factory': lambda: RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_leaf=2,
                n_jobs=-1, random_state=random_state),
            'scale': False,
        },
        'XGBoost': {
            'factory': xgb_factory,
            'scale': False,
        },
        'LightGBM': {
            'factory': lambda: lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                n_jobs=-1, random_state=random_state, verbosity=-1),
            'scale': False,
        },
        'CatBoost': {
            'factory': catboost_factory,
            'scale': False,
        },
        'MLPRegressor': {
            'factory': lambda: MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True,
                random_state=random_state),
            'scale': True,
        },
    }


def make_multi_registry(use_gpu=False, random_state=RANDOM_STATE):
    """Construir el registro de modelos multi-output.

    RandomForest y MLPRegressor soportan multi-output nativamente; XGBoost,
    LightGBM y CatBoost se envuelven en
    :class:`~sklearn.multioutput.MultiOutputRegressor`.

    Parameters
    ----------
    use_gpu : bool, default False
        Igual que :func:`make_single_registry`.
    random_state : int, default ``RANDOM_STATE``
        Semilla para los modelos.

    Returns
    -------
    dict[str, dict]
        Registro análogo a :func:`make_single_registry` pero para
        multi-output.

    Notes
    -----
    Adaptación de literal a factory de ``models_v2.ipynb`` cell 9
    (Sub-paso 3.2). Los factories multi-output instancian el modelo single
    subyacente y lo envuelven si hace falta — exactamente como el notebook.
    """
    single = make_single_registry(use_gpu=use_gpu, random_state=random_state)
    return {
        'RandomForest':  {'factory': single['RandomForest']['factory'], 'scale': False},
        'XGBoost':       {'factory': lambda: MultiOutputRegressor(single['XGBoost']['factory']()),  'scale': False},
        'LightGBM':      {'factory': lambda: MultiOutputRegressor(single['LightGBM']['factory']()), 'scale': False},
        'CatBoost':      {'factory': lambda: MultiOutputRegressor(single['CatBoost']['factory']()), 'scale': False},
        'MLPRegressor':  {'factory': single['MLPRegressor']['factory'], 'scale': True},
    }
