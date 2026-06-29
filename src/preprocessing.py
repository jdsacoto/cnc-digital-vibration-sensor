"""Helpers de preprocesado del proyecto "Sensor virtual de vibraciones CNC".

Funciones extraídas de ``read_data_v2.ipynb`` (Sub-paso 3.2 de Fase 3). La
lógica es idéntica a la del notebook: copy-paste + docstring, sin
optimizaciones ni cambios de comportamiento. Las únicas adaptaciones
documentadas (firmas que reciben DataFrames por parámetro, reorganización
de bloques procedurales en funciones cohesivas) están listadas en
``PHASE_3_2_EXTRACTION_MAP.md`` §6.

Notas sobre validación:

- :func:`load_h5_with_date` NO se valida en Sub-paso 3.2 con el H5 real.
  Su validación real ocurre en Sub-paso 3.3 mediante smoke test de
  ``read_data_v2.ipynb`` adaptado a imports (mapa §7.2 A3).
- Las demás funciones se validan con DataFrames sintéticos pequeños en
  el proyecto, sin tocar ``df_train.csv``/``df_test.csv``/H5 (mapa §7.4).
"""
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import OneHotEncoder


def load_h5_with_date(h5_path):
    """Cargar archivo HDF5 jerárquico y construir DataFrame con columna ``DATE``.

    Lee todos los subgrupos (uno por día) bajo cada sensor del HDF5,
    construye un DataFrame por día con las muestras del sensor, le añade
    una columna ``DATE`` (``YYYY-MM-DD``) derivada del timestamp Unix en
    milisegundos del nombre del subgrupo, y concatena todos los días.

    Parameters
    ----------
    h5_path : str | pathlib.Path
        Ruta al archivo HDF5 con la estructura
        ``{sensor}/{timestamp_ms}/values``.

    Returns
    -------
    pandas.DataFrame
        DataFrame con una fila por muestra, una columna por sensor, y la
        columna ``DATE`` añadida al final.

    Notes
    -----
    Copy literal de ``read_data_v2.ipynb`` cell 4 (Sub-paso 3.2).
    """
    with h5py.File(h5_path, 'r') as f:
        sensors = list(f.keys())
        # Reunir todos los subgrupos
        subgroups = set()
        for s in sensors:
            subgroups.update(f[s].keys())
        subgroups = sorted(subgroups)
        print(f'Sensores: {len(sensors)}; subgrupos (días) detectados: {len(subgroups)}')

        per_day = []
        for sg in subgroups:
            data = {}
            for s in sensors:
                if sg in f[s] and 'values' in f[s][sg]:
                    data[s] = f[s][sg]['values'][...]
            if not data:
                continue
            df_day = pd.DataFrame(data)
            # Convertir timestamp ms -> fecha
            ts = int(sg) / 1000
            date_str = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime('%Y-%m-%d')
            df_day['DATE'] = date_str
            per_day.append(df_day)

        return pd.concat(per_day, ignore_index=True)


def remove_consecutive_duplicates(df):
    """Eliminar filas consecutivas idénticas a la previa.

    Reduce los periodos de máquina parada / idle donde varias filas
    seguidas repiten los mismos valores en todas las columnas.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con índice numérico secuencial.

    Returns
    -------
    pandas.DataFrame
        DataFrame con índice reseteado, conservando la primera fila de
        cada secuencia de duplicados.

    Notes
    -----
    Copy literal de ``read_data_v2.ipynb`` cell 8 (Sub-paso 3.2).
    """
    # Comparar cada fila con la previa; conservar solo si difiere en al menos
    # una columna.
    return df.loc[(df.shift() != df).any(axis=1)].reset_index(drop=True)


def split_by_date(df, n_test_days, date_col='DATE'):
    """Split temporal: las últimas ``n_test_days`` fechas van a test.

    Toma todas las fechas únicas del DataFrame, las ordena ascendentemente
    y asigna las últimas ``n_test_days`` al conjunto de test. Garantiza
    que ``df_train[date_col].max() < df_test[date_col].min()`` (sin overlap
    temporal).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con una columna de fechas (string ``YYYY-MM-DD`` por
        defecto).
    n_test_days : int
        Número de fechas (no de filas) a reservar para test.
    date_col : str, default 'DATE'
        Nombre de la columna que contiene las fechas.

    Returns
    -------
    tuple
        ``(df_train, df_test, train_dates, test_dates)`` con ambos
        DataFrames con índice reseteado y las listas de fechas asignadas
        a cada split.

    Raises
    ------
    AssertionError
        Si hay menos de ``n_test_days + 1`` fechas únicas en ``df``.

    Notes
    -----
    Adaptación ligera de ``read_data_v2.ipynb`` cells 11–12 (Sub-paso 3.2):
    se condensa la lógica de las dos celdas (validación + split) en una
    sola función. Sin cambiar cálculos.
    """
    dates_sorted = sorted(df[date_col].unique())
    n_dates = len(dates_sorted)
    assert n_dates >= n_test_days + 1, (
        f'Solo {n_dates} fechas, no se puede reservar {n_test_days} para test'
    )

    test_dates = dates_sorted[-n_test_days:]
    train_dates = dates_sorted[:-n_test_days]

    df_train = df[df[date_col].isin(train_dates)].reset_index(drop=True).copy()
    df_test = df[df[date_col].isin(test_dates)].reset_index(drop=True).copy()

    return df_train, df_test, train_dates, test_dates


def fit_outlier_thresholds(df_train, strategies):
    """Calcular umbrales de outliers a partir SOLO de los datos de train.

    Para cada columna con estrategia definida en ``strategies``, calcula
    los límites según el método indicado (``iqr``, ``percentile``,
    ``winsorize``, ``domain``). Los umbrales se calculan sobre
    ``df_train`` exclusivamente para evitar leakage de información del
    test.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Datos de entrenamiento.
    strategies : dict
        ``{columna: {'method': str, ...params...}}``. Métodos soportados:

        - ``iqr`` con ``factor`` (default 1.5): bounds = ``[Q1-f*IQR, Q3+f*IQR]``.
        - ``percentile`` con ``lower_p`` y ``upper_p``: bounds = cuantiles.
        - ``winsorize`` con ``lower_p`` y ``upper_p``: idem percentile, pero
          al aplicar se clipea en vez de eliminar filas.
        - ``domain`` con ``min_val`` y ``max_val``: bounds fijos.

    Returns
    -------
    dict
        ``{columna: {'method': str, 'low': float, 'high': float}}`` con
        los umbrales aplicables a cualquier DataFrame con esas columnas.

    Notes
    -----
    Copy literal de ``read_data_v2.ipynb`` cell 14 (Sub-paso 3.2).
    """
    thr = {}
    for col, cfg in strategies.items():
        if col not in df_train.columns:
            continue
        m = cfg['method']
        if m == 'iqr':
            q1 = df_train[col].quantile(0.25)
            q3 = df_train[col].quantile(0.75)
            iqr = q3 - q1
            f = cfg.get('factor', 1.5)
            thr[col] = {'method': 'iqr', 'low': q1 - f * iqr, 'high': q3 + f * iqr}
        elif m == 'percentile':
            thr[col] = {
                'method': 'percentile',
                'low': df_train[col].quantile(cfg['lower_p']),
                'high': df_train[col].quantile(cfg['upper_p']),
            }
        elif m == 'winsorize':
            thr[col] = {
                'method': 'winsorize',
                'low': df_train[col].quantile(cfg['lower_p']),
                'high': df_train[col].quantile(cfg['upper_p']),
            }
        elif m == 'domain':
            thr[col] = {'method': 'domain', 'low': cfg['min_val'], 'high': cfg['max_val']}
    return thr


def apply_outlier_thresholds(df, thresholds):
    """Aplicar umbrales pre-calculados: drop o clip según el método.

    Para cada columna con umbrales:

    - ``winsorize``: valores fuera de ``[low, high]`` se clipean (no se
      eliminan filas).
    - ``iqr``, ``percentile``, ``domain``: filas con valores fuera de
      ``[low, high]`` se eliminan.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame a procesar.
    thresholds : dict
        Salida de :func:`fit_outlier_thresholds`.

    Returns
    -------
    tuple
        ``(df_clean, n_dropped)`` con el DataFrame procesado (índice
        reseteado) y el número de filas eliminadas. Las columnas con
        ``winsorize`` no aportan a ``n_dropped`` (clipean, no eliminan).

    Notes
    -----
    Copy literal de ``read_data_v2.ipynb`` cell 14 (Sub-paso 3.2).
    """
    df = df.copy()
    n0 = len(df)
    for col, t in thresholds.items():
        if col not in df.columns:
            continue
        if t['method'] == 'winsorize':
            df[col] = df[col].clip(lower=t['low'], upper=t['high'])
        else:
            df = df[(df[col] >= t['low']) & (df[col] <= t['high'])]
    return df.reset_index(drop=True), n0 - len(df)


def decide_ohe_columns(df_train, known_categoricals, threshold_binary=2):
    """Decidir qué columnas deben pasar por OneHotEncoding.

    Empieza con la intersección de ``known_categoricals`` y las columnas
    presentes en ``df_train``. Si ``RAPID_TRAVERSING`` existe y tiene
    más de ``threshold_binary`` valores únicos, se añade al conjunto a
    OHE-ar; si tiene 2 o menos (binaria), se preserva como numérica.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Datos de entrenamiento.
    known_categoricals : list[str]
        Lista de nombres de columnas que se sabe son categóricas.
    threshold_binary : int, default 2
        Umbral por encima del cual ``RAPID_TRAVERSING`` deja de tratarse
        como binaria y se añade al OHE.

    Returns
    -------
    list[str]
        Columnas a pasar al :class:`~sklearn.preprocessing.OneHotEncoder`.

    Notes
    -----
    Adaptación ligera de ``read_data_v2.ipynb`` cell 15 (Sub-paso 3.2):
    bloque procedural reorganizado en función cohesiva. Sin cambiar
    lógica.
    """
    ohe_cols = [c for c in known_categoricals if c in df_train.columns]
    if 'RAPID_TRAVERSING' in df_train.columns:
        rt_unique = sorted(df_train['RAPID_TRAVERSING'].unique())
        if len(rt_unique) > threshold_binary:
            ohe_cols.append('RAPID_TRAVERSING')
    return ohe_cols


def fit_ohe(df_train, ohe_cols):
    """Ajustar un :class:`~sklearn.preprocessing.OneHotEncoder` sobre train.

    Utiliza ``handle_unknown='ignore'``, ``drop='first'`` y
    ``sparse_output=False``, coherente con el comportamiento del notebook v2.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Datos de entrenamiento.
    ohe_cols : list[str]
        Columnas a ajustar (típicamente la salida de
        :func:`decide_ohe_columns`).

    Returns
    -------
    sklearn.preprocessing.OneHotEncoder
        Encoder ya ajustado.

    Notes
    -----
    Adaptación ligera de ``read_data_v2.ipynb`` cell 15 (Sub-paso 3.2):
    inline 2 líneas convertidas en función con docstring.
    """
    ohe = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    ohe.fit(df_train[ohe_cols])
    return ohe


def apply_ohe(df, ohe, ohe_cols):
    """Aplicar un OneHotEncoder previamente ajustado a un DataFrame.

    Sustituye las columnas ``ohe_cols`` por su versión codificada con los
    nombres canónicos de ``ohe.get_feature_names_out(...)``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame a transformar (train o test).
    ohe : sklearn.preprocessing.OneHotEncoder
        Encoder ya ajustado (típicamente con :func:`fit_ohe`).
    ohe_cols : list[str]
        Columnas originales a codificar (las mismas usadas en el fit).

    Returns
    -------
    pandas.DataFrame
        DataFrame sin las columnas originales y con las codificadas.

    Notes
    -----
    Copy literal de ``read_data_v2.ipynb`` cell 15 (Sub-paso 3.2).
    """
    arr = ohe.transform(df[ohe_cols])
    new_names = ohe.get_feature_names_out(ohe_cols)
    df_new = pd.DataFrame(arr, columns=new_names, index=df.index)
    return pd.concat([df.drop(columns=ohe_cols), df_new], axis=1)


def fit_correlation_drop(df_train, threshold, exclude=()):
    """Identificar columnas correlacionadas a eliminar (por mayor varianza).

    Para cada par de columnas numéricas con correlación absoluta superior
    al umbral, conserva la de mayor varianza y marca la otra para drop.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Datos de entrenamiento. La correlación se calcula solo sobre train.
    threshold : float
        Umbral de correlación absoluta (típicamente 0.95).
    exclude : iterable[str], default ``()``
        Columnas a no considerar (típicamente targets y la columna de
        fecha).

    Returns
    -------
    tuple
        ``(to_drop, pairs)``:

        - ``to_drop`` : list[str], columnas a eliminar.
        - ``pairs`` : list[tuple[str, str, float]] con los pares
          ``(col_a, col_b, |corr|)`` que superaron el umbral.

    Notes
    -----
    Copy literal de ``read_data_v2.ipynb`` cell 16 (Sub-paso 3.2).

    La conversión de ``set`` a ``list`` mantiene la ordenación no
    determinista entre plataformas (causa de la diferencia cosmética
    cross-OS en ``filter_metadata.json`` documentada en
    ``READ_DATA_V2_SMOKE_TEST.md`` §5.1). La mejora ``sorted(to_drop)``
    queda explícitamente fuera del alcance del Sub-paso 3.2 (regla copy +
    docstring).
    """
    cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in exclude]
    corr = df_train[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    variances = df_train[cols].var()
    # Recorrer todos los pares (i, j) con |corr| > threshold
    pairs = []
    for j in upper.columns:
        for i in upper.index:
            v = upper.loc[i, j]
            if pd.notna(v) and v > threshold:
                pairs.append((i, j, v))

    for i, j, v in pairs:
        if i in to_drop or j in to_drop:
            continue
        # Conservar la de mayor varianza
        loser = i if variances[i] < variances[j] else j
        to_drop.add(loser)

    return list(to_drop), pairs
