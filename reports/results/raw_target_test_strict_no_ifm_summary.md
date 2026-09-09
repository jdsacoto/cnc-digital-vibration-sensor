# Test temporal sin filtrar targets — strict_no_ifm — resumen

Bloque 3 · features **strict_no_ifm** (sin SPEED_RMS ni TEMPERATURE) · LightGBM/XGBoost · single-output.

**Control de fidelidad MD5:** train regenerado = OK (idéntico a df_train.csv); test filtrado regenerado = OK (idéntico a df_test.csv).

**Filas de test:** filtrado = 35832 · raw-target = 37257 · recuperadas = **+1425** (4.0% más).


## 1–2. R² y MAE: test filtrado vs test raw-target (condición `all`)

| Modelo | Target | R² filtrado | R² raw | Δ R² | MAE filtrado | MAE raw | Δ MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | ACCEL_PEAK | 0.7770 | 0.7156 | -0.0614 | 86.73 | 108.91 | +22.18 |
| LightGBM | ACCEL_RMS | 0.7903 | 0.7501 | -0.0402 | 34.50 | 41.54 | +7.05 |
| LightGBM | ACCEL_RMS_FREQ | 0.5792 | 0.5591 | -0.0201 | 74.70 | 84.46 | +9.77 |
| XGBoost | ACCEL_PEAK | 0.7685 | 0.7184 | -0.0501 | 100.49 | 121.35 | +20.85 |
| XGBoost | ACCEL_RMS | 0.7860 | 0.7550 | -0.0310 | 34.90 | 41.51 | +6.62 |
| XGBoost | ACCEL_RMS_FREQ | 0.5869 | 0.5667 | -0.0201 | 74.26 | 83.91 | +9.65 |

### Detalle por condición de carga (Δ R² raw − filtrado)

| Modelo | Target | all | load | no_load |
|---|---|---:|---:|---:|
| LightGBM | ACCEL_PEAK | -0.0614 | -0.1024 | -0.0481 |
| LightGBM | ACCEL_RMS | -0.0402 | -0.0572 | -0.0391 |
| LightGBM | ACCEL_RMS_FREQ | -0.0201 | -0.0303 | -0.0038 |
| XGBoost | ACCEL_PEAK | -0.0501 | -0.0947 | -0.0239 |
| XGBoost | ACCEL_RMS | -0.0310 | -0.0466 | -0.0348 |
| XGBoost | ACCEL_RMS_FREQ | -0.0201 | -0.0355 | +0.0044 |

## 3. Filas recuperadas: 1425 (las que el IQR de targets eliminaba del test)


## 4. Estadísticas de los targets en las filas recuperadas

| Target | n | min | max | media | std | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACCEL_PEAK | 1425 | 42.3 | 2683.1 | 1321.1 | 845.6 | 2345.8 | 2470.8 |
| ACCEL_RMS | 1425 | 16.5 | 1338.4 | 495.3 | 307.2 | 838.2 | 913.3 |
| ACCEL_RMS_FREQ | 1425 | 13.3 | 1081.2 | 582.1 | 348.3 | 1024.2 | 1064.0 |

(Comparar con el rango del test filtrado, p. ej. ACCEL_RMS filtrado min=6.6 max=685.8.)

## 5. Interpretación metodológica

- **Test filtrado (in-distribution):** mide el desempeño sobre el régimen de operación normal, excluyendo los picos de vibración que el filtro IQR retira. Es una cota optimista pero honesta del comportamiento en condiciones típicas.
- **Test raw-target (temporal con picos conservados):** más realista respecto a uso operativo, porque en producción no conoceríamos el target antes de predecirlo y los picos SÍ ocurren. La diferencia de R² mide la **sensibilidad del sensor ante eventos extremos**, no un fallo del modelo.
- **Resultado principal recomendado:** reportar AMBOS. El test raw-target debe ser la **evaluación principal de validez operativa**; el test filtrado, el **análisis in-distribution** (comparabilidad con el registro histórico).
- La memoria debe distinguir explícitamente *evaluación in-distribution* (filtrada) de *evaluación temporal con picos conservados* (raw-target), sin presentar una como sustituta de la otra. Si el R² cae en raw-target, es la degradación esperable ante extremos eliminados por el IQR.

*Generado por `src/experiments/evaluate_raw_target_test.py`. Cifras del CSV homónimo; ninguna escrita a mano.*
