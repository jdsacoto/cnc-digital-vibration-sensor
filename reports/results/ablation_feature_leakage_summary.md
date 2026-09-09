# Ablación de features por sospecha de leakage — resumen

Bloque 1 · LightGBM y XGBoost · single-output · misma partición temporal · `df_train.csv`/`df_test.csv` actuales.

Escenarios: **A** `baseline_current` (todas las features) · **B** `no_speed_rms` (sin SPEED_RMS) · **C** `strict_no_ifm` (sin SPEED_RMS ni TEMPERATURE).


## 1. Mejor modelo por escenario y target (condición `all`)

| Escenario | Target | Mejor modelo | R² | MAE |
|---|---|---|---:|---:|
| baseline_current | ACCEL_PEAK | LightGBM | 0.8513 | 66.61 |
| baseline_current | ACCEL_RMS | LightGBM | 0.8547 | 26.50 |
| baseline_current | ACCEL_RMS_FREQ | XGBoost | 0.5838 | 74.84 |
| no_speed_rms | ACCEL_PEAK | LightGBM | 0.7869 | 87.99 |
| no_speed_rms | ACCEL_RMS | LightGBM | 0.8014 | 33.70 |
| no_speed_rms | ACCEL_RMS_FREQ | XGBoost | 0.5855 | 76.83 |
| strict_no_ifm | ACCEL_PEAK | LightGBM | 0.7770 | 86.73 |
| strict_no_ifm | ACCEL_RMS | LightGBM | 0.7903 | 34.50 |
| strict_no_ifm | ACCEL_RMS_FREQ | XGBoost | 0.5869 | 74.26 |

## 2. Caída de R² al quitar SPEED_RMS (B − A)

| Modelo | Target | Carga | R² A | R² B | Δ R² |
|---|---|---|---:|---:|---:|
| LightGBM | ACCEL_PEAK | all | 0.8513 | 0.7869 | -0.0644 |
| LightGBM | ACCEL_PEAK | no_load | 0.5961 | 0.4293 | -0.1668 |
| LightGBM | ACCEL_PEAK | load | 0.8106 | 0.7363 | -0.0743 |
| LightGBM | ACCEL_RMS | all | 0.8547 | 0.8014 | -0.0533 |
| LightGBM | ACCEL_RMS | no_load | 0.6226 | 0.4267 | -0.1959 |
| LightGBM | ACCEL_RMS | load | 0.8094 | 0.7565 | -0.0528 |
| LightGBM | ACCEL_RMS_FREQ | all | 0.5815 | 0.5783 | -0.0032 |
| LightGBM | ACCEL_RMS_FREQ | no_load | 0.5073 | 0.5127 | +0.0054 |
| LightGBM | ACCEL_RMS_FREQ | load | 0.5144 | 0.5124 | -0.0020 |
| XGBoost | ACCEL_PEAK | all | 0.8463 | 0.7801 | -0.0662 |
| XGBoost | ACCEL_PEAK | no_load | 0.5812 | 0.3310 | -0.2502 |
| XGBoost | ACCEL_PEAK | load | 0.8041 | 0.7493 | -0.0548 |
| XGBoost | ACCEL_RMS | all | 0.8541 | 0.7976 | -0.0565 |
| XGBoost | ACCEL_RMS | no_load | 0.6145 | 0.3909 | -0.2236 |
| XGBoost | ACCEL_RMS | load | 0.8051 | 0.7524 | -0.0527 |
| XGBoost | ACCEL_RMS_FREQ | all | 0.5838 | 0.5855 | +0.0017 |
| XGBoost | ACCEL_RMS_FREQ | no_load | 0.5009 | 0.4785 | -0.0224 |
| XGBoost | ACCEL_RMS_FREQ | load | 0.4988 | 0.4963 | -0.0025 |

## 3. Caída adicional de R² al quitar TEMPERATURE (C − B)

| Modelo | Target | Carga | R² B | R² C | Δ R² |
|---|---|---|---:|---:|---:|
| LightGBM | ACCEL_PEAK | all | 0.7869 | 0.7770 | -0.0098 |
| LightGBM | ACCEL_PEAK | no_load | 0.4293 | 0.4291 | -0.0002 |
| LightGBM | ACCEL_PEAK | load | 0.7363 | 0.6980 | -0.0382 |
| LightGBM | ACCEL_RMS | all | 0.8014 | 0.7903 | -0.0111 |
| LightGBM | ACCEL_RMS | no_load | 0.4267 | 0.4208 | -0.0059 |
| LightGBM | ACCEL_RMS | load | 0.7565 | 0.7276 | -0.0289 |
| LightGBM | ACCEL_RMS_FREQ | all | 0.5783 | 0.5792 | +0.0008 |
| LightGBM | ACCEL_RMS_FREQ | no_load | 0.5127 | 0.5119 | -0.0008 |
| LightGBM | ACCEL_RMS_FREQ | load | 0.5124 | 0.5135 | +0.0011 |
| XGBoost | ACCEL_PEAK | all | 0.7801 | 0.7685 | -0.0116 |
| XGBoost | ACCEL_PEAK | no_load | 0.3310 | 0.3455 | +0.0145 |
| XGBoost | ACCEL_PEAK | load | 0.7493 | 0.7145 | -0.0348 |
| XGBoost | ACCEL_RMS | all | 0.7976 | 0.7860 | -0.0115 |
| XGBoost | ACCEL_RMS | no_load | 0.3909 | 0.4007 | +0.0098 |
| XGBoost | ACCEL_RMS | load | 0.7524 | 0.7393 | -0.0131 |
| XGBoost | ACCEL_RMS_FREQ | all | 0.5855 | 0.5869 | +0.0014 |
| XGBoost | ACCEL_RMS_FREQ | no_load | 0.4785 | 0.4831 | +0.0046 |
| XGBoost | ACCEL_RMS_FREQ | load | 0.4963 | 0.4996 | +0.0033 |

## 4. Lectura cuantitativa

- Quitar **SPEED_RMS** (B vs A): Δ R² medio = **-0.0741**, peor caso = **-0.2502** (XGBoost/ACCEL_PEAK/no_load).
- Quitar también **TEMPERATURE** (C vs A): Δ R² medio = **-0.0814**, peor caso = **-0.2357** (XGBoost/ACCEL_PEAK/no_load).
- Resultado estrella (LightGBM·ACCEL_RMS·all): A=0.8547 → B=0.8014 → C=0.7903.

## 5. Recomendación metodológica

- **Criterio rector:** no es la correlación con el target, sino la *disponibilidad legítima* de la variable en un sensor virtual basado en señales internas del PLC.
- **SPEED_RMS:** el rango real de los datos (≈0.09–11) es incompatible con la velocidad de giro del husillo (`SPINDLE_ACTUAL_SPEED` ≈ 10 000 rpm) y compatible con la velocidad RMS de vibración del IFM (mm/s). Mientras su origen no se demuestre como señal interna del PLC, **debe excluirse del escenario principal**.
- **TEMPERATURE:** procede del módulo IFM (temperatura de la taladrina). No es vibración, pero tampoco se demuestra como señal interna del PLC. Para una postura estricta de pureza, el escenario principal debe ser **`strict_no_ifm`** (C).
- **Escenario principal propuesto:** **C `strict_no_ifm`** (sin SPEED_RMS ni TEMPERATURE).
- **Escenario aumentado / sensibilidad:** **A `baseline_current`**, declarado explícitamente como registro histórico con variables del IFM, no como resultado principal.
- **Qué debe decir la memoria:** (1) que el sensor virtual usa señales internas del PLC; (2) que SPEED_RMS y TEMPERATURE provienen del IFM y por eso se excluyen del experimento principal; (3) reportar el resultado principal con C y el delta frente a A como análisis de sensibilidad.

*Generado por `src/experiments/ablation_feature_leakage.py`. Las cifras provienen del CSV homónimo; ninguna está escrita a mano.*
