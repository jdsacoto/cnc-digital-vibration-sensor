# Registro maestro PRINCIPAL — cnc_available_all (v3) — resumen

Escenario **principal del TFM: `cnc_available_all`** (todas las señales disponibles de la CNC; n=28, incluye SPEED_RMS y TEMPERATURE). 5 familias × 3 targets × 3 cargas × {single, multi}, evaluadas sobre **filtered_test** (in-distribution) y **raw_target_test** (picos conservados). Mismos hiperparámetros que models_v2.


## 1. Top-10 por R² — filtered_test (in-distribution)

| # | model | output | target | load | R² | MAE | RMSE |
|---|---|---|---|---|---:|---:|---:|
| 1 | LightGBM | single | ACCEL_RMS | all | 0.8547 | 26.50 | 54.96 |
| 2 | LightGBM | multi | ACCEL_RMS | all | 0.8547 | 26.50 | 54.96 |
| 3 | XGBoost | multi | ACCEL_RMS | all | 0.8541 | 26.76 | 55.07 |
| 4 | XGBoost | single | ACCEL_RMS | all | 0.8541 | 26.76 | 55.07 |
| 5 | LightGBM | single | ACCEL_PEAK | all | 0.8513 | 66.61 | 133.70 |
| 6 | LightGBM | multi | ACCEL_PEAK | all | 0.8513 | 66.61 | 133.70 |
| 7 | RandomForest | multi | ACCEL_RMS | all | 0.8513 | 27.73 | 55.59 |
| 8 | RandomForest | single | ACCEL_RMS | all | 0.8497 | 26.27 | 55.90 |
| 9 | XGBoost | single | ACCEL_PEAK | all | 0.8463 | 69.76 | 135.93 |
| 10 | XGBoost | multi | ACCEL_PEAK | all | 0.8463 | 69.76 | 135.93 |

## 2. Top-10 por R² — raw_target_test (picos conservados)

| # | model | output | target | load | R² | MAE | RMSE |
|---|---|---|---|---|---:|---:|---:|
| 1 | XGBoost | single | ACCEL_RMS | all | 0.8214 | 32.63 | 72.22 |
| 2 | XGBoost | multi | ACCEL_RMS | all | 0.8214 | 32.63 | 72.22 |
| 3 | LightGBM | multi | ACCEL_RMS | all | 0.8204 | 32.43 | 72.43 |
| 4 | LightGBM | single | ACCEL_RMS | all | 0.8204 | 32.43 | 72.43 |
| 5 | RandomForest | multi | ACCEL_RMS | all | 0.8156 | 33.83 | 73.38 |
| 6 | RandomForest | single | ACCEL_RMS | all | 0.8131 | 32.46 | 73.89 |
| 7 | CatBoost | single | ACCEL_RMS | all | 0.8022 | 36.12 | 76.01 |
| 8 | CatBoost | multi | ACCEL_RMS | all | 0.8022 | 36.12 | 76.01 |
| 9 | XGBoost | multi | ACCEL_PEAK | all | 0.7921 | 89.26 | 196.04 |
| 10 | XGBoost | single | ACCEL_PEAK | all | 0.7921 | 89.26 | 196.04 |

## 3. Mejor modelo por target y evaluation_type (condición all)

| evaluation_type | target | mejor modelo | output | R² | MAE |
|---|---|---|---|---:|---:|
| filtered_test | ACCEL_PEAK | LightGBM | multi | 0.8513 | 66.61 |
| filtered_test | ACCEL_RMS | LightGBM | multi | 0.8547 | 26.50 |
| filtered_test | ACCEL_RMS_FREQ | XGBoost | single | 0.5838 | 74.84 |
| raw_target_test | ACCEL_PEAK | XGBoost | single | 0.7921 | 89.26 |
| raw_target_test | ACCEL_RMS | XGBoost | single | 0.8214 | 32.63 |
| raw_target_test | ACCEL_RMS_FREQ | LightGBM | multi | 0.5663 | 85.68 |

## 4. Mejor modelo por target y condición de carga (filtered_test)

| target | load | mejor modelo | output | R² |
|---|---|---|---|---:|
| ACCEL_PEAK | all | LightGBM | multi | 0.8513 |
| ACCEL_PEAK | no_load | LightGBM | multi | 0.5961 |
| ACCEL_PEAK | load | LightGBM | multi | 0.8106 |
| ACCEL_RMS | all | LightGBM | multi | 0.8547 |
| ACCEL_RMS | no_load | LightGBM | multi | 0.6226 |
| ACCEL_RMS | load | RandomForest | multi | 0.8141 |
| ACCEL_RMS_FREQ | all | XGBoost | single | 0.5838 |
| ACCEL_RMS_FREQ | no_load | LightGBM | multi | 0.5073 |
| ACCEL_RMS_FREQ | load | LightGBM | multi | 0.5144 |

## 5. Resultado principal (ACCEL_RMS · all) y comparación con escenarios de sensibilidad

| Escenario / evaluación | mejor modelo | R² |
|---|---|---:|
| **cnc_available_all · filtered_test (PRINCIPAL in-distribution)** | LightGBM (multi) | **0.8547** |
| **cnc_available_all · raw_target_test (PRINCIPAL picos conservados)** | XGBoost (single) | **0.8214** |
| no_speed_rms (sensibilidad) | ver ablation_feature_sensitivity.csv | — |
| conservative_no_speed_temperature (conservador secundario) | ver ablation_feature_sensitivity.csv | — |

(Control de fidelidad: el filtered_test de cnc_available_all reproduce models_v2; el histórico R²=0,8547 es, por tanto, el resultado principal in-distribution, no un baseline con fuga.)

## 6. Interpretación metodológica

- **cnc_available_all es el resultado principal** del TFM: usa todas las señales disponibles de la CNC (incl. SPEED_RMS y TEMPERATURE), conforme a la definición oficial del problema.
- **Dos tests:** filtered_test mide el desempeño in-distribution (operación normal); raw_target_test mide la robustez ante los picos que el filtro IQR retiraba (validez operativa).
- **Principal a reportar:** filtered_test LightGBM R²=0.8547 (in-distribution) y raw_target_test XGBoost R²=0.8214 (picos conservados), ambos en ACCEL_RMS·all.
- **Sensibilidad/robustez:** no_speed_rms (B) y conservative_no_speed_temperature (D) cuantifican la contribución de SPEED_RMS/TEMPERATURE; NO son leakage, sino ambigüedad documental.

## 7. Advertencia sobre el modelo ganador

- Ganador filtered_test (all): LightGBM (single) en ACCEL_RMS (R²=0.8547).
- Ganador raw_target_test (all): XGBoost (single) en ACCEL_RMS (R²=0.8214).
- Ranking ACCEL_RMS·all en raw_target_test (no asumir que gana LightGBM):
  - XGBoost (single): 0.8214
  - XGBoost (multi): 0.8214
  - LightGBM (multi): 0.8204
  - LightGBM (single): 0.8204
  - RandomForest (multi): 0.8156
  - RandomForest (single): 0.8131
  - CatBoost (multi): 0.8022
  - CatBoost (single): 0.8022
  - MLPRegressor (multi): 0.7659
  - MLPRegressor (single): 0.7487

## 8. Recomendaciones para actualizar el Word

- Mantener cnc_available_all como resultado principal; reportar filtered (in-distribution) y raw-target (picos conservados).
- Declarar SPEED_RMS y TEMPERATURE como señales disponibles de la CNC, con nota de ambigüedad documental sobre su origen exacto; presentar B/D como análisis de sensibilidad.
- Revisar el modelo declarado como mejor según el ranking real de la sección 7.
- Regenerar figuras del Cap. 8–9 desde 01_models_v3_cnc_available_all_results.csv (Bloque 5B).

*Generado por `src/experiments/models_v3_cnc_available_all.py`. Cifras del CSV homónimo; ninguna a mano.*
