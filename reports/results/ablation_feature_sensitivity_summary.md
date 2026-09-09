# Análisis de sensibilidad de features — resumen (nomenclatura corregida)

**Reencuadre 2026-06-28:** lo que antes se etiquetó como *leakage* se reinterpreta como **ambigüedad documental y análisis de sensibilidad**. Todas las columnas (incluidas SPEED_RMS y TEMPERATURE) son señales disponibles de la CNC. El escenario **principal es A `cnc_available_all`**.

Escenarios presentes (datos del Bloque 1, LightGBM/XGBoost, single-output, test filtrado):

- **A `cnc_available_all`** (n=28, incluye SPEED_RMS y TEMPERATURE) — **PRINCIPAL**.
- **B `no_speed_rms`** (A sin SPEED_RMS) — sensibilidad.
- **D `conservative_no_speed_temperature`** (A sin SPEED_RMS ni TEMPERATURE) — conservador secundario.
- **C `no_temperature`** (A sin TEMPERATURE) — **PENDIENTE** (requiere corrida de sensibilidad; no se entrena hasta confirmación).

## Sensibilidad de R² frente al escenario principal A (condición all)

| Modelo | Target | A cnc_available_all | B no_speed_rms (Δ) | D conservative (Δ) |
|---|---|---:|---:|---:|
| LightGBM | ACCEL_PEAK | 0.8513 | 0.7869 (-0.0644) | 0.7770 (-0.0743) |
| LightGBM | ACCEL_RMS | 0.8547 | 0.8014 (-0.0533) | 0.7903 (-0.0644) |
| LightGBM | ACCEL_RMS_FREQ | 0.5815 | 0.5783 (-0.0032) | 0.5792 (-0.0024) |
| XGBoost | ACCEL_PEAK | 0.8463 | 0.7801 (-0.0662) | 0.7685 (-0.0778) |
| XGBoost | ACCEL_RMS | 0.8541 | 0.7976 (-0.0565) | 0.7860 (-0.0680) |
| XGBoost | ACCEL_RMS_FREQ | 0.5838 | 0.5855 (+0.0017) | 0.5869 (+0.0031) |

## Lectura

- **Resultado principal (A, in-distribution):** LightGBM·ACCEL_RMS·all **R²=0,8547** sigue siendo el resultado principal del TFM.
- **Sensibilidad:** retirar SPEED_RMS baja el R² (≈−0,05 en magnitud); retirar también TEMPERATURE baja un poco más. Esto **mide la contribución** de esas señales, no un fallo metodológico.
- La memoria debe presentar A como principal y B/C/D como análisis de sensibilidad/robustez, declarando la ambigüedad documental sobre el origen de SPEED_RMS/TEMPERATURE.

*Reencuadre de `ablation_feature_leakage.csv` (Bloque 1). Sin reentrenar. Los archivos `*leakage*` se conservan como históricos.*
