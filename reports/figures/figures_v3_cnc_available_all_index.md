# Índice de figuras v3 — cnc_available_all

Todas generadas por `src/experiments/generate_figures_v3.py` (solo lectura de CSV, sin reentrenar). Escenario principal **cnc_available_all** (28 features, incluye SPEED_RMS y TEMPERATURE).

| Figura | Fuente CSV | Qué muestra | Word | Frase de interpretación |
|---|---|---|---|---|
| `fig_top10_r2_filtered_v3_cnc_available_all.png` | 01_models_v3…results.csv (filtered_test) | Top-10 R² in-distribution | Cap. 5 | «Los boosters de árboles dominan el registro in-distribution; ACCEL_RMS lidera con R²≈0,85.» |
| `fig_top10_r2_raw_target_v3_cnc_available_all.png` | 01_models_v3…results.csv (raw_target_test) | Top-10 R² con picos conservados | Cap. 5 | «Al conservar los picos, el orden se mantiene y los R² máximos rondan 0,82.» |
| `fig_filtered_vs_raw_by_target_v3_cnc_available_all.png` | 01_models_v3…results.csv | Caída filtered→raw por target (all) | Cap. 5 / 6 | «La degradación al conservar los picos es moderada (−0,017 a −0,059): el sensor pierde precisión pero no se rompe.» |
| `fig_best_model_by_target_raw_v3_cnc_available_all.png` | 01_models_v3…results.csv (raw) | Mejor modelo por target en raw | Cap. 5 | «XGBoost encabeza en magnitud (PEAK, RMS) y LightGBM en frecuencia, con diferencias marginales: empate técnico.» |
| `fig_model_family_raw_all_v3_cnc_available_all.png` | 01_models_v3…results.csv (raw) | Comparativa de familias en raw (all) | Anexo A / Cap. 5 | «Las cinco familias convergen en magnitud; el MLP queda por debajo y la frecuencia es difícil para todas.» |
| `fig_target_difficulty_v3_cnc_available_all.png` | 01_models_v3…results.csv | Dificultad relativa de los 3 targets | Cap. 6 | «ACCEL_RMS_FREQ tiene un techo ≈0,57–0,58 muy por debajo de las magnitudes temporales: límite de la información disponible.» |
| `fig_speed_rms_sensitivity_v3.png` | ablation_feature_sensitivity.csv | Sensibilidad a SPEED_RMS/TEMPERATURE | Cap. 6 / Anexo A | «Retirar SPEED_RMS/TEMPERATURE reduce R² de forma controlada; es análisis de sensibilidad por ambigüedad documental, no fuga demostrada.» |

**Uso recomendado:** Cap. 5 (Resultados) → figuras 1–4; Cap. 6 (Discusión) → figuras 3, 6 y 7; Anexo A → figura 5 y duplicados de robustez.
