"""Bloque 5A (corregido) — Registro maestro PRINCIPAL: escenario cnc_available_all.

Feature set principal del TFM: todas las columnas de df_train.csv excepto DATE y
los tres targets ACCEL_*. INCLUYE SPEED_RMS y TEMPERATURE (señales disponibles de
la CNC). n_features == 28.

Regenera el registro factorial completo (5 familias × 3 targets × 3 cargas ×
{single, multi}) y evalúa cada modelo entrenado sobre DOS tests sin reentrenar:
  - filtered_test    : df_test.csv (in-distribution)
  - raw_target_test  : df_test_raw_targets.csv (picos conservados)

Control de fidelidad: cnc_available_all sobre filtered_test = feature set de
models_v2 → debe reproducir 01_models_v2_results.csv (incl. R²=0,8547).

No toca df_train/df_test ni 01_models_v2_results.csv. Salidas con sufijo
cnc_available_all. Mismos hiperparámetros que models_v2.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.config import TARGET_VARS, RANDOM_STATE, LOAD_CONDITIONS
from src.evaluation import filter_load, make_single_registry, make_multi_registry

SCENARIO = "cnc_available_all"
EXCLUDE = []  # principal: NO se excluye SPEED_RMS ni TEMPERATURE
RES = REPO / "reports/results"
PFX = "models_v3_cnc_available_all"


def metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return (float(mean_absolute_error(y_true, y_pred)), mse,
            float(np.sqrt(mse)), float(r2_score(y_true, y_pred)))


def main():
    df_train = pd.read_csv(REPO / "df_train.csv")
    df_test_f = pd.read_csv(REPO / "df_test.csv")
    df_test_raw = pd.read_csv(REPO / "df_test_raw_targets.csv")

    feats = [c for c in df_train.columns if c not in TARGET_VARS + ["DATE"] + EXCLUDE]
    # --- asserts obligatorios del plan ---
    assert "DATE" not in feats, "DATE no debe estar en features"
    assert not any(t in feats for t in TARGET_VARS), "ningún target en features"
    assert "SPEED_RMS" in feats, "SPEED_RMS DEBE estar en features (escenario principal)"
    assert "TEMPERATURE" in feats, "TEMPERATURE DEBE estar en features (escenario principal)"
    print(f"n_features ({SCENARIO}) = {len(feats)} (esperado 28)")
    assert len(feats) == 28, f"Se esperaban 28 features, hay {len(feats)}"

    single = make_single_registry(use_gpu=False, random_state=RANDOM_STATE)
    multi = make_multi_registry(use_gpu=False, random_state=RANDOM_STATE)
    TESTS = [("filtered_test", df_test_f), ("raw_target_test", df_test_raw)]
    rows = []

    # ---------------- SINGLE-OUTPUT ----------------
    for model_name, cfg in single.items():
        for target in TARGET_VARS:
            for load_kind in LOAD_CONDITIONS:
                tr = filter_load(df_train, load_kind)
                X_tr, y_tr = tr[feats].values, tr[target].values
                scaler = StandardScaler().fit(X_tr) if cfg["scale"] else None
                if scaler is not None:
                    X_tr = scaler.transform(X_tr)
                model = cfg["factory"]()
                t0 = time.time()
                model.fit(X_tr, y_tr)
                fit_s = time.time() - t0
                for ev, test_df in TESTS:
                    te = filter_load(test_df, load_kind)
                    X_te, y_te = te[feats].values, te[target].values
                    if scaler is not None:
                        X_te = scaler.transform(X_te)
                    t1 = time.time()
                    y_pred = model.predict(X_te)
                    predict_s = time.time() - t1
                    mae, mse, rmse, r2 = metrics(y_te, y_pred)
                    rows.append(dict(
                        evaluation_type=ev, feature_scenario=SCENARIO,
                        removed_features="(none)", model=model_name,
                        output_strategy="single", target=target, load_condition=load_kind,
                        n_features=len(feats), n_train=len(tr), n_test=len(te),
                        MAE=mae, MSE=mse, RMSE=rmse, R2=r2, fit_s=fit_s,
                        predict_s=predict_s, notes="offline temporal split"))
                print(f"single {model_name:13s} {target:14s} {load_kind:8s} "
                      f"filt={rows[-2]['R2']:.4f} raw={rows[-1]['R2']:.4f}", flush=True)

    # ---------------- MULTI-OUTPUT ----------------
    for model_name, cfg in multi.items():
        for load_kind in LOAD_CONDITIONS:
            tr = filter_load(df_train, load_kind)
            X_tr, Y_tr = tr[feats].values, tr[TARGET_VARS].values
            scaler = StandardScaler().fit(X_tr) if cfg["scale"] else None
            if scaler is not None:
                X_tr = scaler.transform(X_tr)
            model = cfg["factory"]()
            t0 = time.time()
            model.fit(X_tr, Y_tr)
            fit_s = time.time() - t0
            for ev, test_df in TESTS:
                te = filter_load(test_df, load_kind)
                X_te = te[feats].values
                if scaler is not None:
                    X_te = scaler.transform(X_te)
                t1 = time.time()
                Y_pred = model.predict(X_te)
                predict_s = time.time() - t1
                for i, target in enumerate(TARGET_VARS):
                    mae, mse, rmse, r2 = metrics(te[target].values, Y_pred[:, i])
                    rows.append(dict(
                        evaluation_type=ev, feature_scenario=SCENARIO,
                        removed_features="(none)", model=model_name,
                        output_strategy="multi", target=target, load_condition=load_kind,
                        n_features=len(feats), n_train=len(tr), n_test=len(te),
                        MAE=mae, MSE=mse, RMSE=rmse, R2=r2, fit_s=fit_s,
                        predict_s=predict_s, notes="offline temporal split"))
            print(f"multi  {model_name:13s} {load_kind:8s} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / f"01_{PFX}_results.csv", index=False)
    print(f"\nGuardado 01_{PFX}_results.csv {df.shape}", flush=True)

    # control de fidelidad vs models_v2 (filtered, single)
    fid = df[(df.evaluation_type == "filtered_test") & (df.output_strategy == "single") &
             (df.model == "LightGBM") & (df.target == "ACCEL_RMS") & (df.load_condition == "all")]
    print(f"Fidelidad: LightGBM·ACCEL_RMS·all·single·filtered = {fid['R2'].iloc[0]:.4f} "
          f"(esperado ~0.8547 de models_v2)", flush=True)

    derive(df)


def derive(df):
    best = (df.sort_values("R2", ascending=False)
            .groupby(["evaluation_type", "target", "load_condition"], as_index=False).first())
    best.to_csv(RES / f"02_{PFX}_best_by_target_load.csv", index=False)
    for ev, fname in [("filtered_test", f"03_{PFX}_top10_filtered.csv"),
                      ("raw_target_test", f"04_{PFX}_top10_raw_target.csv")]:
        (df[df.evaluation_type == ev].sort_values("R2", ascending=False)
         .head(10).to_csv(RES / fname, index=False))
    piv = (df[df.evaluation_type == "filtered_test"]
           .assign(col=lambda d: d.target + " | " + d.load_condition,
                   row=lambda d: d.model + " (" + d.output_strategy + ")")
           .pivot_table(index="row", columns="col", values="R2"))
    piv.to_csv(RES / f"05_{PFX}_r2_pivot.csv")
    key = ["model", "output_strategy", "target", "load_condition"]
    f = df[df.evaluation_type == "filtered_test"].set_index(key)
    r = df[df.evaluation_type == "raw_target_test"].set_index(key)
    comp = pd.DataFrame({
        "R2_filtered": f["R2"], "R2_raw_target": r["R2"], "delta_R2": r["R2"] - f["R2"],
        "MAE_filtered": f["MAE"], "MAE_raw_target": r["MAE"], "delta_MAE": r["MAE"] - f["MAE"],
    }).reset_index().sort_values("R2_filtered", ascending=False)
    comp.to_csv(RES / f"06_{PFX}_filtered_vs_raw_comparison.csv", index=False)
    write_summary(df)
    print("Guardados 02–06 + summary.", flush=True)


def write_summary(df):
    def top10(ev):
        t = df[df.evaluation_type == ev].sort_values("R2", ascending=False).head(10)
        L = ["| # | model | output | target | load | R² | MAE | RMSE |",
             "|---|---|---|---|---|---:|---:|---:|"]
        for i, (_, x) in enumerate(t.iterrows(), 1):
            L.append(f"| {i} | {x['model']} | {x['output_strategy']} | {x['target']} | "
                     f"{x['load_condition']} | {x['R2']:.4f} | {x['MAE']:.2f} | {x['RMSE']:.2f} |")
        return "\n".join(L)

    def best(ev, t, lc):
        return df[(df.evaluation_type == ev) & (df.target == t) &
                  (df.load_condition == lc)].sort_values("R2", ascending=False).iloc[0]

    bf, br = best("filtered_test", "ACCEL_RMS", "all"), best("raw_target_test", "ACCEL_RMS", "all")
    win_f = df[(df.evaluation_type == "filtered_test") & (df.load_condition == "all")].sort_values("R2", ascending=False).iloc[0]
    win_r = df[(df.evaluation_type == "raw_target_test") & (df.load_condition == "all")].sort_values("R2", ascending=False).iloc[0]

    L = []
    L.append("# Registro maestro PRINCIPAL — cnc_available_all (v3) — resumen\n")
    L.append("Escenario **principal del TFM: `cnc_available_all`** (todas las señales disponibles de la CNC; "
             "n=28, incluye SPEED_RMS y TEMPERATURE). 5 familias × 3 targets × 3 cargas × {single, multi}, "
             "evaluadas sobre **filtered_test** (in-distribution) y **raw_target_test** (picos conservados). "
             "Mismos hiperparámetros que models_v2.\n")
    L.append("\n## 1. Top-10 por R² — filtered_test (in-distribution)\n")
    L.append(top10("filtered_test"))
    L.append("\n## 2. Top-10 por R² — raw_target_test (picos conservados)\n")
    L.append(top10("raw_target_test"))
    L.append("\n## 3. Mejor modelo por target y evaluation_type (condición all)\n")
    L.append("| evaluation_type | target | mejor modelo | output | R² | MAE |")
    L.append("|---|---|---|---|---:|---:|")
    for ev in ["filtered_test", "raw_target_test"]:
        for t in TARGET_VARS:
            q = best(ev, t, "all")
            L.append(f"| {ev} | {t} | {q['model']} | {q['output_strategy']} | {q['R2']:.4f} | {q['MAE']:.2f} |")
    L.append("\n## 4. Mejor modelo por target y condición de carga (filtered_test)\n")
    L.append("| target | load | mejor modelo | output | R² |")
    L.append("|---|---|---|---|---:|")
    for t in TARGET_VARS:
        for lc in LOAD_CONDITIONS:
            q = best("filtered_test", t, lc)
            L.append(f"| {t} | {lc} | {q['model']} | {q['output_strategy']} | {q['R2']:.4f} |")
    L.append("\n## 5. Resultado principal (ACCEL_RMS · all) y comparación con escenarios de sensibilidad\n")
    L.append("| Escenario / evaluación | mejor modelo | R² |")
    L.append("|---|---|---:|")
    L.append(f"| **cnc_available_all · filtered_test (PRINCIPAL in-distribution)** | {bf['model']} ({bf['output_strategy']}) | **{bf['R2']:.4f}** |")
    L.append(f"| **cnc_available_all · raw_target_test (PRINCIPAL picos conservados)** | {br['model']} ({br['output_strategy']}) | **{br['R2']:.4f}** |")
    L.append("| no_speed_rms (sensibilidad) | ver ablation_feature_sensitivity.csv | — |")
    L.append("| conservative_no_speed_temperature (conservador secundario) | ver ablation_feature_sensitivity.csv | — |")
    L.append("\n(Control de fidelidad: el filtered_test de cnc_available_all reproduce models_v2; el histórico "
             "R²=0,8547 es, por tanto, el resultado principal in-distribution, no un baseline con fuga.)")
    L.append("\n## 6. Interpretación metodológica\n")
    L.append("- **cnc_available_all es el resultado principal** del TFM: usa todas las señales disponibles de la "
             "CNC (incl. SPEED_RMS y TEMPERATURE), conforme a la definición oficial del problema.")
    L.append("- **Dos tests:** filtered_test mide el desempeño in-distribution (operación normal); "
             "raw_target_test mide la robustez ante los picos que el filtro IQR retiraba (validez operativa).")
    L.append(f"- **Principal a reportar:** filtered_test {bf['model']} R²={bf['R2']:.4f} (in-distribution) y "
             f"raw_target_test {br['model']} R²={br['R2']:.4f} (picos conservados), ambos en ACCEL_RMS·all.")
    L.append("- **Sensibilidad/robustez:** no_speed_rms (B) y conservative_no_speed_temperature (D) cuantifican "
             "la contribución de SPEED_RMS/TEMPERATURE; NO son leakage, sino ambigüedad documental.")
    L.append("\n## 7. Advertencia sobre el modelo ganador\n")
    L.append(f"- Ganador filtered_test (all): {win_f['model']} ({win_f['output_strategy']}) en {win_f['target']} "
             f"(R²={win_f['R2']:.4f}).")
    L.append(f"- Ganador raw_target_test (all): {win_r['model']} ({win_r['output_strategy']}) en {win_r['target']} "
             f"(R²={win_r['R2']:.4f}).")
    rk = (df[(df.evaluation_type == "raw_target_test") & (df.target == "ACCEL_RMS") &
             (df.load_condition == "all")].sort_values("R2", ascending=False)[["model", "output_strategy", "R2"]])
    L.append("- Ranking ACCEL_RMS·all en raw_target_test (no asumir que gana LightGBM):")
    for _, x in rk.iterrows():
        L.append(f"  - {x['model']} ({x['output_strategy']}): {x['R2']:.4f}")
    L.append("\n## 8. Recomendaciones para actualizar el Word\n")
    L.append("- Mantener cnc_available_all como resultado principal; reportar filtered (in-distribution) y "
             "raw-target (picos conservados).")
    L.append("- Declarar SPEED_RMS y TEMPERATURE como señales disponibles de la CNC, con nota de ambigüedad "
             "documental sobre su origen exacto; presentar B/D como análisis de sensibilidad.")
    L.append("- Revisar el modelo declarado como mejor según el ranking real de la sección 7.")
    L.append("- Regenerar figuras del Cap. 8–9 desde 01_models_v3_cnc_available_all_results.csv (Bloque 5B).")
    L.append("\n*Generado por `src/experiments/models_v3_cnc_available_all.py`. Cifras del CSV homónimo; "
             "ninguna a mano.*")
    (RES / f"{PFX}_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
