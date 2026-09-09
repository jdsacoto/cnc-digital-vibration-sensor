"""Bloque 1 — Ablación de features por sospecha de leakage (SPEED_RMS, TEMPERATURE).

Compara tres conjuntos de features sobre el mismo split temporal y los mismos
df_train.csv / df_test.csv, reutilizando la lógica de evaluación del baseline
(src.evaluation) para que la comparación sea justa:

  A) baseline_current : todas las features actuales (incluye SPEED_RMS y TEMPERATURE)
  B) no_speed_rms     : sin SPEED_RMS
  C) strict_no_ifm    : sin SPEED_RMS y sin TEMPERATURE

Modelos: LightGBM y XGBoost (single-output). Targets: ACCEL_PEAK, ACCEL_RMS,
ACCEL_RMS_FREQ. Condiciones: all, load, no_load.

No modifica datos ni la memoria. Solo escribe en reports/results/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.config import TARGET_VARS, RANDOM_STATE, LOAD_CONDITIONS
from src.evaluation import evaluate_single, make_single_registry

OUT_CSV = REPO / "reports/results/ablation_feature_leakage.csv"
OUT_MD = REPO / "reports/results/ablation_feature_leakage_summary.md"

MODELS_USED = ["LightGBM", "XGBoost"]

SCENARIOS = {
    "baseline_current": [],
    "no_speed_rms": ["SPEED_RMS"],
    "strict_no_ifm": ["SPEED_RMS", "TEMPERATURE"],
}


def main() -> None:
    df_train = pd.read_csv(REPO / "df_train.csv")
    df_test = pd.read_csv(REPO / "df_test.csv")
    full_features = [c for c in df_train.columns if c not in TARGET_VARS + ["DATE"]]
    registry = make_single_registry(use_gpu=False, random_state=RANDOM_STATE)

    rows = []
    for scenario, removed in SCENARIOS.items():
        feats = [c for c in full_features if c not in removed]
        for model_name in MODELS_USED:
            cfg = registry[model_name]
            for target in TARGET_VARS:
                for load_kind in LOAD_CONDITIONS:
                    r = evaluate_single(
                        cfg["factory"], model_name, target, load_kind,
                        df_train, df_test, feats, scale=cfg["scale"],
                    )
                    r["scenario"] = scenario
                    r["removed_features"] = ";".join(removed) if removed else "(none)"
                    r["n_features"] = len(feats)
                    rows.append(r)
                    print(f"{scenario:16s} {model_name:9s} {target:14s} {load_kind:8s} "
                          f"R2={r['R2']:.4f} MAE={r['MAE']:.2f} ({r['fit_s']:.1f}s)")

    df = pd.DataFrame(rows)

    # --- deltas vs baseline_current (mismo model+target+load) ---
    base = (df[df.scenario == "baseline_current"]
            .set_index(["model", "target", "load"])[["R2", "MAE"]])
    df["delta_R2_vs_baseline_current"] = df.apply(
        lambda x: x["R2"] - base.loc[(x["model"], x["target"], x["load"]), "R2"], axis=1)
    df["delta_MAE_vs_baseline_current"] = df.apply(
        lambda x: x["MAE"] - base.loc[(x["model"], x["target"], x["load"]), "MAE"], axis=1)

    cols = ["scenario", "removed_features", "model", "target", "load", "n_train", "n_test",
            "n_features", "MAE", "MSE", "RMSE", "R2", "fit_s",
            "delta_R2_vs_baseline_current", "delta_MAE_vs_baseline_current"]
    df = df[cols].rename(columns={"load": "load_condition"})
    df.to_csv(OUT_CSV, index=False)
    print("\nGuardado:", OUT_CSV, df.shape)

    write_summary(df)
    print("Guardado:", OUT_MD)


def write_summary(df: pd.DataFrame) -> None:
    def r2(s, m, t, lc):
        q = df[(df.scenario == s) & (df.model == m) & (df.target == t) & (df.load_condition == lc)]
        return float(q["R2"].iloc[0]) if len(q) else float("nan")

    L = []
    L.append("# Ablación de features por sospecha de leakage — resumen\n")
    L.append("Bloque 1 · LightGBM y XGBoost · single-output · misma partición temporal · "
             "`df_train.csv`/`df_test.csv` actuales.\n")
    L.append("Escenarios: **A** `baseline_current` (todas las features) · **B** `no_speed_rms` "
             "(sin SPEED_RMS) · **C** `strict_no_ifm` (sin SPEED_RMS ni TEMPERATURE).\n")

    # 1. Mejor modelo por escenario y target (condición all)
    L.append("\n## 1. Mejor modelo por escenario y target (condición `all`)\n")
    L.append("| Escenario | Target | Mejor modelo | R² | MAE |")
    L.append("|---|---|---|---:|---:|")
    for s in SCENARIOS:
        for t in TARGET_VARS:
            q = df[(df.scenario == s) & (df.target == t) & (df.load_condition == "all")]
            b = q.loc[q["R2"].idxmax()]
            L.append(f"| {s} | {t} | {b['model']} | {b['R2']:.4f} | {b['MAE']:.2f} |")

    # 2. Caída de R² al quitar SPEED_RMS (B vs A)
    L.append("\n## 2. Caída de R² al quitar SPEED_RMS (B − A)\n")
    L.append("| Modelo | Target | Carga | R² A | R² B | Δ R² |")
    L.append("|---|---|---|---:|---:|---:|")
    for m in MODELS_USED:
        for t in TARGET_VARS:
            for lc in LOAD_CONDITIONS:
                a, b = r2("baseline_current", m, t, lc), r2("no_speed_rms", m, t, lc)
                L.append(f"| {m} | {t} | {lc} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")

    # 3. Caída adicional al quitar TEMPERATURE (C vs B)
    L.append("\n## 3. Caída adicional de R² al quitar TEMPERATURE (C − B)\n")
    L.append("| Modelo | Target | Carga | R² B | R² C | Δ R² |")
    L.append("|---|---|---|---:|---:|---:|")
    for m in MODELS_USED:
        for t in TARGET_VARS:
            for lc in LOAD_CONDITIONS:
                b, c = r2("no_speed_rms", m, t, lc), r2("strict_no_ifm", m, t, lc)
                L.append(f"| {m} | {t} | {lc} | {b:.4f} | {c:.4f} | {c-b:+.4f} |")

    # cifras agregadas para la recomendación
    dB = df[df.scenario == "no_speed_rms"]["delta_R2_vs_baseline_current"]
    dC = df[df.scenario == "strict_no_ifm"]["delta_R2_vs_baseline_current"]
    worst_B = df[df.scenario == "no_speed_rms"].sort_values("delta_R2_vs_baseline_current").iloc[0]
    worst_C = df[df.scenario == "strict_no_ifm"].sort_values("delta_R2_vs_baseline_current").iloc[0]

    L.append("\n## 4. Lectura cuantitativa\n")
    L.append(f"- Quitar **SPEED_RMS** (B vs A): Δ R² medio = **{dB.mean():+.4f}**, "
             f"peor caso = **{worst_B['delta_R2_vs_baseline_current']:+.4f}** "
             f"({worst_B['model']}/{worst_B['target']}/{worst_B['load_condition']}).")
    L.append(f"- Quitar también **TEMPERATURE** (C vs A): Δ R² medio = **{dC.mean():+.4f}**, "
             f"peor caso = **{worst_C['delta_R2_vs_baseline_current']:+.4f}** "
             f"({worst_C['model']}/{worst_C['target']}/{worst_C['load_condition']}).")
    rms_all = (r2("baseline_current", "LightGBM", "ACCEL_RMS", "all"),
               r2("no_speed_rms", "LightGBM", "ACCEL_RMS", "all"),
               r2("strict_no_ifm", "LightGBM", "ACCEL_RMS", "all"))
    L.append(f"- Resultado estrella (LightGBM·ACCEL_RMS·all): A={rms_all[0]:.4f} → "
             f"B={rms_all[1]:.4f} → C={rms_all[2]:.4f}.")

    L.append("\n## 5. Recomendación metodológica\n")
    L.append("- **Criterio rector:** no es la correlación con el target, sino la *disponibilidad legítima* "
             "de la variable en un sensor virtual basado en señales internas del PLC.")
    L.append("- **SPEED_RMS:** el rango real de los datos (≈0.09–11) es incompatible con la velocidad de giro "
             "del husillo (`SPINDLE_ACTUAL_SPEED` ≈ 10 000 rpm) y compatible con la velocidad RMS de vibración "
             "del IFM (mm/s). Mientras su origen no se demuestre como señal interna del PLC, **debe excluirse "
             "del escenario principal**.")
    L.append("- **TEMPERATURE:** procede del módulo IFM (temperatura de la taladrina). No es vibración, pero "
             "tampoco se demuestra como señal interna del PLC. Para una postura estricta de pureza, el escenario "
             "principal debe ser **`strict_no_ifm`** (C).")
    L.append("- **Escenario principal propuesto:** **C `strict_no_ifm`** (sin SPEED_RMS ni TEMPERATURE).")
    L.append("- **Escenario aumentado / sensibilidad:** **A `baseline_current`**, declarado explícitamente como "
             "registro histórico con variables del IFM, no como resultado principal.")
    L.append("- **Qué debe decir la memoria:** (1) que el sensor virtual usa señales internas del PLC; "
             "(2) que SPEED_RMS y TEMPERATURE provienen del IFM y por eso se excluyen del experimento principal; "
             "(3) reportar el resultado principal con C y el delta frente a A como análisis de sensibilidad.")
    L.append("\n*Generado por `src/experiments/ablation_feature_leakage.py`. Las cifras provienen del CSV "
             "homónimo; ninguna está escrita a mano.*")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
