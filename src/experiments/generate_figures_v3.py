"""Bloque 5B — Figuras del Cap. 8–9 desde el registro v3 (cnc_available_all).

Solo lectura de CSV; sin reentrenar; matplotlib limpio (sin seaborn). No
sobrescribe figuras antiguas (nombres con sufijo _v3_cnc_available_all).

Fuentes:
  - reports/results/01_models_v3_cnc_available_all_results.csv  (figuras 1–6)
  - reports/results/ablation_feature_sensitivity.csv            (figura 7)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "reports/results"
FIG = REPO / "reports/figures"
SRC_MAIN = RES / "01_models_v3_cnc_available_all_results.csv"
SRC_SENS = RES / "ablation_feature_sensitivity.csv"

TARGETS = ["ACCEL_PEAK", "ACCEL_RMS", "ACCEL_RMS_FREQ"]
FAM_COLOR = {"RandomForest": "#4c72b0", "XGBoost": "#dd8452", "LightGBM": "#55a868",
             "CatBoost": "#c44e52", "MLPRegressor": "#8172b3"}
C_FILT, C_RAW = "#4c72b0", "#dd8452"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

df = pd.read_csv(SRC_MAIN)


def best_row(ev, target, lc="all"):
    q = df[(df.evaluation_type == ev) & (df.target == target) & (df.load_condition == lc)]
    return q.loc[q["R2"].idxmax()]


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Guardada:", p.name)


# --- 1 & 2: Top-10 por R² (filtered / raw) ------------------------------------
def fig_top10(ev, fname, title):
    t = df[df.evaluation_type == ev].sort_values("R2", ascending=False).head(10).iloc[::-1]
    labels = [f"{r.model}·{r.output_strategy}\n{r.target}·{r.load_condition}" for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(t)), t["R2"], color=[FAM_COLOR[m] for m in t["model"]])
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("R² (test temporal)")
    ax.set_xlim(0, 0.92)
    ax.set_title(title)
    for i, v in enumerate(t["R2"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=7.5)
    save(fig, fname)


fig_top10("filtered_test", "fig_top10_r2_filtered_v3_cnc_available_all.png",
          "Top-10 por R² — filtered_test (in-distribution)\ncnc_available_all")
fig_top10("raw_target_test", "fig_top10_r2_raw_target_v3_cnc_available_all.png",
          "Top-10 por R² — raw_target_test (picos conservados)\ncnc_available_all")

# --- 3: filtered vs raw por target (all) --------------------------------------
import numpy as np
filt = [best_row("filtered_test", t)["R2"] for t in TARGETS]
raw = [best_row("raw_target_test", t)["R2"] for t in TARGETS]
x = np.arange(len(TARGETS))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.2, filt, 0.4, label="filtered_test (in-distribution)", color=C_FILT)
ax.bar(x + 0.2, raw, 0.4, label="raw_target_test (picos conservados)", color=C_RAW)
for i in range(len(TARGETS)):
    ax.text(x[i] - 0.2, filt[i] + 0.008, f"{filt[i]:.4f}", ha="center", fontsize=8)
    ax.text(x[i] + 0.2, raw[i] + 0.008, f"{raw[i]:.4f}", ha="center", fontsize=8)
    ax.annotate(f"{raw[i]-filt[i]:+.3f}", (x[i], min(filt[i], raw[i]) - 0.04),
                ha="center", fontsize=8, color="#a00")
ax.set_xticks(x); ax.set_xticklabels(TARGETS)
ax.set_ylabel("R² (mejor modelo por target)")
ax.set_ylim(0, 0.95)
ax.set_title("Caída moderada in-distribution → picos conservados\ncnc_available_all · condición all")
ax.legend(fontsize=8)
save(fig, "fig_filtered_vs_raw_by_target_v3_cnc_available_all.png")

# --- 4: mejor modelo por target en raw (all) ----------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
br = [best_row("raw_target_test", t) for t in TARGETS]
vals = [r["R2"] for r in br]
cols = [FAM_COLOR[r["model"]] for r in br]
bars = ax.bar(TARGETS, vals, color=cols, width=0.55)
for i, r in enumerate(br):
    ax.text(i, r["R2"] + 0.008, f"{r['model']}\n({r['output_strategy']})\nR²={r['R2']:.4f}",
            ha="center", fontsize=8)
ax.set_ylabel("R² (raw_target_test)")
ax.set_ylim(0, 0.95)
ax.set_title("Mejor modelo por target — raw_target_test\ncnc_available_all · condición all")
save(fig, "fig_best_model_by_target_raw_v3_cnc_available_all.png")

# --- 5: comparativa de familias en raw (all) ----------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
fams = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "MLPRegressor"]
xf = np.arange(len(fams))
w = 0.26
for j, t in enumerate(TARGETS):
    vals = []
    for m in fams:
        q = df[(df.evaluation_type == "raw_target_test") & (df.model == m) &
               (df.target == t) & (df.load_condition == "all")]
        vals.append(q["R2"].max())
    ax.bar(xf + (j - 1) * w, vals, w, label=t)
ax.set_xticks(xf); ax.set_xticklabels(fams, fontsize=9)
ax.set_ylabel("R² (raw_target_test, mejor output)")
ax.set_ylim(0, 0.9)
ax.set_title("Comparativa de familias de modelos — raw_target_test\ncnc_available_all · condición all")
ax.legend(fontsize=8)
save(fig, "fig_model_family_raw_all_v3_cnc_available_all.png")

# --- 6: dificultad por target -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(TARGETS))
ax.bar(x - 0.2, filt, 0.4, label="filtered_test", color=C_FILT)
ax.bar(x + 0.2, raw, 0.4, label="raw_target_test", color=C_RAW)
ax.axhspan(0.5663, 0.5838, color="#c44e52", alpha=0.12)
ax.annotate("ACCEL_RMS_FREQ:\ntecho ≈ 0,566–0,584", (2, 0.50), ha="center", fontsize=8, color="#a00")
ax.set_xticks(x); ax.set_xticklabels(TARGETS)
ax.set_ylabel("R² (mejor modelo por target)")
ax.set_ylim(0, 0.95)
ax.set_title("Dificultad predictiva por target\nFREQ muy por debajo de las magnitudes temporales")
ax.legend(fontsize=8)
save(fig, "fig_target_difficulty_v3_cnc_available_all.png")

# --- 7: sensibilidad de SPEED_RMS (ambigüedad documental) ---------------------
sens = pd.read_csv(SRC_SENS)
order = ["cnc_available_all", "no_speed_rms", "conservative_no_speed_temperature"]
labels = ["cnc_available_all\n(principal)", "no_speed_rms\n(sensibilidad)",
          "conservative_no_speed\n_temperature (secundario)"]
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = np.arange(len(order))
w = 0.38
for j, m in enumerate(["LightGBM", "XGBoost"]):
    vals = []
    for s in order:
        q = sens[(sens.scenario == s) & (sens.model == m) &
                 (sens.target == "ACCEL_RMS") & (sens.load_condition == "all")]
        vals.append(float(q["R2"].iloc[0]) if len(q) else np.nan)
    ax.bar(xs + (j - 0.5) * w, vals, w, label=m,
           color=[C_FILT, C_RAW][j])
    for i, v in enumerate(vals):
        ax.text(xs[i] + (j - 0.5) * w, v + 0.006, f"{v:.4f}", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("R² — ACCEL_RMS · all (test filtrado)")
ax.set_ylim(0, 0.92)
ax.set_title("Sensibilidad a SPEED_RMS / TEMPERATURE (ACCEL_RMS)\n"
             "Ambigüedad documental — análisis de sensibilidad, NO fuga demostrada")
ax.legend(fontsize=8)
save(fig, "fig_speed_rms_sensitivity_v3.png")

# --- índice markdown ----------------------------------------------------------
idx = [
    "# Índice de figuras v3 — cnc_available_all\n",
    "Todas generadas por `src/experiments/generate_figures_v3.py` (solo lectura de CSV, sin reentrenar). "
    "Escenario principal **cnc_available_all** (28 features, incluye SPEED_RMS y TEMPERATURE).\n",
    "| Figura | Fuente CSV | Qué muestra | Word | Frase de interpretación |",
    "|---|---|---|---|---|",
    "| `fig_top10_r2_filtered_v3_cnc_available_all.png` | 01_models_v3…results.csv (filtered_test) | "
    "Top-10 R² in-distribution | Cap. 8 | «Los boosters de árboles dominan el registro in-distribution; "
    "ACCEL_RMS lidera con R²≈0,85.» |",
    "| `fig_top10_r2_raw_target_v3_cnc_available_all.png` | 01_models_v3…results.csv (raw_target_test) | "
    "Top-10 R² con picos conservados | Cap. 8 | «Al conservar los picos, el orden se mantiene y los R² "
    "máximos rondan 0,82.» |",
    "| `fig_filtered_vs_raw_by_target_v3_cnc_available_all.png` | 01_models_v3…results.csv | "
    "Caída filtered→raw por target (all) | Cap. 8 / 9 | «La degradación al conservar los picos es moderada "
    "(−0,017 a −0,059): el sensor pierde precisión pero no se rompe.» |",
    "| `fig_best_model_by_target_raw_v3_cnc_available_all.png` | 01_models_v3…results.csv (raw) | "
    "Mejor modelo por target en raw | Cap. 8 | «XGBoost encabeza en magnitud (PEAK, RMS) y LightGBM en "
    "frecuencia, con diferencias marginales: empate técnico.» |",
    "| `fig_model_family_raw_all_v3_cnc_available_all.png` | 01_models_v3…results.csv (raw) | "
    "Comparativa de familias en raw (all) | Anexo / Cap. 8 | «Las cinco familias convergen en magnitud; "
    "el MLP queda por debajo y la frecuencia es difícil para todas.» |",
    "| `fig_target_difficulty_v3_cnc_available_all.png` | 01_models_v3…results.csv | "
    "Dificultad relativa de los 3 targets | Cap. 9 | «ACCEL_RMS_FREQ tiene un techo ≈0,57–0,58 muy por "
    "debajo de las magnitudes temporales: límite de la información disponible.» |",
    "| `fig_speed_rms_sensitivity_v3.png` | ablation_feature_sensitivity.csv | "
    "Sensibilidad a SPEED_RMS/TEMPERATURE | Cap. 9 / Anexo | «Retirar SPEED_RMS/TEMPERATURE reduce R² de "
    "forma controlada; es análisis de sensibilidad por ambigüedad documental, no fuga demostrada.» |",
    "\n**Uso recomendado:** Cap. 8 (Resultados) → figuras 1–4; Cap. 9 (Discusión) → figuras 3, 6 y 7; "
    "Anexo → figura 5 y duplicados de robustez.",
]
(FIG / "figures_v3_cnc_available_all_index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
print("Guardado: figures_v3_cnc_available_all_index.md")
print("\nTODAS LAS FIGURAS SALEN DE:", SRC_MAIN.name, "y", SRC_SENS.name)
