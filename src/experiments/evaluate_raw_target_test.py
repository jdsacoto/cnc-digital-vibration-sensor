"""Bloque 3 — Evaluación del escenario strict_no_ifm sobre un test temporal
sin filtrar filas por los targets ACCEL_*.

Reconstruye el pipeline desde el H5 reutilizando src.preprocessing, de modo que:
  - el TRAIN se regenera idéntico al df_train.csv commiteado (control MD5),
  - el TEST FILTRADO se regenera idéntico al df_test.csv commiteado (control MD5),
  - el TEST RAW-TARGET se construye igual que el filtrado SALVO que se omiten los
    ACCEL_* del descarte por outliers (las filas con picos de vibración se conservan).
    Todos los demás filtros (features), el OHE y la poda por correlación —ajustados
    en train— se aplican igual, para aislar el efecto del filtro IQR sobre targets.

Escenario de features: strict_no_ifm (sin SPEED_RMS ni TEMPERATURE).
Modelos: LightGBM, XGBoost (single-output). Targets: ACCEL_*. Cargas: all/load/no_load.

No sobrescribe df_train.csv ni df_test.csv. Escribe:
  - df_test_raw_targets.csv
  - reports/results/raw_target_test_strict_no_ifm.csv
  - reports/results/raw_target_test_strict_no_ifm_summary.md
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.config import (
    TARGET_VARS, RANDOM_STATE, TOOL_VALIDATION, N_TEST_DAYS,
    CORR_THRESHOLD, OUTLIER_STRATEGIES, KNOWN_CATEGORICALS, LOAD_CONDITIONS,
)
from src.preprocessing import (
    load_h5_with_date, remove_consecutive_duplicates, split_by_date,
    fit_outlier_thresholds, apply_outlier_thresholds,
    decide_ohe_columns, fit_ohe, apply_ohe, fit_correlation_drop,
)
from src.evaluation import evaluate_single, make_single_registry

H5_PATH = REPO / "Data" / "upload_5hz_2020_6.h5"
EXCLUDE_IFM = ["SPEED_RMS", "TEMPERATURE"]
MODELS_USED = ["LightGBM", "XGBoost"]
OUT_TEST = REPO / "df_test_raw_targets.csv"
OUT_CSV = REPO / "reports/results/raw_target_test_strict_no_ifm.csv"
OUT_MD = REPO / "reports/results/raw_target_test_strict_no_ifm_summary.md"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def build():
    print("Cargando H5 y reconstruyendo pipeline…", flush=True)
    df = load_h5_with_date(H5_PATH)
    df = df[df["ACTIVE_TOOL_ACTUAL"] == TOOL_VALIDATION].copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("ACTIVE_TOOL")])

    # Filtro 2 — columnas constantes (read_data_v2 cell #4)
    const_cols = [c for c in df.columns
                  if c != "DATE" and df[c].dropna().nunique() <= 1]
    df = df.drop(columns=const_cols)

    # Filtro 3 — categóricas de alta cardinalidad (read_data_v2 cell #5)
    from src.config import HIGH_CARDINALITY_THRESHOLD
    protected = set(OUTLIER_STRATEGIES.keys()) | set(TARGET_VARS)
    drop_hc = []
    for col in df.columns:
        if col == "DATE" or col in protected:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        nu = s.nunique()
        is_int_like = pd.api.types.is_integer_dtype(s) or (
            pd.api.types.is_float_dtype(s) and (s % 1 == 0).all())
        is_object = pd.api.types.is_object_dtype(s)
        if (is_int_like or is_object) and HIGH_CARDINALITY_THRESHOLD < nu <= 200:
            drop_hc.append(col)
    df = df.drop(columns=drop_hc)

    df = remove_consecutive_duplicates(df)
    df = df.dropna().reset_index(drop=True)
    df_train, df_test, train_dates, test_dates = split_by_date(df, N_TEST_DAYS)

    thr = fit_outlier_thresholds(df_train, OUTLIER_STRATEGIES)
    thr_no_targets = {c: t for c, t in thr.items() if c not in TARGET_VARS}

    df_train_f, _ = apply_outlier_thresholds(df_train, thr)
    df_test_f, _ = apply_outlier_thresholds(df_test, thr)          # control (== df_test.csv)
    df_test_raw, _ = apply_outlier_thresholds(df_test, thr_no_targets)  # conserva picos

    ohe_cols = decide_ohe_columns(df_train_f, KNOWN_CATEGORICALS)
    ohe = fit_ohe(df_train_f, ohe_cols)
    df_train_f = apply_ohe(df_train_f, ohe, ohe_cols)
    df_test_f = apply_ohe(df_test_f, ohe, ohe_cols)
    df_test_raw = apply_ohe(df_test_raw, ohe, ohe_cols)

    corr_dropped, _ = fit_correlation_drop(
        df_train_f, CORR_THRESHOLD, exclude=set(["DATE"] + TARGET_VARS))
    df_train_f = df_train_f.drop(columns=corr_dropped)
    df_test_f = df_test_f.drop(columns=corr_dropped)
    df_test_raw = df_test_raw.drop(columns=corr_dropped)

    # alinear columnas del raw-target al orden del train
    df_test_raw = df_test_raw[list(df_train_f.columns)]
    df_test_f = df_test_f[list(df_train_f.columns)]
    return df_train_f, df_test_f, df_test_raw


def fidelity_check(df_train_f, df_test_f):
    df_train_f.to_csv(REPO / "_tmp_train_regen.csv", index=False)
    df_test_f.to_csv(REPO / "_tmp_test_regen.csv", index=False)
    tr_ok = md5(REPO / "_tmp_train_regen.csv") == md5(REPO / "df_train.csv")
    te_ok = md5(REPO / "_tmp_test_regen.csv") == md5(REPO / "df_test.csv")
    (REPO / "_tmp_train_regen.csv").unlink()
    (REPO / "_tmp_test_regen.csv").unlink()
    print(f"Fidelidad MD5 -> train: {'OK' if tr_ok else 'DIFIERE'} | "
          f"test filtrado: {'OK' if te_ok else 'DIFIERE'}")
    return tr_ok, te_ok


def main():
    df_train_f, df_test_f, df_test_raw = build()
    tr_ok, te_ok = fidelity_check(df_train_f, df_test_f)

    df_test_raw.to_csv(OUT_TEST, index=False)
    print(f"Guardado {OUT_TEST}: {df_test_raw.shape} "
          f"(filtrado: {df_test_f.shape[0]} filas, +{len(df_test_raw)-len(df_test_f)})")

    feats = [c for c in df_train_f.columns
             if c not in TARGET_VARS + ["DATE"] + EXCLUDE_IFM]
    print(f"strict_no_ifm features: {len(feats)}")
    registry = make_single_registry(use_gpu=False, random_state=RANDOM_STATE)

    rows = []
    for ev_type, test_df in [("filtered_test", df_test_f), ("raw_target_test", df_test_raw)]:
        for m in MODELS_USED:
            cfg = registry[m]
            for t in TARGET_VARS:
                for lc in LOAD_CONDITIONS:
                    r = evaluate_single(cfg["factory"], m, t, lc,
                                        df_train_f, test_df, feats, scale=cfg["scale"])
                    r["evaluation_type"] = ev_type
                    rows.append(r)
                    print(f"{ev_type:16s} {m:9s} {t:14s} {lc:8s} "
                          f"R2={r['R2']:.4f} MAE={r['MAE']:.2f} n_test={r['n_test']}")
    res = pd.DataFrame(rows)

    # tabla pivote para deltas raw vs filtered
    res["feature_scenario"] = "strict_no_ifm"
    res["removed_features"] = ";".join(EXCLUDE_IFM)
    base = (res[res.evaluation_type == "filtered_test"]
            .set_index(["model", "target", "load"])[["R2", "MAE", "n_test"]])
    rawn = (res[res.evaluation_type == "raw_target_test"]
            .set_index(["model", "target", "load"])["n_test"])

    def add(x):
        key = (x["model"], x["target"], x["load"])
        x["n_test_filtered"] = int(base.loc[key, "n_test"])
        x["n_test_raw_targets"] = int(rawn.loc[key])
        x["n_extra_test_rows"] = x["n_test_raw_targets"] - x["n_test_filtered"]
        x["delta_R2_vs_strict_filtered_test"] = x["R2"] - base.loc[key, "R2"]
        x["delta_MAE_vs_strict_filtered_test"] = x["MAE"] - base.loc[key, "MAE"]
        return x

    res = res.apply(add, axis=1)
    cols = ["evaluation_type", "feature_scenario", "removed_features", "model", "target",
            "load", "n_train", "n_test_filtered", "n_test_raw_targets", "n_extra_test_rows",
            "MAE", "MSE", "RMSE", "R2", "fit_s",
            "delta_R2_vs_strict_filtered_test", "delta_MAE_vs_strict_filtered_test"]
    res = res[cols].rename(columns={"load": "load_condition"})
    res.to_csv(OUT_CSV, index=False)
    print("Guardado:", OUT_CSV, res.shape)

    write_summary(res, df_test_f, df_test_raw, tr_ok, te_ok)
    print("Guardado:", OUT_MD)


def recovered_target_stats(df_test_f, df_test_raw):
    """Stats de los targets en las filas recuperadas (en raw, ausentes del filtrado)."""
    # las recuperadas son las que el filtro IQR de targets habría eliminado:
    thr = {}  # recomputar máscara de supervivencia por target con percentiles del propio test_f
    # Identificar recuperadas por anti-join sobre todas las columnas comunes no-target sería frágil;
    # en su lugar: una fila raw es "recuperada" si algún target queda fuera del rango observado en test_f.
    lo = {t: df_test_f[t].min() for t in TARGET_VARS}
    hi = {t: df_test_f[t].max() for t in TARGET_VARS}
    mask_in = np.ones(len(df_test_raw), dtype=bool)
    for t in TARGET_VARS:
        mask_in &= df_test_raw[t].between(lo[t], hi[t])
    rec = df_test_raw[~mask_in]
    stats = {}
    for t in TARGET_VARS:
        s = rec[t]
        stats[t] = dict(n=len(rec), min=s.min(), max=s.max(), mean=s.mean(),
                        std=s.std(), p95=np.percentile(s, 95), p99=np.percentile(s, 99))
    return rec, stats


def write_summary(res, df_test_f, df_test_raw, tr_ok, te_ok):
    rec, stats = recovered_target_stats(df_test_f, df_test_raw)
    n_extra = len(df_test_raw) - len(df_test_f)

    def get(ev, m, t, lc, col="R2"):
        q = res[(res.evaluation_type == ev) & (res.model == m) &
                (res.target == t) & (res.load_condition == lc)]
        return float(q[col].iloc[0])

    L = []
    L.append("# Test temporal sin filtrar targets — strict_no_ifm — resumen\n")
    L.append(f"Bloque 3 · features **strict_no_ifm** (sin SPEED_RMS ni TEMPERATURE) · "
             f"LightGBM/XGBoost · single-output.\n")
    L.append(f"**Control de fidelidad MD5:** train regenerado = "
             f"{'OK (idéntico a df_train.csv)' if tr_ok else 'DIFIERE'}; "
             f"test filtrado regenerado = {'OK (idéntico a df_test.csv)' if te_ok else 'DIFIERE'}.\n")
    L.append(f"**Filas de test:** filtrado = {len(df_test_f)} · raw-target = {len(df_test_raw)} · "
             f"recuperadas = **+{n_extra}** ({100*n_extra/len(df_test_f):.1f}% más).\n")

    L.append("\n## 1–2. R² y MAE: test filtrado vs test raw-target (condición `all`)\n")
    L.append("| Modelo | Target | R² filtrado | R² raw | Δ R² | MAE filtrado | MAE raw | Δ MAE |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS_USED:
        for t in TARGET_VARS:
            rf, rr = get("filtered_test", m, t, "all"), get("raw_target_test", m, t, "all")
            mf, mr = get("filtered_test", m, t, "all", "MAE"), get("raw_target_test", m, t, "all", "MAE")
            L.append(f"| {m} | {t} | {rf:.4f} | {rr:.4f} | {rr-rf:+.4f} | "
                     f"{mf:.2f} | {mr:.2f} | {mr-mf:+.2f} |")

    L.append("\n### Detalle por condición de carga (Δ R² raw − filtrado)\n")
    L.append("| Modelo | Target | all | load | no_load |")
    L.append("|---|---|---:|---:|---:|")
    for m in MODELS_USED:
        for t in TARGET_VARS:
            ds = [get("raw_target_test", m, t, lc) - get("filtered_test", m, t, lc)
                  for lc in ["all", "load", "no_load"]]
            L.append(f"| {m} | {t} | {ds[0]:+.4f} | {ds[1]:+.4f} | {ds[2]:+.4f} |")

    L.append(f"\n## 3. Filas recuperadas: {len(rec)} (las que el IQR de targets eliminaba del test)\n")
    L.append("\n## 4. Estadísticas de los targets en las filas recuperadas\n")
    L.append("| Target | n | min | max | media | std | p95 | p99 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in TARGET_VARS:
        s = stats[t]
        L.append(f"| {t} | {s['n']} | {s['min']:.1f} | {s['max']:.1f} | {s['mean']:.1f} | "
                 f"{s['std']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} |")
    L.append("\n(Comparar con el rango del test filtrado, p. ej. ACCEL_RMS filtrado "
             f"min={df_test_f['ACCEL_RMS'].min():.1f} max={df_test_f['ACCEL_RMS'].max():.1f}.)")

    L.append("\n## 5. Interpretación metodológica\n")
    L.append("- **Test filtrado (in-distribution):** mide el desempeño sobre el régimen de operación normal, "
             "excluyendo los picos de vibración que el filtro IQR retira. Es una cota optimista pero honesta "
             "del comportamiento en condiciones típicas.")
    L.append("- **Test raw-target (temporal con picos conservados):** más realista respecto a uso operativo, "
             "porque en producción no conoceríamos el target antes de predecirlo y los picos SÍ ocurren. La "
             "diferencia de R² mide la **sensibilidad del sensor ante eventos extremos**, no un fallo del "
             "modelo.")
    L.append("- **Resultado principal recomendado:** reportar AMBOS. El test raw-target debe ser la "
             "**evaluación principal de validez operativa**; el test filtrado, el **análisis in-distribution** "
             "(comparabilidad con el registro histórico).")
    L.append("- La memoria debe distinguir explícitamente *evaluación in-distribution* (filtrada) de "
             "*evaluación temporal con picos conservados* (raw-target), sin presentar una como sustituta de la "
             "otra. Si el R² cae en raw-target, es la degradación esperable ante extremos eliminados por el IQR.")
    L.append("\n*Generado por `src/experiments/evaluate_raw_target_test.py`. Cifras del CSV homónimo; "
             "ninguna escrita a mano.*")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
