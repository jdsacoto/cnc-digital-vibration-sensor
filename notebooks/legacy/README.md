# notebooks/legacy/ — Notebooks originales pre-refactor

## Propósito

Esta carpeta conserva los notebooks originales del proyecto tal como existían antes de la Fase 3 de refactor. **Son evidencia histórica únicamente. No constituyen el pipeline reproducible del proyecto.**

Para reproducir el proyecto use exclusivamente los notebooks en `notebooks/` (un directorio más arriba):

- `notebooks/read_data_v2.ipynb` — pipeline de datos (genera `df_train.csv`, `df_test.csv`, `filter_metadata.json`).
- `notebooks/models_v2.ipynb` — registro de modelos + métricas (genera `models_v2_results.csv`).
- `notebooks/results_figures.ipynb` — figuras y tablas del proyecto (genera `reports/figures/*.png` y `reports/results/*.csv`).

## Contenido

| Notebook | Rol original | Estado |
|---|---|---|
| `read_data.ipynb` | Pipeline original de lectura, filtrado y preparación de datos (versión pre-refactor con leakage metodológico documentado en `METHODOLOGY_VERIFICATION.md`) | Congelado |
| `Modelos.ipynb` | Evaluación de modelos pre-entrenados (RandomForest, XGBoost en variantes single y multi-output, con y sin filtro SPINDLE_LOAD) | Congelado |
| `Modelos2.ipynb` | Evaluación masiva con 6 familias de modelos × 8 escenarios | Congelado |
| `Creacion_modelos.ipynb` | Entrenamiento y serialización de modelos a `Modelos/`, `Modelos2/`, `Modelos3/` | Congelado |
| `convert_pdf_2_images.ipynb` | Utility para conversión PDF → imagen (sin rol en el pipeline de modelado) | Congelado |

## Por qué los paths internos ya no resuelven

Los notebooks legacy contienen paths relativos al CWD original del proyecto (`./Data/...`, `./Modelos/...`, `./df_final.csv`, `./df_filtered_forest._without_cathegorical.csv`, etc.). Estos paths apuntaban a la raíz del repositorio cuando los notebooks vivían allí.

Tras el movimiento a `notebooks/legacy/` en el Sub-paso 3.5 (commit con mensaje `chore: mover notebooks a notebooks/ y notebooks/legacy/`), el CWD efectivo al ejecutar uno de estos notebooks pasaría a ser `notebooks/legacy/`. Los paths relativos apuntarían entonces a `notebooks/legacy/Data/...`, `notebooks/legacy/Modelos/...`, etc., que **no existen**.

**Esta ruptura es intencional y aceptada.** La regla operativa es: estos notebooks no se ejecutan; sólo se leen como referencia histórica.

## Si necesita re-ejecutar uno de estos notebooks

No es el flujo soportado. Si tuviese una razón académica concreta (por ejemplo, regenerar una figura citable del Modelos2.ipynb original), las opciones son:

1. **Recomendada:** copiar el notebook a la raíz del repositorio temporalmente, ejecutarlo allí desde un entorno con las dependencias adecuadas, y descartar la copia tras obtener el artefacto necesario. No commitear la copia.
2. **Alternativa:** crear un fork del notebook con paths absolutos (`<REPO_ROOT>/Data/...`) en una rama experimental separada. No fusionar a `main`.

En ambos casos, los resultados obtenidos están sujetos a las advertencias metodológicas documentadas en `METHODOLOGY_VERIFICATION.md`: split aleatorio sobre serie temporal 5 Hz, outliers/OHE/correlación pre-split, `StandardScaler.fit_transform` antes del split en algunas variantes, etc. Estos defectos son la razón por la que el pipeline reproducible canónico vive en `notebooks/*_v2.ipynb`.

## Historial Git

Los renames de raíz a `notebooks/legacy/` se hicieron con `git mv` para preservar la historia. Use `git log --follow notebooks/legacy/<archivo>.ipynb` para ver commits anteriores al movimiento.
