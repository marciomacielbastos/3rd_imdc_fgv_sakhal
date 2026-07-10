# IMDC 2026 Forecast Repository: FGV Sakhal

This repository contains the FGV Sakhal validation-phase submission for IMDC 2026. The model is a Bayesian hierarchical dengue forecaster implemented with JAX and NumPyro.

## Team and Contributors

- Team name: Sakhal
- Institution: Fundacao Getulio Vargas (FGV)
- Main contact: Marcio, FGV
- Contributions: Marcio, FGV, handled preprocessing, model development, forecast generation, validation exports, and repository maintenance.

## Repository Structure

```text
.
+-- dengue_hardpulse/
|   +-- model.py
|   +-- seasonal.py
|   +-- inference.py
|   +-- plot_forecast.py
|   +-- run_model.py
+-- scripts/
|   +-- build_dengue_pulse_inputs.py
|   +-- 7_prepare_tensors.py
|   +-- merge_small_city_units.py
|   +-- submit_mosqlimate_predictions.py
+-- pyproject.toml
+-- README.md
```

Generated data, posterior samples, plots, and exported forecasts are written under `outputs/` and are not committed.

## Libraries and Dependencies

Python 3.10 or 3.11 is required. Dependencies are listed in `pyproject.toml` and include JAX, NumPyro, NumPy, pandas, SciPy, scikit-learn, pyarrow, matplotlib, epiweeks, mosqlient, python-dotenv, and Jupyter.

```bash
poetry install
```

## Data and Variables

Inputs are the official IMDC/Mosqlimate challenge files and static municipal covariates:

- `data/dengue.csv.gz`: weekly probable dengue cases and train/target flags.
- `data/datasus_population_2001_2025.csv.gz`: annual municipal population, interpolated to weekly exposure.
- `outputs/geospatial_results/built_stats_municipios.parquet`: centroid and built-area variables.
- `outputs/ifdm_results/ifdm_weekly_2002_2025.parquet`: supplemental socioeconomic covariates.
- `outputs/weather_results/weather_geo_koppen.parquet`: Koppen class and altitude.

`scripts/build_dengue_pulse_inputs.py` joins the files, builds a complete municipality-week panel, interpolates population, adds log density, optionally merges very small municipalities, and calls `scripts/7_prepare_tensors.py`. The tensor builder creates `[municipality, epidemic season, week]` arrays on an EW41-based 52-week season. EW53 is folded into the final pre-January slot.

## Model Training Process

`dengue_hardpulse/model.py` defines a Negative Binomial observation model. Expected cases combine a state incidence baseline, city deviations, a wrapped-Gaussian seasonal pulse, national year effects, state and city amplitude terms, recent-burden momentum, and immunity memory from previous observed training seasons.

Single split run:

```bash
python scripts/build_dengue_pulse_inputs.py --split 4 --merge-small-cities
python dengue_hardpulse/run_model.py --data-dir outputs/preprocessed_data_pulse --out-dir outputs/results_hardpulse --n-forecast-draws 200
```

Sequential validation run:

```bash
python scripts/build_dengue_pulse_inputs.py --all-splits --include-previous-target --merge-small-cities --challenge-work-dir outputs/challenge_hardpulse
python dengue_hardpulse/run_model.py --challenge-sequential --challenge-work-dir outputs/challenge_hardpulse --challenge-export-dir outputs/challenge_hardpulse/mandatory_state_forecasts --n-forecast-draws 200
```

The workflow fits a representative subset with SVI, optionally refines global parameters with NUTS, fits all municipalities with SVI, and generates posterior predictive draws in batches. Later validation splits can warm-start from the previous split's all-city median parameters.

## EW25 Data Usage Restriction

Operational forecasts from EW41 of the current year through EW40 of the next year must train only on rows available through EW25 of the current year. The preprocessing uses the official train/test flags. `scripts/7_prepare_tensors.py` creates `Y_train`, `Y_test`, `mask_train`, and `mask_test`; only `mask_train` observations enter the likelihood. Target-period cases remain masked and are used only for indexing and export. Historical burden features are computed from `Y_train` only.

Static geography, Koppen class, built area, and population exposure do not use future dengue observations. Sequential validation may add the immediately previous released target to the next training set; the final operational forecast should not use that option unless those observations are officially available by EW25.

## Prediction Intervals

`dengue_hardpulse/inference.py` samples posterior predictive reported-case draws. Mandatory state forecasts sum municipal draws within each UF before quantiles are computed. Exported intervals are the 25/75, 10/90, 5/95, and 2.5/97.5 percentiles, with the median as the 50th percentile. `scripts/submit_mosqlimate_predictions.py` validates non-negative, nested intervals before upload.

## References

No DOI is available for this validation-phase repository.
