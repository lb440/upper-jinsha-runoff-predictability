# Upper Jinsha runoff predictability

This repository contains the code and non-restricted supporting materials for the manuscript *Lead-time-dependent shifts in predictive information and lag structure for daily runoff forecasting: Evidence from the Upper Jinsha River*.

## Scope

The experiments evaluate daily runoff predictability conditioned on antecedent upstream-flow and hydro-meteorological information. Future meteorological forecasts are not used. The repository includes scripts for ERA5-Land preparation, 90-day sample construction within model scripts, training seven model families, hyperparameter records, fixed-architecture SHAP attribution, seasonal baselines, input-factor combinations, cross-correlation analysis, and manuscript figures.

## Data availability

ERA5-Land variables are publicly available from the Copernicus Climate Data Store: https://doi.org/10.24381/cds.e2161bac. The observed runoff series at Zhimenda and Shigu originate from the *Hydrological Yearbook of the People's Republic of China* and cannot be redistributed under the applicable data-use restrictions. No restricted runoff observations, trained model weights, daily predictions, or individual-level output tables are included here.

The `data/` directory provides a field dictionary and a synthetic CSV for schema checks only. It cannot reproduce manuscript results. Users who lawfully obtain the runoff records should prepare a local file matching `data/data_dictionary.csv` and set `RUNOFF_DATA_PATH` to that path.

## Installation

Create an environment with Python 3.7 and install the pinned study packages:

```bash
python -m pip install -r requirements.txt
python scripts/make_synthetic_example.py
```

For CUDA execution, install the PyTorch 1.13.1 build appropriate to the local CUDA driver before installing the remaining packages.

## Reproduction workflow

1. Download ERA5-Land total precipitation, 2 m temperature, total evaporation, and snow depth from CDS with `src/data/download_era5_land.py`.
2. Supply local catchment and interval shapefiles plus a lawfully obtained runoff table. Set `RUNOFF_DATA_ROOT`, `BASIN_SHP_DIR`, `ERA5_PRECIP_DIR`, `ERA5_EVAP_DIR`, and `RUNOFF_DATA_PATH` as required.
3. Run `src/data/preprocess_era5_land.py` to derive daily precipitation and evaporation using 24-hour accumulations. Run `src/data/aggregate_daily_state_variables.py` to derive basin-average daily mean temperature and snow depth.
4. Run `src/data/construct_samples.py` to construct 90-day chronological windows for each lead time. The model scripts use the same sample definition internally.
5. Use the model scripts in `src/models/` with the 90-day input window, chronological split, and seed `222` defined in `config/study_config.yaml`.
6. Consult `config/selected_hyperparameters.csv` for the reported lead-specific optimized parameters. Use `run_fixed_lstm_transformer_shap_analysis.py` for the fixed-architecture attribution sensitivity analysis.
7. Run `run_fixed_architecture_ablation_7_15.py` for the 7- and 15-day fixed-architecture predictor-deletion experiment. Set `ABLATION_OUTPUT_ROOT` if the default output directory should be changed.
8. Run `run_lstm_transformer_factor_combinations.py`, `run_seasonal_baseline_control.py`, and `src/analysis/qz_shigu_cross_correlation.py` for the supporting analyses.
9. Use scripts in `src/figures/` to regenerate the manuscript figures from locally generated non-restricted summary tables.

## Key conventions

- Random seed: `222`.
- Input window: 90 days.
- Lead times: 1, 3, 7, and 15 days.
- Split: training through 2014-12-31; validation from 2015-01-01 to 2016-12-31; test from 2017-01-01 to 2020-12-31.
- ERA5-Land precipitation and evaporation: daily 24-hour accumulations; evaporation retains the ERA5-Land native sign convention.
- ERA5-Land temperature and snow depth: daily means.

## Repository layout

| Path | Content |
| --- | --- |
| `src/data/` | ERA5-Land download and corrected P/E preprocessing scripts |
| `src/models/` | Persistence, XGBoost, MLP, GRU, LSTM, Transformer, LSTM-Transformer, SHAP, and supporting analyses |
| `src/analysis/` | Qz-Shigu cross-correlation workflow |
| `src/figures/` | Publication figure scripts |
| `config/` | Study settings and selected hyperparameters |
| `data/` | Data dictionary, synthetic schema example, and non-restricted figure/table summary data |

## Citation

If you use this code, cite the associated manuscript once a DOI is assigned. A fixed, versioned archival release is prepared for manuscript submission; its DOI will be added here when the Zenodo record is registered.
