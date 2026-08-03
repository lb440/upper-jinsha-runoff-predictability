from sklearn.ensemble import HistGradientBoostingRegressor
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from baseline_shared_utils import (
    FEATURE_COLS,
    build_supervised_splits,
    calculate_metrics,
    calculate_nse,
    ensure_dir,
    flatten_and_scale_splits,
    plot_optimization_history,
    prefixed_name,
    save_json,
    save_pickle,
    save_prediction_outputs,
    summarize_split_sizes,
)

XGBOOST_SEARCH_SPACE_VERSION = "strong_regularization_v1"


def _can_import_xgboost():
    try:
        from xgboost import XGBRegressor  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_backend(requested_backend):
    if requested_backend == "auto":
        return "xgboost" if _can_import_xgboost() else "hist_gbdt"
    if requested_backend == "xgboost" and not _can_import_xgboost():
        raise ImportError(
            "xgboost is not installed in the current Python environment. "
            "Use --backend hist_gbdt for smoke tests or install xgboost to run the real baseline."
        )
    if requested_backend not in ("xgboost", "hist_gbdt"):
        raise ValueError(f"Unsupported backend: {requested_backend}")
    return requested_backend


def resolve_compute_device(requested_device, backend):
    if backend != "xgboost":
        return "cpu"
    if requested_device == "auto":
        return "cuda"
    if requested_device not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported compute_device: {requested_device}")
    return requested_device


def build_boosting_model(params, backend, random_seed, compute_device="auto"):
    if backend == "xgboost":
        from xgboost import XGBRegressor
        resolved_compute_device = resolve_compute_device(
            params.get("resolved_compute_device", compute_device),
            backend,
        )
        if resolved_compute_device == "cuda":
            tree_method = "gpu_hist"
            predictor = "gpu_predictor"
            extra_kwargs = {"gpu_id": 0}
        else:
            tree_method = "hist"
            predictor = "auto"
            extra_kwargs = {}

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            colsample_bylevel=float(params.get("colsample_bylevel", 0.7)),
            colsample_bynode=float(params.get("colsample_bynode", 0.7)),
            min_child_weight=float(params["min_child_weight"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            gamma=float(params["gamma"]),
            random_state=random_seed,
            n_jobs=-1,
            tree_method=tree_method,
            predictor=predictor,
            early_stopping_rounds=50,
            **extra_kwargs,
        )

    return HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        max_bins=int(params["max_bins"]),
        random_state=random_seed,
    )


def fit_boosting_model(model, backend, x_train, y_train, x_val, y_val):
    if backend == "xgboost":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        return
    model.fit(x_train, y_train)


def _suggest_boosting_params(trial, backend):
    if backend == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 8e-2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 0.8),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 0.8),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 30.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1.0, 10.0),
        }

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80, step=5),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }


def run_xgboost_experiment(
    kim,
    horizon,
    output_dir,
    params,
    feature_cols=None,
    backend="auto",
    prefix="",
    group_name=None,
    random_seed=222,
    compute_device="auto",
):
    feature_cols = list(feature_cols or FEATURE_COLS)
    resolved_backend = resolve_backend(params.get("resolved_backend", backend))
    resolved_compute_device = resolve_compute_device(
        params.get("resolved_compute_device", compute_device),
        resolved_backend,
    )
    splits = build_supervised_splits(feature_cols, kim=kim, horizon=horizon)
    scaled = flatten_and_scale_splits(splits)
    output_dir = ensure_dir(output_dir)

    model = build_boosting_model(
        params,
        resolved_backend,
        random_seed=random_seed,
        compute_device=resolved_compute_device,
    )
    fit_boosting_model(
        model,
        resolved_backend,
        scaled["X_train"],
        splits.y_train.flatten(),
        scaled["X_val"],
        splits.y_val.flatten(),
    )

    train_true = splits.y_train.flatten()
    val_true = splits.y_val.flatten()
    test_true = splits.y_test.flatten()
    train_pred = model.predict(scaled["X_train"]).clip(min=0.0)
    val_pred = model.predict(scaled["X_val"]).clip(min=0.0)
    test_pred = model.predict(scaled["X_test"]).clip(min=0.0)

    train_metrics = calculate_metrics(train_true, train_pred)
    val_metrics = calculate_metrics(val_true, val_pred)
    test_metrics = calculate_metrics(test_true, test_pred)

    metadata = {
        "model_name": "XGBoost",
        "resolved_backend": resolved_backend,
        "requested_backend": backend,
        "compute_device": resolved_compute_device,
        "group_name": group_name,
        "random_seed": random_seed,
        "kim": kim,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "params": params,
        "search_space_version": params.get("search_space_version", XGBOOST_SEARCH_SPACE_VERSION),
    }
    metadata.update(summarize_split_sizes(splits))

    save_pickle(
        output_dir / prefixed_name(prefix, "xgboost_model.pkl"),
        {
            "model": model,
            "params": params,
            "resolved_backend": resolved_backend,
            "compute_device": resolved_compute_device,
            "feature_cols": feature_cols,
            "x_scaler": scaled["x_scaler"],
        },
    )
    save_json(output_dir / prefixed_name(prefix, "xgboost_metadata.json"), metadata)
    save_prediction_outputs(
        output_dir=output_dir,
        prefix=prefix,
        run_label=f"XGBoost[{resolved_backend}]",
        horizon=horizon,
        time_test=splits.time_test,
        y_test_true=test_true,
        y_test_pred=test_pred,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        extra_sections=[("Model Metadata", metadata)],
    )

    return {
        "metadata": metadata,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_xgboost_bo(
    kim,
    horizon,
    output_dir,
    feature_cols=None,
    backend="auto",
    n_trials=30,
    random_seed=222,
    compute_device="auto",
):
    feature_cols = list(feature_cols or FEATURE_COLS)
    resolved_backend = resolve_backend(backend)
    resolved_compute_device = resolve_compute_device(compute_device, resolved_backend)
    splits = build_supervised_splits(feature_cols, kim=kim, horizon=horizon)
    scaled = flatten_and_scale_splits(splits)
    output_dir = ensure_dir(output_dir)

    def objective(trial):
        params = _suggest_boosting_params(trial, resolved_backend)
        model = build_boosting_model(
            params,
            resolved_backend,
            random_seed=random_seed,
            compute_device=resolved_compute_device,
        )
        fit_boosting_model(
            model,
            resolved_backend,
            scaled["X_train"],
            splits.y_train.flatten(),
            scaled["X_val"],
            splits.y_val.flatten(),
        )
        val_pred = model.predict(scaled["X_val"]).clip(min=0.0)
        return calculate_nse(splits.y_val.flatten(), val_pred)

    sampler = TPESampler(seed=random_seed)
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    params_json = dict(best_trial.params)
    params_json.update(
        {
            "best_nse": float(best_trial.value),
            "kim": kim,
            "horizon": horizon,
            "n_trials": n_trials,
            "random_seed": random_seed,
            "requested_backend": backend,
            "resolved_backend": resolved_backend,
            "requested_compute_device": compute_device,
            "resolved_compute_device": resolved_compute_device,
            "feature_cols": feature_cols,
            "search_space_version": XGBOOST_SEARCH_SPACE_VERSION,
        }
    )
    params_json.update(summarize_split_sizes(splits))

    save_json(output_dir / "xgboost_best_params.json", params_json)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    plot_optimization_history(
        output_dir / "optimization_history.png",
        trials_df=trials_df,
        title=f"XGBoost Validation NSE [{resolved_backend}]",
        best_value=best_trial.value,
    )
    return params_json
