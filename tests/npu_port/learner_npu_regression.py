#!/usr/bin/env python3
"""Opt-in learner regression checks, with no device execution by default.

Run through tests/npu_port/run_python.sh. Supply --library explicitly so this
check never silently tests an installed wheel. --run-npu requires an NPU build
and an available device; it is deliberately separate from --run-cpu.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-npu", action="store_true")
    mode.add_argument("--run-cpu", action="store_true")
    args = parser.parse_args()
    if not (args.run_npu or args.run_cpu):
        print("LEARNER_NPU_SKIP: device execution requires --run-npu; CPU checks require --run-cpu")
        return
    if args.library is None or not args.library.resolve().is_file():
        parser.error("an existing --library is required for opt-in execution")

    root = Path(__file__).resolve().parents[2]
    # Preload only the package's library locator, then override it for this process.
    # This avoids modifying the checkout or the user's Python/library environment.
    spec = importlib.util.spec_from_file_location(
        "xgboost.libpath", root / "python-package/xgboost/libpath.py"
    )
    assert spec is not None and spec.loader is not None
    libpath = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(libpath)
    library = args.library.resolve()
    libpath.find_lib_path = lambda: [str(library)]
    sys.modules["xgboost.libpath"] = libpath
    sys.path.insert(0, str(root / "python-package"))

    import numpy as np
    import xgboost as xgb

    assert Path(xgb.core._LIB._name).resolve() == library
    device = "npu:0" if args.run_npu else "cpu"
    if args.run_npu:
        assert xgb.build_info().get("USE_NPU") is True, xgb.build_info()
    features = np.arange(96, dtype=np.float32).reshape(24, 4) / 32
    labels = (features[:, 0] * 0.75 - 0.25).astype(np.float32)
    weights = np.arange(24, dtype=np.float32) % 4  # Includes zero-weight rows.
    data = xgb.DMatrix(features, label=labels, weight=weights, nthread=2)
    params = dict(device=device, tree_method="hist", objective="reg:squarederror",
                  nthread=2, seed=20261017, max_depth=2, max_bin=8,
                  eval_metric=["rmse", "mae"])

    # eta=0 isolates official weighted-mean initialization from tree contributions.
    intercept = xgb.train(params | dict(eta=0), data, num_boost_round=1)
    expected = np.average(labels.astype(np.float64), weights=weights)
    np.testing.assert_allclose(intercept.predict(data), expected, rtol=1e-6, atol=1e-6)
    history = {}
    model = xgb.train(params, data, num_boost_round=3, evals=[(data, "train")],
                      evals_result=history, verbose_eval=False)
    prediction = model.predict(data)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(model.predict(data, output_margin=True), prediction)
    # Public API training=true must export resident predictions as well.
    np.testing.assert_array_equal(model.predict(data, training=True), prediction)
    rmse = np.sqrt(np.average((prediction - labels) ** 2, weights=weights))
    mae = np.average(np.abs(prediction - labels), weights=weights)
    np.testing.assert_allclose(history["train"]["rmse"][-1], rmse, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(history["train"]["mae"][-1], mae, rtol=1e-5, atol=1e-6)

    restored = xgb.Booster(model_file=model.save_raw())
    restored.set_param(dict(device=device, nthread=2))
    np.testing.assert_array_equal(restored.predict(data), prediction)
    restored.load_config(model.save_config())
    np.testing.assert_array_equal(restored.predict(data), prediction)
    cpu_copy = xgb.Booster(model_file=model.save_raw())
    cpu_copy.set_param(dict(device="cpu", nthread=2))
    np.testing.assert_allclose(cpu_copy.predict(data), prediction, rtol=1e-5, atol=1e-6)

    def reject(call, expected_text):
        try:
            call()
        except xgb.core.XGBoostError as error:
            assert expected_text in str(error), str(error)
        else:
            raise AssertionError(f"expected rejection containing {expected_text!r}")

    all_zero = xgb.DMatrix(features, label=labels, weight=np.zeros_like(weights), nthread=2)
    reject(lambda: xgb.train(params, all_zero, 1), "weights must contain at least one non-zero")
    if args.run_npu:
        for override, expected_text in [
            ({"objective": "binary:logistic"}, "only objective=reg:squarederror"),
            ({"booster": "gblinear"}, "only booster=gbtree"),
            ({"booster": "dart"}, "only booster=gbtree"),
            ({"num_target": 2}, "only num_target=1"),
            ({"scale_pos_weight": 2}, "requires scale_pos_weight=1"),
            ({"multi_strategy": "multi_output_tree"}, "reduced gradients"),
            ({"eval_metric": "logloss"}, "explicit CPU rmse and mae"),
        ]:
            reject(lambda override=override: xgb.Booster(params=params | override,
                                                        cache=[data]).save_config(), expected_text)
        reject(lambda: model.boost(data, iteration=3, grad=np.ones_like(labels),
                                   hess=np.ones_like(labels)),
               "requires native_gradient")
        reject(lambda: model.predict(data, pred_leaf=True), "only value/margin prediction")
    else:
        default_params = {key: value for key, value in params.items() if key != "device"}
        default_model = xgb.train(default_params, data, num_boost_round=3)
        np.testing.assert_array_equal(default_model.predict(data), prediction)
    print(json.dumps(dict(status="passed", device=device, library=str(library),
                          checks="weighted_intercept,metrics,training_export,model_config_roundtrip,guards")))


if __name__ == "__main__":
    main()