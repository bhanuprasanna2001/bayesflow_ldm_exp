import json
import os
import numpy as np
import bayesflow as bf

from . import config as C


def mode_coverage(all_samples):
    rows = {}
    for name, s in all_samples.items():
        mode1 = float((s[:, 0] + s[:, 1] < 0).mean())
        rows[name] = {"mode_1": mode1, "mode_2": 1.0 - mode1}
    return rows


def training_summary(histories):
    rows = {}
    for name, h in histories.items():
        rows[name] = {
            "final_train": float(h.history["loss"][-1]),
            "final_val": float(h.history["val_loss"][-1]),
            "best_val": float(min(h.history["val_loss"])),
        }
    return rows


def calibration(sbc_samples, targets):
    rms = bf.diagnostics.metrics.root_mean_squared_error(
        estimates=sbc_samples, targets=targets
    )
    contraction = bf.diagnostics.metrics.posterior_contraction(
        estimates=sbc_samples, targets=targets
    )
    z = bf.diagnostics.metrics.posterior_z_score(
        estimates=sbc_samples, targets=targets
    )
    cal_err = bf.diagnostics.metrics.calibration_error(
        estimates=sbc_samples, targets=targets
    )
    return {
        "rmse": rms["values"].tolist(),
        "posterior_contraction": contraction["values"].tolist(),
        "posterior_z_score_mean": float(np.mean(z["values"])),
        "calibration_error": cal_err["values"].tolist(),
        "variable_names": rms.get("variable_names"),
    }


def save_json(obj, name):
    path = os.path.join(C.OUTPUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def print_summary(training, mode_cov, calibration_per_model):
    print(f"\n{'Model':<20} {'Train':>10} {'Val':>10} {'BestVal':>10} {'Mode1':>8} {'Mode2':>8}")
    print("-" * 70)
    for name in training:
        t = training[name]
        m = mode_cov[name]
        print(f"{name:<20} {t['final_train']:>10.4f} {t['final_val']:>10.4f} "
              f"{t['best_val']:>10.4f} {m['mode_1']:>8.1%} {m['mode_2']:>8.1%}")
    print("\nCalibration:")
    for name, c in calibration_per_model.items():
        print(f"  {name}: RMSE={c['rmse']}  CalErr={c['calibration_error']}  "
              f"z̄={c['posterior_z_score_mean']:.3f}")
