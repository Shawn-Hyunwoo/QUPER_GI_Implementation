"""Training loops for QuPer-GI (paper Algorithm 1)."""

from __future__ import annotations

from dataclasses import replace
from math import log2

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .ansatz import transfer_theta
from .configs import GIConfig
from .dsm_circuit import QuPerDSM
from .losses import total_loss
from .metrics import frobenius_mismatch, hidden_perm_overlap
from .projection import project_best

SUCCESS_TOL = 1.0e-9


def initialize_theta(num_params: int, seed: int) -> pnp.ndarray:
    """Initialize parameters uniformly in pi/2 +- 0.05 (Algorithm 1, line 1)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(
        low=np.pi / 2.0 - 0.05,
        high=np.pi / 2.0 + 0.05,
        size=(num_params,),
    )
    return pnp.array(theta, requires_grad=True)


def to_numpy_matrix(x) -> np.ndarray:
    """Convert a PennyLane/autograd object to a NumPy array."""
    return np.asarray(qml.math.toarray(x), dtype=np.float64)


def train_single_m(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_star: np.ndarray | None,
    cfg: GIConfig,
    theta0: np.ndarray | None = None,
) -> dict:
    """Train one QuPer-GI model for a fixed number of ancilla pairs.

    Parameters
    ----------
    theta0:
        Optional warm-start parameters (used by the ancilla schedule).

    Returns a dictionary containing the best projected solution found, plus
    `theta_final` (the parameters after the last optimizer step, used to
    warm-start the next ancilla stage).
    """
    model = QuPerDSM(
        num_vertices=cfg.num_vertices,
        ancillas=cfg.ancillas,
        ansatz=cfg.ansatz,
        device_name=cfg.device,
        diff_method=cfg.diff_method,
    )

    if theta0 is None:
        theta = initialize_theta(model.num_params, cfg.seed)
    else:
        if len(theta0) != model.num_params:
            raise ValueError("theta0 length does not match the model layout.")
        theta = pnp.array(np.asarray(theta0, dtype=np.float64), requires_grad=True)

    a_train = pnp.array(a_mat, requires_grad=False)
    b_train = pnp.array(b_mat, requires_grad=False)

    opt = qml.AdamOptimizer(stepsize=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps)
    rng = np.random.default_rng(cfg.seed + 10_000)

    best = {
        "value": float("inf"),
        "p": None,
        "theta": None,
        "step": None,
        "projection": None,
    }

    history: list[dict] = []

    def objective(current_theta):
        return total_loss(current_theta, model, a_train, b_train, cfg)

    print(model.summary())

    for step in range(cfg.steps):
        theta, loss_value = opt.step_and_cost(objective, theta)

        record = {
            "step": int(step),
            "train_loss": float(loss_value),
        }

        if step % cfg.projection_interval == 0 or step == cfg.steps - 1:
            p_hat = to_numpy_matrix(model(theta))

            p_proj, proj_value, proj_name = project_best(
                p_hat=p_hat,
                a_mat=a_mat,
                b_mat=b_mat,
                rng=rng,
                num_random=cfg.num_random_projections,
            )

            if proj_value < best["value"]:
                best.update(
                    {
                        "value": float(proj_value),
                        "p": p_proj,
                        "theta": np.asarray(theta, dtype=np.float64).copy(),
                        "step": int(step),
                        "projection": proj_name,
                    }
                )

            record["projected_loss"] = float(proj_value)
            record["projection"] = proj_name

            if p_star is not None and best["p"] is not None:
                record["hidden_overlap"] = hidden_perm_overlap(best["p"], p_star)

            if step % max(1, 10 * cfg.projection_interval) == 0 or step == cfg.steps - 1:
                print(
                    f"step={step:05d} "
                    f"train={float(loss_value):.6g} "
                    f"proj={proj_value:.6g} "
                    f"best={best['value']:.6g} "
                    f"via={proj_name}"
                )

            if best["value"] <= SUCCESS_TOL:
                print("Success: projected GI loss reached zero.")
                record["best_projected_loss"] = float(best["value"])
                history.append(record)
                break

        record["best_projected_loss"] = float(best["value"])
        history.append(record)

    best["history"] = history
    best["theta_final"] = np.asarray(theta, dtype=np.float64).copy()

    if best["p"] is not None:
        best["final_mismatch"] = frobenius_mismatch(a_mat, b_mat, best["p"])

    return best


def train_quper(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_star: np.ndarray | None,
    cfg: GIConfig,
) -> dict:
    """Full QuPer ancilla schedule (paper Algorithm 1).

    Trains m = 0, 1, ..., cfg.ancillas in sequence. When moving from m to m+1
    the trained parameters are transferred to the larger layout and the new
    gates are initialized at pi/8 (Algorithm 1, line 10). Returns the overall
    best result; per-stage summaries are stored under "stages" and each
    stage's history under "history_by_m".
    """
    q = int(log2(cfg.num_vertices))

    overall_best: dict | None = None
    theta_prev: np.ndarray | None = None
    stages: list[dict] = []
    history_by_m: dict[int, list[dict]] = {}

    for m in range(cfg.ancillas + 1):
        cfg_m = replace(cfg, ancillas=m)

        theta0 = None
        if theta_prev is not None:
            theta0 = transfer_theta(theta_prev, cfg.ansatz, q, m - 1, m)

        result = train_single_m(a_mat, b_mat, p_star, cfg_m, theta0=theta0)
        theta_prev = result["theta_final"]

        history_by_m[m] = result["history"]
        stages.append(
            {
                "ancillas": m,
                "value": result["value"],
                "step": result["step"],
                "projection": result["projection"],
            }
        )

        if overall_best is None or result["value"] < overall_best["value"]:
            overall_best = dict(result)
            overall_best["ancillas"] = m

        if overall_best["value"] <= SUCCESS_TOL:
            break

    assert overall_best is not None
    overall_best["stages"] = stages
    overall_best["history_by_m"] = history_by_m
    return overall_best
