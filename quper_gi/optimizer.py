"""Training loops for QuPer-GI (paper Algorithm 1)."""

from __future__ import annotations

from dataclasses import replace
from math import log2

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

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


def _step_with_grad(opt, objective, theta):
    """One Adam step that also reports the gradient norm (single evaluation).

    Replicates step_and_cost's internals (compute_grad + apply_grad) so the
    gradient is available for FOM logging without a second circuit evaluation.
    Falls back to plain step_and_cost (grad norm = nan) on API mismatch.
    """
    try:
        grad, forward = opt.compute_grad(objective, (theta,), {})
        new_args = opt.apply_grad(grad, (theta,))
        new_theta = new_args[0] if isinstance(new_args, (list, tuple)) else new_args
        if forward is None:
            forward = objective(theta)
        g = grad[0] if isinstance(grad, (list, tuple)) else grad
        gnorm = float(np.linalg.norm(np.asarray(qml.math.toarray(g)).ravel()))
        return new_theta, float(forward), gnorm
    except Exception:
        new_theta, forward = opt.step_and_cost(objective, theta)
        return new_theta, float(forward), float("nan")


def _step_with_spsa(
    objective,
    theta,
    step: int,
    rng: np.random.Generator,
    cfg: GIConfig,
):
    """One SPSA update from two forward objective evaluations."""
    theta_np = np.asarray(theta, dtype=np.float64)
    k = step + 1
    a0 = cfg.lr if cfg.spsa_a is None else cfg.spsa_a
    a_k = a0 / (k**cfg.spsa_alpha)
    c_k = cfg.spsa_c / (k**cfg.spsa_gamma)

    delta = rng.choice([-1.0, 1.0], size=theta_np.shape)
    loss_plus = float(objective(theta_np + c_k * delta))
    loss_minus = float(objective(theta_np - c_k * delta))

    grad_hat = ((loss_plus - loss_minus) / (2.0 * c_k)) * delta
    next_theta = theta_np - a_k * grad_hat
    loss_estimate = 0.5 * (loss_plus + loss_minus)
    grad_norm = float(np.linalg.norm(grad_hat.ravel()))

    return pnp.array(next_theta, requires_grad=True), loss_estimate, grad_norm


def train_single_m(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_star: np.ndarray | None,
    cfg: GIConfig,
    theta0: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    target_mat: np.ndarray | None = None,
) -> dict:
    """Train one QuPer-GI model for a fixed number of ancilla pairs.

    Parameters
    ----------
    theta0:
        Optional warm-start parameters (used by the ancilla schedule).

    mask, target_mat:
        Optional SGI constraint mask and target (see losses.masked_frobenius_loss).
        With mask=None this is the plain GI objective; projections are scored
        with the same (masked or full) objective.

    Returns a dictionary containing the best projected solution found, plus
    `theta_final` (the parameters after the last optimizer step, used to
    warm-start the next ancilla stage). History records include FOM
    instrumentation: grad_norm, p_hat_entropy, projection_residual,
    distinct_perms.
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
    mask_train = None if mask is None else pnp.array(mask, requires_grad=False)
    target_np = a_mat if target_mat is None else target_mat
    target_train = None if target_mat is None else pnp.array(target_mat, requires_grad=False)

    if mask is None:
        score_fn = None  # project_best defaults to the full GI loss
    else:
        from .metrics import masked_mismatch

        def score_fn(p_mat):
            return masked_mismatch(target_np, b_mat, p_mat, mask)

    projection_rng = np.random.default_rng(cfg.seed + 10_000)
    seen_perms: set[tuple[int, ...]] = set()

    best = {
        "value": float("inf"),
        "p": None,
        "theta": None,
        "step": None,
        "projection": None,
    }

    history: list[dict] = []
    optimizer_result: dict | None = None
    success_announced = False

    def objective(current_theta):
        return total_loss(
            current_theta, model, a_train, b_train, cfg,
            mask=mask_train, target_mat=target_train,
        )

    def record_history(
        step: int,
        theta_for_projection,
        loss_value: float,
        grad_norm: float,
    ) -> bool:
        nonlocal success_announced

        record = {
            "step": int(step),
            "train_loss": float(loss_value),
            "grad_norm": grad_norm,
        }

        if step % cfg.projection_interval == 0 or step == cfg.steps - 1:
            p_hat = to_numpy_matrix(model(theta_for_projection))

            p_proj, proj_value, proj_name = project_best(
                p_hat=p_hat,
                a_mat=a_mat,
                b_mat=b_mat,
                rng=projection_rng,
                num_random=cfg.num_random_projections,
                score_fn=score_fn,
            )

            seen_perms.add(tuple(int(v) for v in np.argmax(p_proj, axis=1)))
            record["p_hat_entropy"] = float(-np.sum(p_hat * np.log(p_hat + 1e-12)))
            record["projection_residual"] = float(np.linalg.norm(p_hat - p_proj))
            record["distinct_perms"] = len(seen_perms)

            if proj_value < best["value"]:
                best.update(
                    {
                        "value": float(proj_value),
                        "p": p_proj,
                        "theta": np.asarray(
                            theta_for_projection, dtype=np.float64
                        ).copy(),
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

            stop = best["value"] <= SUCCESS_TOL
            if stop and not success_announced:
                print("Success: projected GI loss reached zero.")
                success_announced = True

            if stop:
                record["best_projected_loss"] = float(best["value"])
                history.append(record)
                return True

        record["best_projected_loss"] = float(best["value"])
        history.append(record)
        return False

    print(model.summary())

    if cfg.optimizer == "adam":
        opt = qml.AdamOptimizer(
            stepsize=cfg.lr,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            eps=cfg.eps,
        )

        for step in range(cfg.steps):
            theta, loss_value, grad_norm = _step_with_grad(opt, objective, theta)
            if record_history(step, theta, loss_value, grad_norm):
                break

    elif cfg.optimizer == "spsa":
        spsa_rng = np.random.default_rng(cfg.seed + 20_000)

        for step in range(cfg.steps):
            theta, loss_value, grad_norm = _step_with_spsa(
                objective, theta, step, spsa_rng, cfg
            )
            if record_history(step, theta, loss_value, grad_norm):
                break

    elif cfg.optimizer == "cobyla":
        eval_count = 0

        def scipy_objective(theta_vec):
            nonlocal eval_count
            current_theta = np.asarray(theta_vec, dtype=np.float64)
            loss_value = float(objective(current_theta))
            record_history(eval_count, current_theta, loss_value, float("nan"))
            eval_count += 1
            return loss_value

        cobyla_result = minimize(
            scipy_objective,
            np.asarray(theta, dtype=np.float64),
            method="COBYLA",
            options={
                "maxiter": cfg.steps,
                "rhobeg": cfg.cobyla_rhobeg,
                "tol": cfg.cobyla_tol,
            },
        )
        theta = pnp.array(
            np.asarray(cobyla_result.x, dtype=np.float64), requires_grad=True
        )
        optimizer_result = {
            "success": bool(getattr(cobyla_result, "success", False)),
            "status": int(getattr(cobyla_result, "status", 0)),
            "message": str(getattr(cobyla_result, "message", "")),
            "nfev": int(getattr(cobyla_result, "nfev", eval_count)),
            "fun": float(getattr(cobyla_result, "fun", np.nan)),
        }

    else:
        raise ValueError("optimizer must be one of 'adam', 'spsa', or 'cobyla'.")

    best["history"] = history
    best["theta_final"] = np.asarray(theta, dtype=np.float64).copy()
    if optimizer_result is not None:
        best["optimizer_result"] = optimizer_result

    if best["p"] is not None:
        best["final_mismatch"] = frobenius_mismatch(a_mat, b_mat, best["p"])
        if score_fn is not None:
            best["final_masked_mismatch"] = score_fn(best["p"])

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
