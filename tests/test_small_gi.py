"""End-to-end tiny GI recovery test (paper development plan, Phase 3).

Trains the smallest configuration (N = 4, m = 0, Borel) on a planted
isomorphic pair and requires a perfect recovery: projected GI loss == 0.
"""

import numpy as np
import pytest

from quper_gi.configs import GIConfig, validate_config
from quper_gi.graph_data import make_isomorphic_pair, verify_isomorphic_pair
from quper_gi.metrics import is_success, permutation_validity
from quper_gi.optimizer import train_single_m


def _small_pair(cfg: GIConfig):
    a_mat, b_mat, p_star, _ = make_isomorphic_pair(
        num_vertices=cfg.num_vertices,
        edge_prob=cfg.graph_p,
        seed=cfg.seed,
    )
    verify_isomorphic_pair(a_mat, b_mat, p_star)
    return a_mat, b_mat, p_star


def test_tiny_gi_recovery():
    cfg = GIConfig(
        num_vertices=4,
        ancillas=0,
        ansatz="borel",
        steps=150,
        seed=0,
    )

    a_mat, b_mat, p_star = _small_pair(cfg)

    result = train_single_m(a_mat, b_mat, p_star, cfg)

    assert result["p"] is not None
    assert permutation_validity(result["p"])
    assert is_success(a_mat, b_mat, result["p"]), (
        f"projected GI loss {result['value']} did not reach zero"
    )


def test_spsa_optimizer_runs_forward_path():
    cfg = GIConfig(
        num_vertices=4,
        ancillas=0,
        ansatz="borel",
        optimizer="spsa",
        steps=2,
        seed=0,
        num_random_projections=2,
    )
    a_mat, b_mat, p_star = _small_pair(cfg)

    result = train_single_m(a_mat, b_mat, p_star, cfg)

    assert result["p"] is not None
    assert result["history"]


def test_cobyla_optimizer_reports_result_metadata():
    cfg = GIConfig(
        num_vertices=4,
        ancillas=0,
        ansatz="borel",
        optimizer="cobyla",
        steps=5,
        seed=0,
        num_random_projections=2,
    )
    a_mat, b_mat, p_star = _small_pair(cfg)

    result = train_single_m(a_mat, b_mat, p_star, cfg)

    assert result["p"] is not None
    assert result["history"]
    assert result["optimizer_result"]["nfev"] > 0


def test_validate_config_rejects_unknown_optimizer():
    cfg = GIConfig(optimizer="bogus")

    with pytest.raises(ValueError, match="optimizer"):
        validate_config(cfg)


def test_parameter_transfer_shapes():
    """Ancilla-schedule warm start must produce the right parameter count."""
    from quper_gi.ansatz import get_num_params, param_keys, transfer_theta

    q = 2
    for ansatz in ("borel", "bruhat"):
        theta_old = np.linspace(0.0, 1.0, get_num_params(ansatz, q))
        theta_new = transfer_theta(theta_old, ansatz, q, m_old=0, m_new=1)
        assert len(theta_new) == get_num_params(ansatz, q + 1)

        # Every old key must survive in the new layout with its value copied.
        old_keys = param_keys(ansatz, q, 0)
        new_keys = param_keys(ansatz, q, 1)
        lookup = dict(zip(new_keys, theta_new))
        for key, value in zip(old_keys, theta_old):
            if key in lookup:
                assert lookup[key] == value
