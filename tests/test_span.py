"""Span tests against paper Theorem 6 / Table 1 for q = 2 (N = 4).

With m = 0 and every parameter in {0, pi}, each parameter vector yields an
exact permutation matrix. Enumerating ALL endpoint combinations must produce
exactly the group sizes proven in the paper:

  - Bruhat (X-B-W-B): |LX_2| = 2^(q(q+1)/2) * prod_k (2^k - 1) = 8 * 3 = 24
    (Table 1, q = 2 -> p = 24 = 4!, i.e. the full symmetric group)
  - Borel (X-B):      |X_2| * |B_2| = 2^q * 2^C(q,2) = 4 * 2 = 8
    (unique factorization, paper Theorem 3 / Prop. 6)

This exercises the entire construction at once: gate endpoint semantics,
Borel/Weyl word ordering, Bell-pair preparation, and the Eq. (3) rescaling.
"""

import itertools

import numpy as np
from pennylane import numpy as pnp

from quper_gi.dsm_circuit import QuPerDSM
from quper_gi.metrics import permutation_validity


def _distinct_endpoint_permutations(ansatz: str) -> set[tuple[int, ...]]:
    model = QuPerDSM(num_vertices=4, ancillas=0, ansatz=ansatz)
    perms: set[tuple[int, ...]] = set()

    for bits in itertools.product([0.0, np.pi], repeat=model.num_params):
        theta = pnp.array(bits, requires_grad=False)
        p_hat = np.rint(np.asarray(model(theta), dtype=np.float64))
        assert permutation_validity(p_hat)
        perms.add(tuple(int(v) for v in np.argmax(p_hat, axis=1)))

    return perms


def test_bruhat_q2_spans_all_24_permutations():
    assert len(_distinct_endpoint_permutations("bruhat")) == 24


def test_borel_q2_span_size_is_8():
    assert len(_distinct_endpoint_permutations("borel")) == 8
