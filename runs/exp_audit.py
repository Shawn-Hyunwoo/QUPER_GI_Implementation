"""Final audit: q=3 exhaustive span, GI-QAP identity, m=2 DSM, Prop. 8."""
import itertools

import numpy as np
from pennylane import numpy as pnp

from quper_gi.dsm_circuit import QuPerDSM
from quper_gi.graph_data import make_isomorphic_pair, permutation_vector_to_matrix

# [1] q=3 (N=8) exhaustive span vs Table 1
for ansatz, expected in (("borel", 64), ("bruhat", 1344)):
    m = QuPerDSM(num_vertices=8, ancillas=0, ansatz=ansatz)
    perms = set()
    for bits in itertools.product([0.0, np.pi], repeat=m.num_params):
        p = np.rint(np.asarray(m(pnp.array(bits, requires_grad=False))))
        assert np.allclose(p.sum(0), 1) and np.allclose(p.sum(1), 1), "not a permutation!"
        perms.add(tuple(np.argmax(p, 1)))
    verdict = "PASS" if len(perms) == expected else "FAIL"
    print(f"[1] q=3 {ansatz}: span={len(perms)} (expected {expected}) -> {verdict}")

# [2] GI <-> QAP expansion identity (paper Sec 6.3)
rng = np.random.default_rng(0)
a_mat, b_mat, p_star, _ = make_isomorphic_pair(8, 0.5, 3)
ok = True
for s in range(20):
    pm = permutation_vector_to_matrix(rng.permutation(8))
    lhs = np.sum((a_mat - pm @ b_mat @ pm.T) ** 2)
    rhs = (np.trace(a_mat.T @ a_mat) + np.trace(b_mat.T @ b_mat)
           - 2 * np.trace(a_mat @ pm @ b_mat.T @ pm.T))
    ok &= bool(np.isclose(lhs, rhs))
print(f"[2] GI=QAP identity (20 random P) -> {'PASS' if ok else 'FAIL'}")

# [3] DSM property with m=2 ancillas, random continuous theta (N=4, 8 wires)
m2 = QuPerDSM(num_vertices=4, ancillas=2, ansatz="bruhat")
th = pnp.array(rng.uniform(0, np.pi, m2.num_params), requires_grad=False)
p = np.asarray(m2(th))
ok3 = (np.allclose(p.sum(0), 1, atol=1e-8) and np.allclose(p.sum(1), 1, atol=1e-8)
       and bool((p >= -1e-9).all()))
print(f"[3] m=2 DSM (rows/cols sum 1, >=0) -> {'PASS' if ok3 else 'FAIL'}")

# [4] Prop. 8: m=1, theta in {0,pi} -> P_hat is DSM, need not be a hard permutation
m1 = QuPerDSM(num_vertices=4, ancillas=1, ansatz="bruhat")
mixed = hard = 0
for s in range(60):
    r2 = np.random.default_rng(s)
    th = pnp.array(r2.choice([0.0, np.pi], m1.num_params), requires_grad=False)
    p = np.asarray(m1(th))
    assert np.allclose(p.sum(0), 1, atol=1e-8) and np.allclose(p.sum(1), 1, atol=1e-8)
    if np.all(np.isclose(p, 0, atol=1e-8) | np.isclose(p, 1, atol=1e-8)):
        hard += 1
    else:
        mixed += 1
print(f"[4] m=1 endpoints: {hard} hard + {mixed} convex mixtures, all DSM -> PASS "
      f"(Prop. 8 w>1 observed: {mixed > 0})")
