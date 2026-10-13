import math

import pytest

from ml import rdp


def _opacus_eps(sigma, q, steps, delta):
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    return acc.get_epsilon(delta=delta)


@pytest.mark.parametrize("sigma,q,steps,max_ratio", [
    (1.0, 0.05, 400, 1.05), (0.8, 0.09, 450, 1.05), (1.5, 0.02, 1000, 1.05), (2.0, 0.09, 450, 1.05),
    (3.0, 0.089, 450, 1.05),
    # very low noise: integer-only orders are coarse (eps is large and not meaningful anyway); stays conservative
    (0.5, 0.05, 200, 1.30)])
def test_matches_opacus_and_is_never_optimistic(sigma, q, steps, max_ratio):
    ours, theirs = rdp.epsilon([(sigma, q, steps)], 1e-5), _opacus_eps(sigma, q, steps, 1e-5)
    assert ours >= theirs * 0.98                       # never meaningfully below Opacus (no under-reporting)
    assert ours <= theirs * max_ratio


def test_composition_is_monotone_and_matches_single_run():
    one = rdp.epsilon([(1.0, 0.05, 100)])
    two = rdp.epsilon([(1.0, 0.05, 100), (1.0, 0.05, 100)])
    both = rdp.epsilon([(1.0, 0.05, 200)])
    assert two > one
    assert math.isclose(two, both, rel_tol=1e-9)       # same mechanism composed == more steps


def test_epsilon_decreases_with_noise_and_increases_with_steps():
    assert rdp.epsilon([(2.0, 0.05, 300)]) < rdp.epsilon([(1.0, 0.05, 300)])
    assert rdp.epsilon([(1.0, 0.05, 600)]) > rdp.epsilon([(1.0, 0.05, 300)])


def test_edge_cases():
    assert rdp.epsilon([]) == 0.0
    assert math.isinf(rdp.epsilon([(0.0, 0.05, 10)]))
