"""Basic checks for earned_influence_toy.py. Run with: python -m pytest test_earned_influence_toy.py"""
import numpy as np

import earned_influence_toy as m


def test_weights_bounded_by_cap():
    x = np.array([0.0, 0.3, 1.0])
    for K in m.CAPS:
        w = m.weights(x, K)
        assert w.min() >= 1 and w.max() <= K
        assert w[0] == 1 and w[-1] == K


def test_equal_weighting_is_cap_one():
    assert np.all(m.weights(np.random.default_rng(0).random(10), 1) == 1)


def test_sign_change_interpolation():
    deps = np.array([-0.1, 0.0, 0.1])
    assert m.interpolated_sign_changes(deps, [-1.0, -0.5, 0.5]) == [0.05]
    assert m.interpolated_sign_changes(deps, [1.0, 2.0, 3.0]) == []


def test_small_run_is_reproducible_and_paired():
    params = dict(m.BASE, N=11, T=20)
    a = m.simulate(0.5, seed=1, n_pops=5, **params)
    b = m.simulate(0.5, seed=1, n_pops=5, **params)
    assert a == b
    assert set(a["diff"]) == set(m.CAPS[1:])


def test_entrenchment_never_raises_competence():
    p = np.array([0.30, 0.60, 0.60])
    wrong = np.array([True, True, True])
    fix = np.array([False, False, True])
    out = m.update_competence(p, wrong, fix, ETA=0.1, FLOOR=0.40)
    assert out[0] == 0.30            # below the floor: entrenching leaves it unchanged
    assert out[1] < 0.60             # above the floor: entrenching lowers it
    assert out[2] > 0.60             # correcting raises it
