"""[SYNTHETIC mechanism-validation] tests for the controlled-truth oracle and
the learning-dynamics checks. Fast, deterministic, no network/cluster.

These validate the LEARNING MECHANISM against a known truth — they are not a
headline metric (that is the HIDDEN-stratum Brier in scripts/run.py).
"""
import random
from pathlib import Path

import pytest

from owm.codegraph import parse_repo
from owm.graph import LEARNED
from owm.topology import seed_graph

from groundtruth.oracle import sample_incident, stream
from scripts.dynamics_check import (
    check_convergence,
    check_order_invariance,
    check_spurious,
    make_factory,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ob_mini"
ORIGIN = "synthetic::ORACLE_KNOB"


@pytest.fixture
def factory():
    """Fresh-base factory backed by the mini OB fixture (isolated, fast)."""
    base = seed_graph(parse_repo(FIXTURE))
    return make_factory(ORIGIN, cwm=base)


@pytest.fixture
def services(factory):
    return factory().services


# --------------------------------------------------------------------------- #
# oracle.sample_incident
# --------------------------------------------------------------------------- #
def test_sample_incident_p1_always_includes_coupled(services):
    rng = random.Random(0)
    svc = services[0]
    for _ in range(50):
        impacted = sample_incident(ORIGIN, {svc: 1.0}, rng, all_services=services)
        assert svc in impacted


def test_sample_incident_p0_never_includes_coupled(services):
    rng = random.Random(0)
    svc = services[0]
    for _ in range(50):
        impacted = sample_incident(ORIGIN, {svc: 0.0}, rng, all_services=services)
        assert svc not in impacted


def test_sample_incident_no_noise_never_adds_uncoupled(services):
    rng = random.Random(1)
    coupled = services[0]
    others = set(services) - {coupled}
    for _ in range(50):
        impacted = sample_incident(
            ORIGIN, {coupled: 1.0}, rng, noise=0.0, all_services=services
        )
        # only the coupled service may appear; never an uncoupled one
        assert impacted == {coupled}
        assert impacted.isdisjoint(others)


def test_sample_incident_is_deterministic_given_seed(services):
    a = stream(ORIGIN, {services[0]: 0.5, services[1]: 0.5}, 20,
               random.Random(42), all_services=services)
    b = stream(ORIGIN, {services[0]: 0.5, services[1]: 0.5}, 20,
               random.Random(42), all_services=services)
    assert a == b


def test_sample_incident_noise_can_add_uncoupled(services):
    # With high noise over many draws, at least one uncoupled service should appear.
    rng = random.Random(3)
    coupled = services[0]
    seen_uncoupled = False
    for _ in range(100):
        impacted = sample_incident(
            ORIGIN, {coupled: 0.0}, rng, noise=1.0, all_services=services
        )
        if impacted - {coupled}:
            seen_uncoupled = True
            break
    assert seen_uncoupled


# --------------------------------------------------------------------------- #
# convergence: a deterministic stream drives the learned weight toward ~1.0
# --------------------------------------------------------------------------- #
def test_deterministic_stream_converges_and_is_learned(factory, services):
    svc = services[0]
    res = check_convergence(factory, ORIGIN, {svc: 1.0}, n=10, seed=5)
    w = res["learned_weights"][svc]
    assert w > 0.8, f"expected learned weight > 0.8, got {w}"
    assert res["converged"][svc] is True
    assert res["sources"][svc] == LEARNED


def test_convergence_returns_measured_dict(factory, services):
    svc = services[0]
    res = check_convergence(factory, ORIGIN, {svc: 1.0}, n=8, seed=1)
    assert res["origin"] == ORIGIN
    assert res["n"] == 8
    assert svc in res["learned_weights"]
    assert res["n_services"] == len(services)


# --------------------------------------------------------------------------- #
# spurious edges under noise (precision risk; expected > 0)
# --------------------------------------------------------------------------- #
def test_spurious_edges_appear_under_noise(factory, services):
    svc = services[0]
    res = check_spurious(factory, ORIGIN, {svc: 1.0}, n=20, seed=2, noise=0.9)
    # high co-degradation noise over 20 incidents should mint at least one
    # spurious LEARNED edge to an uncoupled service (the precision risk).
    assert res["n_spurious"] >= 1
    assert all(s not in {svc} for s in res["spurious_edges"])
    assert res["n_uncoupled_services"] == len(services) - 1


def test_no_spurious_without_noise(factory, services):
    svc = services[0]
    res = check_spurious(factory, ORIGIN, {svc: 1.0}, n=20, seed=2, noise=0.0)
    assert res["n_spurious"] == 0


# --------------------------------------------------------------------------- #
# order-invariance of the Beta posterior
# --------------------------------------------------------------------------- #
def test_order_invariance_holds(factory, services):
    true_probs = {services[0]: 1.0, services[1]: 0.6}
    res = check_order_invariance(factory, ORIGIN, true_probs, n=15, seed=4)
    assert res["order_invariant"] is True
    assert len(res["per_edge"]) >= 1
    for k, v in res["per_edge"].items():
        assert v["match"] is True
