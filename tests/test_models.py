import importlib.util
import types
import numpy as np
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use = lambda *a, **k: None

modules_pkg = types.ModuleType('modules')
sys.modules['modules'] = modules_pkg

spec_km = importlib.util.spec_from_file_location('modules.kinetic_models', os.path.join(ROOT, 'modules', 'kinetic_models.py'))
km = importlib.util.module_from_spec(spec_km)
sys.modules['modules.kinetic_models'] = km
spec_km.loader.exec_module(km)
extended_tofts_tikhonov = km.extended_tofts_tikhonov
residue_metrics = km.residue_metrics
residue_to_cbf = km.residue_to_cbf

spec_ai = importlib.util.spec_from_file_location('modules.AI_tissue_functions', os.path.join(ROOT, 'modules', 'AI_tissue_functions.py'), submodule_search_locations=[os.path.join(ROOT, 'modules')])
ai = importlib.util.module_from_spec(spec_ai)
sys.modules['modules.AI_tissue_functions'] = ai
spec_ai.loader.exec_module(ai)
patlak_total = ai.patlak_total
two_compartment = ai.two_compartment_tikhonov
settings_module = ai.settings


def synthetic_data():
    np.random.seed(0)
    t = np.linspace(0, 60, 40)
    aif = np.exp(-t / 10)
    # Generate tissue curve using simple extended Tofts model
    Ktrans = 0.001
    ve = 0.2
    vp = 0.05
    conv = np.zeros_like(t)
    for i in range(len(t)):
        integ = np.trapz(aif[:i + 1] * np.exp(-(t[i] - t[:i + 1]) * Ktrans / ve), x=t[:i + 1])
        conv[i] = integ
    tissue = Ktrans * conv + vp * aif
    noise = 0.01 * np.random.randn(*tissue.shape)
    return t, aif, tissue + noise


def test_models_differ():
    t, ca, ct = synthetic_data()
    ki_p, _, _ = patlak_total(ct, ca, t)
    ktrans, _, _ = extended_tofts_tikhonov(ca, ct, t, lambd=5.0)
    ki_t = ktrans * 6000
    assert np.isfinite(ki_t)
    assert not np.isclose(ki_p, ki_t)


def test_lambda_effects():
    t, ca, ct = synthetic_data()
    ki_p, _, _ = patlak_total(ct, ca, t)
    k0, _, _ = extended_tofts_tikhonov(ca, ct, t, lambd=0.0)
    assert np.isclose(k0 * 6000, ki_p, rtol=0.2)

    ct_zero = np.zeros_like(t)
    k_unreg, _, _ = extended_tofts_tikhonov(ca, ct_zero, t, lambd=0.0)
    k_reg, _, _ = extended_tofts_tikhonov(ca, ct_zero, t, lambd=100.0)
    assert k_reg < k_unreg


def test_two_compartment_handles_constant_aif():
    t = np.linspace(0, 60, 20)
    ca = np.ones_like(t)
    ct = np.ones_like(t) * 0.2
    ki, _, _, _ = two_compartment(ca, ct, time_array=t)
    assert np.isfinite(ki)


def test_residue_metrics_exponential_residue():
    tau = 5.0
    dt = 1.0
    time = np.arange(0, 301)
    residue = np.exp(-time / tau)
    mtt, cth, h, mu = residue_metrics(residue, dt)
    h_expected = (1.0 / tau) * np.exp(-time / tau)
    assert np.isclose(mtt, tau, rtol=0.05)
    assert np.isclose(mu, tau, rtol=0.05)
    assert np.isclose(cth, tau, rtol=0.05)
    rel_err = np.linalg.norm(h - h_expected) / np.linalg.norm(h_expected)
    assert rel_err < 0.05


def test_two_compartment_uses_patlak_initial_guess():
    t, ca, ct = synthetic_data()
    ki_patlak, _, _ = patlak_total(ct, ca, t)
    captured = {}

    original = ai.extended_tofts_tikhonov
    settings_module = ai.settings

    def recorder(aif, tissue, time_array, lambd=settings_module.TIKHONOV_LAMBDA, x0=(0.001, 0.2, 0.05)):
        captured['x0'] = x0
        return km.extended_tofts_tikhonov(aif, tissue, time_array, lambd=lambd, x0=x0)

    previous_flag = settings_module.TWO_COMPARTMENT_INIT_FROM_PATLAK
    try:
        ai.extended_tofts_tikhonov = recorder
        settings_module.TWO_COMPARTMENT_INIT_FROM_PATLAK = True
        two_compartment(ca, ct, time_array=t)
    finally:
        settings_module.TWO_COMPARTMENT_INIT_FROM_PATLAK = previous_flag
        ai.extended_tofts_tikhonov = original

    assert 'x0' in captured
    assert np.isfinite(captured['x0'][0])
    assert np.isclose(captured['x0'][0], ki_patlak / 6000.0, rtol=0.05)


def test_tikhonov_regularisation_uses_linear_lambda():
    a = np.eye(4)
    ct = np.arange(1, 5, dtype=float)
    lam = 2.0
    sol = km.tikhonov_regularization(a, ct, lam)
    expected = ct / (1.0 + lam)
    assert np.allclose(sol, expected)


def test_residue_to_cbf_scaling():
    value = residue_to_cbf(0.002)
    expected = max(0.002 * 6000.0, 0.0)
    assert np.isclose(value, expected)
