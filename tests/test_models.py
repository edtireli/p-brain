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

spec_ai = importlib.util.spec_from_file_location('modules.AI_tissue_functions', os.path.join(ROOT, 'modules', 'AI_tissue_functions.py'), submodule_search_locations=[os.path.join(ROOT, 'modules')])
ai = importlib.util.module_from_spec(spec_ai)
sys.modules['modules.AI_tissue_functions'] = ai
spec_ai.loader.exec_module(ai)
patlak_total = ai.patlak_total
two_compartment = ai.two_compartment_tikhonov


def synthetic_data():
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
    ki_p, lam_p, _ = patlak_total(ct, ca, t)
    ki_t, _, _, _ = two_compartment(ca, ct, time_array=t)
    assert np.isfinite(ki_t)
    assert not np.isclose(ki_p, ki_t)


def test_two_compartment_handles_constant_aif():
    t = np.linspace(0, 60, 20)
    ca = np.ones_like(t)
    ct = np.ones_like(t) * 0.2
    ki, _, _, _ = two_compartment(ca, ct, time_array=t)
    assert np.isfinite(ki)
