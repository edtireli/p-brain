import os
import importlib.util
import types
import numpy as np
import sys
import matplotlib

os.environ["MPLBACKEND"] = "Agg"
matplotlib.use = lambda *a, **k: None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
modules_pkg = types.ModuleType('modules')
sys.modules['modules'] = modules_pkg
spec_ai = importlib.util.spec_from_file_location('modules.AI_tissue_functions', os.path.join(ROOT, 'modules', 'AI_tissue_functions.py'), submodule_search_locations=[os.path.join(ROOT, 'modules')])
ai = importlib.util.module_from_spec(spec_ai)
sys.modules['modules.AI_tissue_functions'] = ai
spec_ai.loader.exec_module(ai)
_get_first_npy = ai._get_first_npy

def test_get_first_npy_ignores_hidden(tmp_path):
    max_dir = tmp_path / 'Max'
    max_dir.mkdir()
    np.save(max_dir / 'valid.npy', np.array([1,2,3]))
    # Create a hidden macOS resource fork file
    (max_dir / '._valid.npy').write_bytes(b'')
    selected = _get_first_npy(str(max_dir))
    assert selected == 'valid.npy'
