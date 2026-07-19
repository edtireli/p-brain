"""Device resolution + safe Metal provisioning (``pbrain.core.devices``)."""
import types
import warnings

from pbrain.core import devices as D

_NO_GPU = {"cuda_torch": False, "mps_torch": False, "gpu_tf": False}


def _apple(monkeypatch, machine="arm64", system="Darwin"):
    monkeypatch.setattr(D, "platform", types.SimpleNamespace(
        system=lambda: system, machine=lambda: machine))


def test_provision_refuses_metal_on_too_new_tf(monkeypatch):
    """The bug that broke TF: installing tensorflow-metal above its TF cap makes
    `import tensorflow` fail. provision_mps must NEVER install it there."""
    monkeypatch.setattr(D, "probe", lambda: dict(_NO_GPU))
    _apple(monkeypatch)
    monkeypatch.setattr(D, "_tf_version", lambda: (2, 21))
    installs: list[str] = []
    monkeypatch.setattr(D, "_pip_install", lambda pkg, log=None: installs.append(pkg) or True)
    ok, msg = D.provision_mps(auto_install=True)
    assert ok is False
    assert installs == []                        # did not touch the working env
    assert "2.16" in msg and "tensorflow-metal" in msg


def test_provision_installs_metal_when_compatible(monkeypatch):
    monkeypatch.setattr(D, "probe", lambda: dict(_NO_GPU))
    _apple(monkeypatch)
    monkeypatch.setattr(D, "_tf_version", lambda: (2, 16))
    installs: list[str] = []
    monkeypatch.setattr(D, "_pip_install", lambda pkg, log=None: installs.append(pkg) or True)
    ok, msg = D.provision_mps(auto_install=True)
    assert installs == ["tensorflow-metal"]      # compatible → provisions it
    assert "re-run" in msg.lower()               # honest: takes effect next run


def test_provision_no_install_without_optin(monkeypatch):
    monkeypatch.setattr(D, "probe", lambda: dict(_NO_GPU))
    _apple(monkeypatch)
    monkeypatch.setattr(D, "_tf_version", lambda: (2, 16))
    installs: list[str] = []
    monkeypatch.setattr(D, "_pip_install", lambda pkg, log=None: installs.append(pkg) or True)
    ok, msg = D.provision_mps(auto_install=False)
    assert installs == [] and ok is False and "tensorflow-metal" in msg


def test_provision_non_apple(monkeypatch):
    monkeypatch.setattr(D, "probe", lambda: dict(_NO_GPU))
    _apple(monkeypatch, machine="x86_64", system="Linux")
    ok, msg = D.provision_mps(auto_install=True)
    assert ok is False and "platform" in msg.lower()


def test_resolve_mps_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(D, "probe", lambda: dict(_NO_GPU))
    monkeypatch.setattr(D, "provision_mps", lambda auto_install=False, log=None: (False, "no gpu here"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert D.resolve("mps", auto_install=True) == "cpu"
    assert any("no gpu here" in str(x.message) for x in w)


def test_resolve_uses_backend_when_present(monkeypatch):
    monkeypatch.setattr(D, "probe", lambda: {**_NO_GPU, "gpu_tf": True})
    assert D.resolve("mps") == "mps"
    assert D.resolve("auto") == "mps"


def test_resolve_cpu_and_auto_default(monkeypatch):
    monkeypatch.setattr(D, "probe", lambda: dict(_NO_GPU))
    assert D.resolve("cpu") == "cpu"
    assert D.resolve("auto") == "cpu"
