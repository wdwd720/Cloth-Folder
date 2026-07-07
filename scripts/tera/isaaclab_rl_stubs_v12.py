"""Import-only stubs for the RL frameworks that IsaacLab's ``isaaclab_rl``
extension pulls in at load time.

RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY = true
not_for_training = true

WHY THIS EXISTS
---------------
On the Isaac Sim 4.5.0 image, the V12 SO-101 contact-drive scripts fatally fail
at ``SimulationApp`` startup: the IsaacLab ``isaaclab_rl`` extension auto-loads
and executes ``from rl_games.common import env_configurations`` (and equivalents
for ``rsl_rl``, ``stable_baselines3``, ``skrl``) at module-import time. Those
frameworks are NOT installed, so the import raises ``ModuleNotFoundError`` and
Kit treats the extension load as fatal -> the contact-drive test never runs.

Installing the real frameworks is NOT an option: they drag in a conflicting
torch build (cu13) that would break Isaac Sim 4.5's bundled torch 2.5.1+cu118.

FIX
---
Register a ``sys.meta_path`` finder that fabricates *do-nothing* stub modules for
``rl_games`` / ``rsl_rl`` / ``stable_baselines3`` / ``skrl`` (and any submodule of
them). This lets ``isaaclab_rl`` *import* cleanly so SimulationApp starts and the
real contact-physics test can execute. It provides NO real RL functionality.

The stub objects are permissive on purpose (attribute access, call, subscript,
iteration, and use-as-a-base-class via ``__mro_entries__`` all succeed and return
more stubs), because ``isaaclab_rl`` defines wrapper classes like
``class RlGamesVecEnvWrapper(rl_games...IVecEnv)`` at module level. Method BODIES
that would actually use the frameworks only run during training, which we never
invoke here.

DO NOT use this module to run RL training: it makes framework calls silently
no-op. It exists solely to unblock the Isaac import path. The finder is appended
AFTER the real finders, so a genuinely-installed framework always wins; the stub
only fills the gap when the package is truly absent.
"""

from __future__ import annotations

import os
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

# Clear, greppable labels required by the V12 spec.
RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY = True
not_for_training = True

# Root packages we deliberately stub instead of installing (they pull a
# conflicting torch/cu13 that would break Isaac Sim 4.5's torch 2.5.1+cu118).
STUBBED_ROOT_PACKAGES = ("rl_games", "rsl_rl", "stable_baselines3", "skrl")

# Pretend-newest version so any module-level ``version.parse(pkg.__version__) <
# min`` guard (skrl does this) passes without a real install.
_STUB_VERSION = "99.99.99"


def _make_subclass(name: str, module: str, superclass: type = None):
    """Create a real ``type`` standing in for a class from a stubbed package.

    A real class (not just an instance) is returned so it can be subclassed,
    instantiated, and referenced as a base at ``isaaclab_rl`` import time.
    """
    superclass = superclass or _StubObject
    attrs = {
        "__module__": module,
        "__name__": name,
        "__qualname__": name,
        "__display_name__": f"{module}.{name}",
        "__is_isaaclab_rl_stub__": True,
    }
    return type(name, (superclass,), attrs)


class _StubObject:
    """Universal stand-in instance for anything imported from a stubbed package.

    Every operation returns another stub, so module-level code in ``isaaclab_rl``
    (attribute chains, decorator calls, subscripting, subclassing) executes
    without error. Real behaviour is intentionally absent.
    """

    __is_isaaclab_rl_stub__ = True
    __name__ = ""

    def __new__(cls, *args, **kwargs):
        # Support the ``type(name, bases, dict)`` protocol if a stub is ever used
        # directly as a metaclass-style base.
        if len(args) == 3 and isinstance(args[1], tuple):
            return _make_subclass(str(args[0]), cls.__module__, superclass=cls)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    # --- used-as-a-base-class support (PEP 560) --------------------------------
    def __mro_entries__(self, bases):
        # ``class Wrapper(some_stub_instance): ...`` -> substitute the instance's
        # real class as the base.
        return (self.__class__,)

    # --- permissive access -----------------------------------------------------
    def __getattr__(self, name):
        if name == "__version__":
            return _STUB_VERSION
        # dunders that introspection / import machinery expect to be real:
        if name in ("__path__", "__all__", "__file__", "__loader__", "__spec__"):
            raise AttributeError(name)
        return _make_subclass(name, self.__class__.__module__)()

    def __call__(self, *args, **kwargs):
        return _StubObject()

    def __getitem__(self, _key):
        return _StubObject()

    def __setitem__(self, _key, _value):
        pass

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __contains__(self, _item):
        return False

    def __bool__(self):
        return False


class _StubModule(types.ModuleType):
    """A fabricated package module for a stubbed root/subpackage."""

    __is_isaaclab_rl_stub__ = True

    def __init__(self, name: str):
        super().__init__(name)
        self.__all__ = []
        self.__path__ = []          # mark as a package so submodule imports work
        self.__file__ = os.devnull
        self.__version__ = _STUB_VERSION

    def __getattr__(self, name):
        if name == "__version__":
            return _STUB_VERSION
        # Let genuinely-missing import-machinery attributes fail normally.
        if name in ("__wrapped__", "__test__"):
            raise AttributeError(name)
        return _make_subclass(name, self.__name__)()


class _StubLoader(Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):  # nothing to execute
        pass


class _StubFinder(MetaPathFinder):
    """Resolves any import whose root package is in ``STUBBED_ROOT_PACKAGES``."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in STUBBED_ROOT_PACKAGES:
            return ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def install():
    """Idempotently append the stub finder to ``sys.meta_path``.

    Appended (not inserted) so real finders run first: a genuinely-installed
    framework wins, and the stub only activates when the package is absent.
    Returns a small dict describing what was installed (for logging/JSON).
    """
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder())
    return {
        "RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY": RL_STUBS_FOR_ISAACLAB_IMPORT_ONLY,
        "not_for_training": not_for_training,
        "stubbed_root_packages": list(STUBBED_ROOT_PACKAGES),
    }


# Activate on import: simply importing this module installs the stubs, so it can
# be dropped in as the FIRST import of any script that later launches Isaac.
_INSTALL_INFO = install()


if __name__ == "__main__":
    # Self-test: prove the four frameworks import and behave benignly.
    import importlib
    import json

    report = {"install_info": _INSTALL_INFO, "imports": {}}
    for mod in STUBBED_ROOT_PACKAGES:
        try:
            m = importlib.import_module(mod)
            report["imports"][mod] = {
                "imported": True,
                "is_stub": bool(getattr(m, "__is_isaaclab_rl_stub__", False)),
                "version": getattr(m, "__version__", None),
            }
        except Exception as exc:  # noqa: BLE001
            report["imports"][mod] = {"imported": False, "error": repr(exc)}

    # Exercise the exact patterns isaaclab_rl uses at import time.
    checks = {}
    try:
        from rl_games.common import env_configurations, vecenv  # noqa: F401
        from rl_games.common.algo_observer import AlgoObserver  # noqa: F401

        class _WrapCheck(AlgoObserver):  # subclass a stubbed class (base-class path)
            pass

        env_configurations.register("x", {})  # call a stubbed attr
        checks["rl_games_from_import_and_subclass"] = True
    except Exception as exc:  # noqa: BLE001
        checks["rl_games_from_import_and_subclass"] = repr(exc)

    for spec in (
        "from rsl_rl.env import VecEnv",
        "from stable_baselines3.common.vec_env import VecEnv",
        "from skrl.utils.spaces.torch import compute_space_size",
    ):
        try:
            exec(spec)  # noqa: S102
            checks[spec] = True
        except Exception as exc:  # noqa: BLE001
            checks[spec] = repr(exc)

    try:
        from packaging import version

        skrl = importlib.import_module("skrl")
        checks["skrl_version_parse_ok"] = bool(
            version.parse(skrl.__version__) >= version.parse("1.0.0")
        )
    except Exception as exc:  # noqa: BLE001
        checks["skrl_version_parse_ok"] = repr(exc)

    report["import_time_patterns"] = checks
    print(json.dumps(report, indent=2))
