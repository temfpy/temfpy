import logging as _logging

# Try to get version from hatch-vcs generated file first, then fallback
try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("temfpy")


def setup_logging(level=_logging.INFO):
    _logging.basicConfig(
        level=level,
    )


# Configure Lazy imports #
# ---------------------- #

__all__ = [
    "slater",
    "pfaffian",
    "gutzwiller",
    "iMPS",
    "schmidt_utils",
    "utils",
    "testing",
]

_lazy_modules = {
    "slater": "temfpy.slater",
    "pfaffian": "temfpy.pfaffian",
    "gutzwiller": "temfpy.gutzwiller",
    "iMPS": "temfpy.iMPS",
    "schmidt_utils": "temfpy.schmidt_utils",
    "utils": "temfpy.utils",
    "testing": "temfpy.testing",
}


def __getattr__(name):
    """Lazy load modules on first access."""
    if name in _lazy_modules:
        import importlib
        module = importlib.import_module(_lazy_modules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
