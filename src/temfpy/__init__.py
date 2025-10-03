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
