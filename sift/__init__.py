from sift import kernels
from sift._constants import KernelType
from sift._sift import SiFT, sifter

__all__ = ["SiFT", "sifter", "kernels", "KernelType"]


def _get_version() -> str:
    import importlib.metadata as importlib_metadata

    return importlib_metadata.version("sift-sc")


def _setup_logger() -> "logging.Logger":  # noqa: F821
    import logging

    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    formatter = logging.Formatter("sift: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # this prevents double outputs
    logger.propagate = False
    return logger


__version__ = _get_version()
logger = _setup_logger()

del _get_version, _setup_logger
