try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "1.1.5"

from .cell import Cell
from .world import World
from .component import Component

__all__ = ["__version__", "Cell", "World", "Component"]
