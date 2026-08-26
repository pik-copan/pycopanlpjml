"""pycopanlpjml - copan:LPJmL coupling framework."""

import warnings
from .cell import Cell
from .region import Region, Country, WorldRegion
from .world import World
from .model import Model
from .run import run_simulation
from .output import Output, OutputDefinitionMixin

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "2.0.0"

warnings.filterwarnings(
    "ignore",
    message=(
        r".*StringDType\(\).*not part in the Zarr format 3 specification.*"
    ),
    category=UserWarning,
    module="zarr",
)
warnings.filterwarnings(
    "ignore",
    message=(
        r".*StringDType\(\).*not part in the Zarr format 3 specification.*"
    ),
    category=UserWarning,
    module="zarr.core.array",
)

__all__ = [
    "__version__",
    "Cell",
    "Region",
    "Country",
    "WorldRegion",
    "World",
    "Model",
    "Output",
    "OutputDefinitionMixin",
    "run_simulation",
]
