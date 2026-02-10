import warnings
from .cell import Cell
from .region import Region, Country, WorldRegion
from .world import World
from .model import Model
from .run import run_simulation

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "1.1.5"

warnings.filterwarnings(
    "ignore",
    message=r".*StringDType\(\).*not part in the Zarr format 3 specification.*",  # noqa: E501
    category=UserWarning,
    module="zarr",
)
warnings.filterwarnings(
    "ignore",
    message=r".*StringDType\(\).*not part in the Zarr format 3 specification.*",  # noqa: E501
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
    "run_simulation",
]
