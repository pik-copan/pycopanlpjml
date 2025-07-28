"""LPJmL-specific region implementations."""

from pycopancore.private._simple_expressions import unknown
import pycopancore.model_components.base.implementation as base
from pycopanlpjml.mixin import AliasMixin
from pycopanlpjml.data import DirtyDataset
from pycopanlpjml.decorator import sync_needed, register_sync


class Region(base.SocialSystem, AliasMixin):
    """An LPJmL-integrating region entity.

    Region entity type (mixin) class for copan:LPJmL component. It inherits the
    copan:CORE region entity and structure and integrates LPJmL input and
    output data (attributes) as well as grid, country and area information as
    `pycoupler.LPJmLData` and `pycoupler.LPJmLDataSet` instances.

    This class serves as the base for both Country and WorldRegion implementations
    in the LPJmL context. It provides the core functionality for handling LPJmL
    data structures and grid-based information.

    Parameters
    ----------
    input : pycoupler.LPJmLDataSet
        Coupled LPJmL model input data for the region
    output : pycoupler.LPJmLData
        Coupled LPJmL model output data for the region
    grid : pycoupler.LPJmLData
        Grid information for the region from LPJmL model
    area : float
        Area of the region in square meters
    world : World
        The world instance this region belongs to
    metabolism : Metabolism, optional
        The metabolism process taxon for this region
    **kwargs : dict
        Additional keyword arguments passed to base Region

    Returns
    -------
    Region
        An instance of the copan:LPJmL Region

    Examples
    --------
    Here's an example demonstrating the initialization of a Region instance
    in the context of an LPJmL simulation:

    First, set up the LPJmL coupling:

    >>> from pycoupler.coupler import LPJmLCoupler
    >>> from pycopanlpjml import World
    >>> from pycopanlpjml.region import Region, Country, WorldRegion

    Configure and connect to LPJmL:

    >>> config_file = "path/to/config_file.json"
    >>> lpjml = LPJmLCoupler(
    ...     config_file=config_file,
    ...     host="localhost",
    ...     port=2042
    ... )

    Initialize the world with LPJmL data:

    >>> world = World(
    ...     input=lpjml.read_input(copy=False),
    ...     output=lpjml.read_historic_output(),
    ...     grid=lpjml.grid,
    ...     country=lpjml.country
    ... )

    Create a region for a specific set of cells:

    >>> cell_indices = [0, 1, 2]  # Example cell indices
    >>> region = Region(
    ...     world=world,
    ...     input=world.input.isel(cell=cell_indices),
    ...     output=world.output.isel(cell=cell_indices),
    ...     grid=world.grid.isel(cell=cell_indices),
    ...     area=world.area.isel(cell=cell_indices)
    ... )

    For country-specific regions, use the Country class:

    >>> germany = Country(
    ...     world=world,
    ...     country_code='DEU',
    ...     input=world.input.sel(country='DEU'),
    ...     output=world.output.sel(country='DEU'),
    ...     grid=world.grid.sel(country='DEU')
    ... )

    For dynamic region groups like the EU, use WorldRegion:

    >>> from country_converter import CountryConverter
    >>> cc = CountryConverter()
    >>> eu = WorldRegion(
    ...     world=world,
    ...     region_name='EU',
    ...     country_converter=cc
    ... )

    Notes
    -----
    - The Region class handles LPJmL-specific data structures and provides
      the foundation for country and region-based analysis
    - All data attributes (input, output, grid) are expected to be compatible
      with the LPJmL data structure
    - The neighbourhood attribute is automatically initialized when grid
      information is provided
    """

    type = "region"

    def __init__(self,
                 name=None,
                 code=None,
                 input=None,
                 output=None,
                 grid=None,
                 area=None,
                 upper_region=None,
                 **kwargs):
        """Initialize an LPJmL region.

        Parameters
        ----------
        input : pycoupler.LPJmLDataSet
            Coupled LPJmL model input
        output : pycoupler.LPJmLData
            Coupled LPJmL model output
        grid : pycoupler.LPJmLData
            Grid of the LPJmL model
        area : float
            Area of each cell in square meters
        **kwargs : dict
            Additional keyword arguments passed to super()
        """
        super().__init__(**kwargs)

        self.type = "region"
        self.code = code
        self.name = name

        # mapping for social_system hierarchy
        if upper_region is not None and self.next_higher_social_system is None:
            self.next_higher_social_system = upper_region
        elif upper_region is not None and self.next_higher_social_system is not None:  # noqa: E501
            raise ValueError(
                "upper_region and next_higher_social_system cannot be set at the same time"  # noqa: E501
            )

        # self._dirty = {'input': False, 'output': False}

        # LPJmL-specific dynamic attributes to be synced with the world
        if input is not None:
            self.input = input
            # self.input = DirtyDataset(self, input, 'input')
            # register_sync(self, 'input')

        if output is not None:
            self.output = output
            # self.output = DirtyDataset(self, output, 'output')
            # register_sync(self, 'output')

        if grid is not None:
            self.grid = grid
            self.neighbourhood = list()

        if area is not None:
            self.area = area

    # def _sync(self, io_type):

    #     if not self._dirty[io_type]:
    #         return

    #     if self.next_higher_region is not None:
    #         next_higher = self.next_higher_region
    #     else:
    #         next_higher = self.world

    #     if len(self.next_lower_regions) > 0:
    #         next_lowers = self.next_lower_regions
    #     else:
    #         next_lowers = None

    #     if io_type == 'input':
    #         if sync_needed(self, io_type):
    #             next_higher.input._ds[
    #                 dict(cell=self.input.cell)
    #             ] = self.input._ds
    #             if next_lowers is not None:
    #                 for next_lower in next_lowers:
    #                     if next_lower.type == "country":
    #                         for key in self.input._ds.keys():
    #                             next_lower.input._ds[key][
    #                                 dict(cell=next_lower.input.cell)
    #                             ].values[:] = self.input._ds[key].values[:]
    #                     else:
    #                         next_lower.input._ds = self.input._ds[
    #                             dict(cell=next_lower.input.cell)
    #                         ]
    #         else:
    #             self.input._ds = next_higher.input._ds.isel(
    #                 cell=self.input.cell
    #             )

    #    elif io_type == 'output':
    #        if sync_needed(self, io_type):
    #            next_higher.output._ds[
    #                dict(cell=self.output.cell)
    #             ] = self.output._ds

    #             if next_lowers is not None:
    #                 for next_lower in next_lowers:
    #                     if next_lower.type == "country":
    #                         for key in self.output._ds.keys():
    #                             next_lower.output._ds[key][
    #                                 dict(cell=next_lower.output.cell)
    #                             ].values[:] = self.output._ds[key].values[:]
    #                     else:
    #                         next_lower.output._ds = self.output._ds[
    #                             dict(cell=next_lower.output.cell)
    #                         ]
    #         else:
    #             self.output._ds = next_higher.output._ds.isel(
    #                 cell=self.output.cell
    #             )

    #    self._dirty[io_type] = False

    @property
    def next_higher_region(self):
        """Get next higher region."""
        return self.next_higher_social_system

    @next_higher_region.setter
    def next_higher_region(self, s):
        """Set next higher region."""
        self.next_higher_social_system = s

    @property  # read-only
    def higher_regions(self):
        """Get higher regions regions recursively."""
        return self.higher_social_systems

    @higher_regions.setter
    def higher_regions(self, u):
        """Set higher regions regions recursively."""
        self.higher_social_systems = u

    @property  # read-only
    def next_lower_regions(self):
        """Get next lower regions."""
        return self.next_lower_social_systems

    @property  # read-only
    def lower_regions(self):
        """Get next lower regions recursively."""
        return self.lower_social_systems


class Country(Region):
    """A Region representing a country based on LPJmL country codes."""

    type = "country"

    def __init__(self,
                 **kwargs):
        """Initialize a country region.

        Parameters
        ----------
        country_code : str
            The LPJmL country code for this country
        **kwargs : dict
            Additional keyword arguments passed to super()
        """
        super().__init__(**kwargs)

        self.type = "country"


class WorldRegion(Region):
    """A Region representing a group of countries (e.g. EU, G7) with dynamic
    membership."""

    type = "world_region"

    def __init__(self,
                 **kwargs):
        """Initialize a world region.

        Parameters
        ----------
        region_name : str
            Name of the region (e.g. 'EU', 'G7')
        country_converter : CountryConverter, optional
            Instance of country-converter for dynamic region resolution
        **kwargs : dict
            Additional keyword arguments passed to super()
        """
        super().__init__(**kwargs)

        self.type = "world_region"
