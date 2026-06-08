"""Cell entity type for copan:LPJmL component.

Cells store VIEWS into Country/World data (shared memory).
Views are passed at init and stored - modifications propagate automatically.
"""

import pycopancore.model_components.base.implementation as base


class Cell(base.Cell):
    """An LPJmL-integrating cell entity.

    Cells hold xarray views into Country (or World) data. Views are passed
    at creation time and stored directly. Since xarray views share memory,
    modifications automatically propagate to the parent data.

    Parameters
    ----------
    world : World
        Reference to World instance (required by pycopancore).
    country : Country or str, optional
        Country this cell belongs to.
    cell_index : int
        Global cell index.
    local_index : int, optional
        Local index within country's data arrays.
    input : xarray view, optional
        View into input data for this cell.
    output : xarray view, optional
        View into output data for this cell.
    grid : xarray view, optional
        View into grid data for this cell.
    area : xarray view, optional
        View into area data for this cell.
    """

    def __init__(
        self,
        world=None,
        country=None,
        cell_index=None,
        local_index=None,
        input=None,
        output=None,
        grid=None,
        area=None,
        **kwargs,
    ):
        if "world" not in kwargs:
            kwargs["world"] = world

        self._country_string = None
        if country is not None and "social_system" not in kwargs:
            if isinstance(country, str):
                self._country_string = country
            else:
                kwargs["social_system"] = country

        super().__init__(**kwargs)

        self._cell_index = cell_index
        self._local_index = local_index
        self.neighbourhood = []

        # Store views directly (passed at init)
        self.input = input
        self.output = output
        self.grid = grid
        self.area = area

    @property
    def cell_index(self):
        """Global cell index."""
        return self._cell_index

    @cell_index.setter
    def cell_index(self, value):
        self._cell_index = value

    @property
    def local_index(self):
        """Local index within country's data arrays."""
        return self._local_index

    @local_index.setter
    def local_index(self, value):
        self._local_index = value

    @property
    def country(self):
        """Country this cell belongs to."""
        if self._country_string is not None:
            return self._country_string
        return getattr(self, "social_system", None)

    @country.setter
    def country(self, value):
        """Set country (either string code or Country object)."""
        if isinstance(value, str):
            self._country_string = value
        else:
            self._country_string = None
            self.social_system = value

    @property
    def to_earth(self):
        """Alias for input (data sent to LPJmL)."""
        return getattr(self, "input", None)

    @property
    def from_earth(self):
        """Alias for output (data from LPJmL)."""
        return getattr(self, "output", None)

    @property
    def country_code(self):
        """ISO 3-letter country code."""
        country = getattr(self, "country", None)
        if country is None:
            return None
        if isinstance(country, str):
            return country
        if hasattr(country, "code"):
            return country.code
        if hasattr(country, "values"):
            return str(country.values)
        return str(country)
