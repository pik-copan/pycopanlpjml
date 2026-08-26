# Examples

A full coupled application is
[InSEEDS](https://github.com/pik-copan/inseeds)
(conservation agriculture and regenerative tillage).

Additional models, presented in the copan:LPJmL paper can be found in 
[landmanager](https://github.com/jnnsbrr/landmanager).

The minimal pattern is a `Model` subclass that builds `World`, countries and
cells, then implements `update(t)`:

```python
import pycopanlpjml as lpjml


class MyModel(lpjml.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world = lpjml.World(
            input=self.lpjml.read_input(copy=False),
            output=self.lpjml.read_historic_output(),
            grid=self.lpjml.grid,
            country_code=self.lpjml.country,
            area=self.lpjml.terr_area,
        )
        self.init_countries(country_class=lpjml.Country)
        self.init_cells(cell_class=lpjml.Cell)

    def update(self, t):
        self.update_countries(t)
        self.update_lpjml(t)
        self.collect_outputs(t)
```

See the [User Guide](../user-guide/index.md) for the yearly coupling contract
and output configuration.
