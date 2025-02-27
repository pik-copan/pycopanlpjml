# pycopanlpjml <a href=''><img src='docs/source/_static/logo.png' align="right" height="85" /></a>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14246191.svg)](https://doi.org/10.5281/zenodo.14246191) 
[![CI](https://github.com/pik-copan/pycopanlpjml/actions/workflows/check.yml/badge.svg)](https://github.com/pik-copan/pycopanlpjml/actions) [![codecov](https://codecov.io/gh/pik-copan/pycopanlpjml/graph/badge.svg?token=A7ONVL4AR4)](https://codecov.io/gh/pik-copan/pycopanlpjml)

*copan:LPJmL, an advanced World-Earth modeling framework extending copan:CORE, integrating LPJmL as the Earth system interface for comprehensive social-ecological simulations.*

## Overview

pycopanlpjml advances pycopancore by integrating the LPJmL model as the Earth
system interface. It provides a Python interface to LPJmL via pycoupler,
allowing to run LPJmL simulations from within a copan:LPJmL model.
The package is designed to be used in combination with pycopancore, pycoupler
and LPJmL.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pycopanlpjml.

```bash
pip install .
```

Please clone and compile [LPJmL](https://github.com/pik/LPJmL) in advance.  
Make sure to also have set the [working environment for LPJmL](https://github.com/PIK-LPJmL/LPJmL/blob/master/INSTALL) correctly if you are not working
on the PIK HPC (with Slurm Workload Manager).
The PIK python libraries [pycoupler](https://github.com/PIK-LPJmL/pycoupler) and [pycopancore](https://github.com/pik-copan/pycopancore) are required as they
serve as the basis for copan:LPJmL.

See [inseeds](https://github.com/pik-copan/inseeds/) for examples on how to
apply the framework.

## Questions / Problems

In case of questions please contact Jannes Breier jannesbr@pik-potsdam.de or [open an issue](https://github.com/pik-copan/pycopanlpjml/issues/new).

## Contributing
Merge requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). For major changes, please open an issue first to discuss what you would like to change.
