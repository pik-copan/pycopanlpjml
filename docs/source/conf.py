import inspect
from enum import Enum
from pycopanlpjml import __version__ as copanlpjml_version


# Function to filter members (e.g., to exclude Enum internals)
def skip_member(app, what, name, obj, skip, options):
    if inspect.isclass(obj) and issubclass(obj, Enum):
        if name not in [e.name for e in obj]:
            return True  # Skip anything that isn’t an enum member
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)


# -- Project information -----------------------------------------------------
project = "copan:LPJmL"
copyright = "2025, PIK copan"
author = "PIK copan"
version = copanlpjml_version
release = copanlpjml_version

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinxcontrib.mermaid",
    "matplotlib.sphinxext.plot_directive",
    "ablog",
]

# Autosummary settings
autosummary_generate = True
autoclass_content = "class"  # Don't repeat __init__ docstrings
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_use_admonition_for_notes = True
napoleon_use_ivar = True
napoleon_use_param = False

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = True
numpydoc_xref_param_type = True

# Plot settings
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["svg"]

# Markdown settings (MyST)
myst_enable_extensions = [
    "colon_fence",
]

# -- Paths -------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_logo = "_static/logo_large.svg"
html_favicon = "_static/logo.svg"

html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "header_links_before_dropdown": 6,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "Source code",
            "url": "https://github.com/pik-copan/pycopanlpjml",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "LPJmL documentation",
            "url": "https://www.pik-potsdam.de/research/projects/lpjml",
            "icon": "fa-solid fa-leaf",
            "type": "fontawesome",
            "attributes": {"target": "_blank"},
        },
    ],
}

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "Python": ("https://docs.python.org/3/", None),
    "NumPy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}
