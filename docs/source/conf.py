import sys
import os
import importlib
from pycopanlpjml._version import __version__ as copanlpjml_version


# -- Add project root to sys.path --------------------------------------------

sys.path.insert(0, os.path.abspath("../.."))

# Dynamically import pycoupler and get its path
pycoupler = importlib.import_module("pycoupler")
pycoupler_path = os.path.dirname(pycoupler.__file__)

# Add pycoupler's directory to sys.path
if pycoupler_path not in sys.path:
    sys.path.insert(0, pycoupler_path)

# Dynamically import pycoupler and get its path
pycopancore = importlib.import_module("pycopancore")
pycopancore_path = os.path.dirname(pycopancore.__file__)

# Add pycoupler's directory to sys.path
if pycopancore_path not in sys.path:
    sys.path.insert(0, pycopancore_path)


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "copan:LPJmL"
copyright = "2025, PIK-copan"
author = "PIK-copan"
version = copanlpjml_version
release = copanlpjml_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # parameters look better than with numpydoc only
    "numpydoc",
    "sphinxcontrib.mermaid",
]

# Add ablog only if not building LaTeX (ablog has LaTeX compatibility issues)
import sys
if 'latex' not in ' '.join(sys.argv).lower():
    extensions.append("ablog")

# autosummaries from source-files
autosummary_generate = True
# dont show __init__ docstring
autoclass_content = "class"
# sort class members
autodoc_member_order = "bysource"

# Skip problematic imports in autosummary
autosummary_imported_members = False

# Notes in boxes
napoleon_use_admonition_for_notes = True
# Attributes like parameters
napoleon_use_ivar = True
# keep "Other Parameters" section
# https://github.com/sphinx-doc/sphinx/issues/10330
napoleon_use_param = False

# this is a nice class-doc layout
numpydoc_show_class_members = True
# class members have no separate file, so they are not in a toctree
numpydoc_class_members_toctree = False
# maybe switch off with:    :no-inherited-members:
numpydoc_show_inherited_class_members = True
# add refs to types also in parameter lists
numpydoc_xref_param_type = True

myst_enable_extensions = [
    "colon_fence",
]

# Blog configuration (only used if ablog is loaded)
if "ablog" in extensions:
    blog_path = "blog/index"
    blog_title = "copan:LPJmL Blog"
    blog_baseurl = "https://copanlpjml.pik-potsdam.de/docs/blog/"

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_logo = "_static/logo_med.svg"
html_favicon = "_static/logo_fav.svg"

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
            "name": "copan:LPJmL home page",
            "url": "https://www.copanlpjml.pik-potsdam.de/",
            "icon": "fa-solid fa-leaf",
            "type": "fontawesome",
            "attributes": {"target": "_blank"},
        },
    ],
    "external_links": [
        {
            "name": "Get Help",
            "url": "https://github.com/pik-copan/pycopanlpjml/discussions",
        },
    ],
}

html_sidebars = {
    "blog/**": [
        "blog/postcard.html",
        "blog/recentposts.html",
        "blog/tagcloud.html",
        "blog/categories.html",
        "blog/archives.html",
    ],
}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "Python": ("https://docs.python.org/3/", None),
    "NumPy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}

# -- Warning suppression -----------------------------------------------------
# Suppress warnings from inherited xarray methods with problematic
# cross-references
suppress_warnings = [
    "autodoc",
    "autodoc.inheritance",
    "app.add_node",
    "app.add_directive",
    "app.add_role",
    "app.add_generic_role",
    "app.add_transform",
    "app.add_post_transform",
    "app.add_js_file",
    "app.add_css_file",
    "ref.docutils",  # Suppress docutils warnings
    "myst.header",   # Suppress myst header warnings
    "autosummary",   # Suppress autosummary warnings
]

# Completely disable numpydoc validation and processing
numpydoc_validation_checks = set()  # Disable all validation checks
numpydoc_show_class_members = False  # Don't show inherited members

# Skip documenting inherited methods from xarray to avoid cross-reference
# issues
autodoc_inherit_docstrings = False
autodoc_typehints = "none"  # Disable type hints processing

# Exclude ALL xarray inherited methods to avoid cross-reference issues
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,  # Changed to False to avoid inherited methods
    "show-inheritance": False,  # Don't show inheritance
    "exclude-members": "__weakref__",
}

# -- Nuclear option: Custom handler to skip xarray errors ------------------


def skip_xarray_errors(
    app, what, name, obj, options, signature, return_annotation
):
    """Skip processing of xarray methods that cause cross-reference errors."""
    try:
        if (
            hasattr(obj, "__module__")
            and obj.__module__
            and "xarray" in obj.__module__
        ):
            return None, None
    except Exception:
        pass
    return signature, return_annotation


def setup(app):
    """Custom Sphinx setup to handle xarray cross-reference errors."""
    app.connect("autodoc-process-signature", skip_xarray_errors)
