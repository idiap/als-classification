# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'ALS and Stress classification'
copyright = '2021, Colombine Verzat'
author = 'Colombine Verzat'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.programoutput",
    "sphinx_click.ext"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# external dependencies
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'pandas': ('http://pandas.pydata.org/pandas-docs/stable', None)}


autodoc_mock_imports = ['pandas', 'sklearn', 'skimage', 'torchvision', 'yaml']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True


# Default processing flags for sphinx
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]
