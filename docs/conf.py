import pathlib
import sys
from datetime import datetime

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

HERE = pathlib.Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]

import sift  # noqa: E402

needs_sphinx = "4.0"

# -- Project information -----------------------------------------------------

project = "sift-sc"
author = "Zoe Piran"
copyright = f"{datetime.now():%Y}, {author}"
release = sift.__version__
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "pykeops": ("https://www.kernel-operations.io/keops/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

master_doc = "index"
source_suffix = [".rst"]
templates_path = ["_templates"]

# syntax highlight
pygments_style = "default"
pygments_dark_style = "native"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# theme
html_theme = "furo"
html_title = "SiFT"
html_static_path = ["_static"]
html_logo = "_static/img/sift_gc.png"
html_css_files = ["css/override.css"]
html_show_sphinx = False
html_show_sourcelink = True
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}

nbsphinx_codecell_lexer = "ipython3"
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. raw:: html

    <div class="docutils container">
        <a class="reference external"
           href="https://colab.research.google.com/github/nitzanlab/sift-sc/blob/main/{{ docname|e }}">
        <img alt="Open in Colab" src="_static/img/colab-badge.svg" width="125px">
        </a>
    </div>
"""

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"
