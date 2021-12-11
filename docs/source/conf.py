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

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "QNetOptimizer"
copyright = "2021, Brian Doolittle and Tom Bromley"
author = "Brian Doolittle and Tom Bromley"

# The full version, including alpha/beta/rc tags
release = "v0.1"

# needs_sphinx = '3.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ["sphinx.ext.autodoc"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    # 'sphinxcontrib.bibtex',
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    # "sphinx_automodapi.automodapi",
    # 'sphinx_copybutton',
    # "m2r2"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "xanadu_theme"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

show_authors = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": [
        "logo-text.html",
        "searchbox.html",
        "globaltoc.html",
        # 'sourcelink.html'
    ]
}


# # -- Xanadu theme ---------------------------------------------------------
html_theme = "xanadu_theme"
html_theme_path = ["."]

html_theme_options = {
    # Set the path to a special layout to include for the homepage
    # "index_template": "special_index.html",
    # Set the name of the project to appear in the left sidebar.
    "project_nav_name": "NISQNet.py",
    # Set your Disqus short name to enable comments
    # "disqus_comments_shortname": "pennylane-1",
    # Set you GA account ID to enable tracking
    # "google_analytics_account": "UA-130507810-2",
    # Path to a touch icon
    # "touch_icon": "logo_new.png",
    # Specify a base_url used to generate sitemap.xml links. If not
    # specified, then no sitemap will be built.
    # "base_url": ""
    # Allow a separate homepage from the master_doc
    # "homepage": "index",
    # Allow the project link to be overriden to a custom URL.
    # "projectlink": "http://myproject.url",
    "large_toc": True,
    # colors
    "navigation_button": "#19b37b",
    "navigation_button_hover": "#0e714d",
    "toc_caption": "#19b37b",
    "toc_hover": "#19b37b",
    "table_header_bg": "#edf7f4",
    "table_header_border": "#19b37b",
    "download_button": "#19b37b",
    # gallery options
    # "github_repo": "PennyLaneAI/pennylane",
    # "gallery_dirs": "tutorials",
}
