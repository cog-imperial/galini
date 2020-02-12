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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_material

# -- Project information -----------------------------------------------------

project = 'GALINI'
copyright = '2020, Francesco Ceccon'
author = 'Francesco Ceccon'
repo_url = 'https://github.com/cog-imperial/galini'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_material',
    'sphinx.ext.mathjax'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'
html_theme_path = sphinx_material.html_theme_path()
html_context = {
    'css_files': [
        '_static/custom.css'
    ],
    **sphinx_material.get_html_context()
}

html_theme_options = {
    'color_primary': 'blue',
    'color_accent': 'cyan',
    'repo_url': repo_url,
    'repo_name': 'cog-imperial/galini',
    'nav_title': 'GALINI: an extensible MINLP solver',
    'nav_links': [
        {
            'href': 'https://optimisation.doc.ic.ac.uk/',
            'title': 'Computational Optimisation Group',
            'internal': False,
        }
    ],
    'version_dropdown': False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']