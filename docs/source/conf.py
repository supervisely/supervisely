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

# PATH_HERE = os.path.abspath(os.path.dirname(__file__))
# PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
# sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.insert(0, os.path.abspath('./repo'))
# sys.path.insert(0, os.path.abspath('../../supervisely_lib'))
# sys.path.insert(0, '/app')


# -- Project information -----------------------------------------------------

# project = 'Supervisely SDK for Python'
project = 'Supervisely'
copyright = '2021, Supervisely Team'
author = 'Supervisely Team'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'm2r2',
    'nbsphinx'
]

jupyter_generate_html = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/images/sly-top-logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/images/favicon.ico'  # need fix later

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_material'
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'Supervisely',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://github.com/supervisely/supervisely',

    # Set the color and the accent color
    'theme_color': 'ffffff',
    'color_primary': 'blue',
    'color_accent': 'light-blue',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/supervisely/supervisely',
    'repo_name': 'Supervisely',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 3,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True,

    'html_prettify': False,
    'html_minify': True,
    'css_minify': True,

    "nav_links": [
        {"href": "index", "internal": True, "title": "Material"},
        {
            "href": "https://supervise.ly/",
            "internal": False,
            "title": "Supervisely Platform",
        },
    ],

    "version_dropdown": True,
    "version_json": "_static/versions.json",
    "version_info": {
        "Release": "https://bashtage.github.io/sphinx-material/",
        "Development": "https://bashtage.github.io/sphinx-material/devel/",
        "Release (rel)": "/sphinx-material/",
        "Development (rel)": "/sphinx-material/devel/",
    },
    "table_classes": ["plain"],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# source_suffix
# The file extensions of source files. Sphinx considers the files with this suffix as sources.
# The value can be a dictionary mapping file extensions to file types. For example:
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown'
}
# docs order
# disable for alphabetical order, enable for order from source .py files
autodoc_member_order = 'groupwise'
# autodoc_member_order = 'bysource'

autosummary_generate = True

autoclass_content = "class"
autodoc_class_signature = 'mixed'


autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

autosectionlabel_prefix_document = True


