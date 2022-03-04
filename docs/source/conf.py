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

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.insert(0, os.path.abspath('../../help'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../../'))

# -- Project information -----------------------------------------------------
project = 'Supervisely'
copyright = '2022, Supervisely Team'
author = 'Supervisely Team'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
    'm2r2',
    'nbsphinx',
    'nbsphinx_link',

    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    "sphinx.ext.autosectionlabel",
    "sphinx_material"
]

# Override default of `utf-8-sig` which can cause problems with autosummary due
# to the extra Unicode Byte Order Mark that gets inserted.
source_encoding = 'utf-8'
master_doc = 'index'
language = 'en'
html_use_index = False

# Don't include "View page source" links, since they aren't very helpful,
# especially for generated pages.
html_copy_source = False

# Skip unnecessary footer text.
html_show_sphinx = False
html_show_copyright = True

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8/", None),
    "numpy": ('https://numpy.org/doc/stable/', ('intersphinx_inv/numpy.inv', None))
}

rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight
.. role:: json(code)
   :language: json
   :class: highlight
"""

templates_path = ['_templates']
html_static_path = ['_static']

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "class"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = True  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = False  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
add_module_names = False  # Remove namespaces from class/method signatures
autodoc_member_order = 'groupwise'
# autodoc_member_order = 'bysource'

autodoc_class_signature = 'separated'
autodoc_default_options = {
    "members": True,
    "methods": True,
    "exclude-members": "__init__",
    "show-inheritance": True,
}

jupyter_generate_html = True
# Add any paths that contain templates here, relative to this directory.


html_css_files = [
    'css/custom.css',
]

html_sidebars = {
    "**": ['fulltoc.html', 'sourcelink.html', 'relations.html', 'searchbox.html', "logo-text.html", "globaltoc.html",
           "localtoc.html", "navigation.html"]
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/images/sly-top-logo-white.png'

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
    'color_primary': 'pink',
    'color_accent': 'light-blue',

    'site_url': 'https://supervise.ly',
    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/supervisely/supervisely',
    'repo_name': 'Supervisely',
    'repo_type': 'github',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': -1,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True,

    'features': [
        'navigation.expand',
        # 'navigation.tabs',
        # 'toc.integrate',
        'navigation.sections',
        # 'navigation.instant',
        # 'header.autohide',
        'navigation.top',
        # 'search.highlight',
        # 'search.share',
    ],

    'palette': [
        {
            'media': '(prefers-color-scheme: dark)',
            'scheme': 'slate',
            'primary': 'green',
            'accent': 'light blue',
            'toggle': {
                'icon': 'material/lightbulb',
                'name': 'Switch to light mode',
            },
        },
        {
            'media': '(prefers-color-scheme: light)',
            'scheme': 'default',
            'primary': 'green',
            'accent': 'light blue',
            'toggle': {
                'icon': 'material/lightbulb-outline',
                'name': 'Switch to dark mode',
            },
        },
    ],


    'html_prettify': False,
    'html_minify': True,
    'css_minify': True,

    "nav_links": [
        # {"href": "index", "internal": True, "title": "Home"},
        {
            "href": "https://supervise.ly",
            "internal": False,
            "title": "Supervisely Platform",
        },
        {
            "href": "https://github.com/supervisely-ecosystem",
            "internal": False,
            "title": "Ecosystem",
        },
        {
            "href": "https://docs.supervise.ly",
            "internal": False,
            "title": "UI Documentation",
        },
        {
            "href": "https://www.youtube.com/c/Supervisely/videos",
            "internal": False,
            "title": "YouTube",
        },
    ],

    # "version_dropdown": True,
    # "version_json": "_static/versions.json",
    # "version_info": {
    #     "Release": "https://bashtage.github.io/sphinx-material/",
    #     "Development": "https://bashtage.github.io/sphinx-material/devel/",
    #     "Release (rel)": "/sphinx-material/",
    #     "Development (rel)": "/sphinx-material/devel/",
    # }
}

# If false, no module index is generated.
html_domain_indices = False

# Disable typehints
autodoc_typehints = "none"

# source_suffix
# The file extensions of source files. Sphinx considers the files with this suffix as sources.
# The value can be a dictionary mapping file extensions to file types. For example:
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown'
}
