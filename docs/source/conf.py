# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# add docs path to python sys.path to allow autodoc-ing a test_py_module
# sourcery skip: merge-list-append, move-assign-in-block
import os
import sys

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.insert(0, os.path.abspath('../../help'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../../'))

# sys.setrecursionlimit(10000)


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

    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinxcontrib.details.directive",
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
    "sphinx_docs": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ('https://numpy.org/doc/stable/', ('intersphinx_inv/numpy.inv', None))
}

default_role = "any"

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

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

extensions.append("sphinx_immaterial")
html_title = "Supervisely SDK"
html_theme = "sphinx_immaterial"
html_favicon = '_static/images/favicon.ico'
html_logo = '_static/images/sly-top-logo-white.png'

# material theme options (see theme.conf for more information)
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://supervise.ly/",
    "repo_url": "https://github.com/supervisely/supervisely",
    "repo_name": "Supervisely",
    "repo_type": "github",
    # "edit_uri": "blob/main/docs",
    "google_analytics": ["UA-XXXXX", "auto"],
    "globaltoc_collapse": True,
    "features": [
        # "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "pink",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "pink",
            "accent": "deep-orange",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    "version_dropdown": False,
    # "version_info": [
    #     {
    #         "version": "https://sphinx-immaterial.rtfd.io",
    #         "title": "Supervisely",
    #         "aliases": []
    #     },
    #     {
    #         "version": "https://jbms.github.io/sphinx-immaterial",
    #         "title": "Supervisely Ecosystem",
    #         "aliases": []
    #     },
    #     {
    #         "version": "https://jbms.github.io/sphinx-immaterial",
    #         "title": "YouTube",
    #         "aliases": []
    #     },
    # ],
    "toc_title_is_page_title": False,
}  # end html_theme_options

# If false, no module index is generated.
html_domain_indices = True

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
