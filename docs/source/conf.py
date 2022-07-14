# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
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
]

source_encoding = 'utf-8'
master_doc = 'index'
language = 'en'
default_role = "any"

html_use_index = False
html_copy_source = False
html_show_sphinx = False
html_show_copyright = True

templates_path = ['_templates']
html_static_path = ['_static']

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8/", None),
    "sphinx_docs": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ('https://numpy.org/doc/stable/', ('intersphinx_inv/numpy.inv', None)),
}

autosummary_generate = True
autoclass_content = "class"
html_show_sourcelink = True
autodoc_inherit_docstrings = False
set_type_checking_flag = True
nbsphinx_allow_errors = True
add_module_names = False
autodoc_member_order = 'groupwise'
autodoc_class_signature = 'separated'

html_domain_indices = True
autodoc_typehints = "none"
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown'
}

autodoc_default_options = {
    "members": True,
    "methods": True,
    "exclude-members": "__init__",
    "show-inheritance": True,
}

html_css_files = [
    'css/custom.css',
]

html_sidebars = {
    "**": ['fulltoc.html', 'sourcelink.html', 'relations.html', 'searchbox.html', "logo-text.html", "globaltoc.html",
           "localtoc.html", "navigation.html"]
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
extensions.append("sphinx_immaterial")
html_title = "Supervisely SDK for Python"
html_theme = "sphinx_immaterial"
html_favicon = '_static/images/favicon.ico'
html_logo = '_static/images/sly-top-logo-white.png'

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://supervise.ly/",
    "repo_url": "https://github.com/supervisely/supervisely",
    "repo_name": "Supervisely",
    "repo_type": "github",
    "google_analytics": ["UA-XXXXX", "auto"],
    "globaltoc_collapse": True,
    "features": [
        "navigation.sections",
        "navigation.top",
        "search.share",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "light-blue",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "light-blue",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    "version_dropdown": False,
    "toc_title_is_page_title": True,
}
