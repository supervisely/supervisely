# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import inspect
import datetime

# -- Path setup --------------------------------------------------------------
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.insert(0, os.path.abspath("../../help"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../../"))

now = datetime.date.today()

# -- Project information -----------------------------------------------------
project = "Supervisely"
copyright = f"{now.year}, Supervisely Team"
author = "Supervisely Team"

# -- General configuration ---------------------------------------------------
MOCK_IMPORTS = [
    "torch",
    "torchvision",
    "decord",
    "pycocotools",
    "sklearn",
    "plotly",
    "tensorboard",
    "tensorboardX",
    "markdown",
    "pymdown_extensions",
    "tbparse",
    "kaleido",
    "motmetrics",
    "imgaug",
    "imagecorruptions",
]

autodoc_mock_imports = MOCK_IMPORTS
autosummary_mock_imports = MOCK_IMPORTS

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_immaterial",
]

myst_all_links_external = True

source_encoding = "utf-8"
master_doc = "index"
language = "en"
default_role = "literal"

suppress_warnings = ["myst.header"]

html_use_index = False
html_copy_source = False
html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False
html_compact_lists = True

object_description_options = [
    ("py:.*", dict(include_fields_in_toc=False, include_rubrics_in_toc=False)),
    # ("py:attribute", dict(include_in_toc=False)),
    ("py:parameter", dict(include_in_toc=False)),
]

templates_path = ["_templates"]
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12/", None),
    "sphinx_docs": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True
autoclass_content = "both"
autodoc_inherit_docstrings = False
add_module_names = False
autodoc_member_order = "groupwise"
autodoc_class_signature = "mixed"

html_domain_indices = True
autodoc_typehints = "none"
source_suffix = {".rst": "restructuredtext", ".txt": "restructuredtext", ".md": "markdown"}

autodoc_default_options = {
    "members": True,
    "methods": True,
    "show-inheritance": True,
}


html_css_files = ["css/custom.css"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
html_title = "Supervisely SDK for Python"
html_theme = "sphinx_immaterial"
html_favicon = "_static/images/favicon.ico"
html_logo = "_static/images/sly-top-logo-white.png"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "view": "material/file-eye-outline",
    },
    "site_url": "https://supervisely.com/",
    "repo_url": "https://github.com/supervisely/supervisely",
    "repo_name": "Supervisely",
    "globaltoc_collapse": True,
    "features": [
        # Content UX
        "content.code.copy",
        "content.tooltips",
        "content.action.view",
        # Navigation UX
        "navigation.sections",
        "navigation.top",
        "navigation.footer",
        "navigation.tracking",
        # Search UX
        "search.share",
        "search.highlight",
        "search.suggest",
        # TOC UX
        "toc.follow",
        "toc.sticky",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
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
                "name": "Switch to system preference",
            },
        },
    ],
    "version_dropdown": False,
    "toc_title_is_page_title": True,
}
