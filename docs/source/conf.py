# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import inspect

# -- Path setup --------------------------------------------------------------
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.insert(0, os.path.abspath("../../help"))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../../"))

# -- Project information -----------------------------------------------------
project = "Supervisely"
copyright = "2024, Supervisely Team"
author = "Supervisely Team"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

# myst_enable_extensions = [
#     "amsmath",
#     "colon_fence",
#     "deflist",
#     "dollarmath",
#     "fieldlist",
#     "html_admonition",
#     "html_image",
#     "linkify",
#     "replacements",
#     "smartquotes",
#     "strikethrough",
#     "substitution",
#     "tasklist",
# ]

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

templates_path = ["_templates"]
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12/", None),
    "sphinx_docs": ("https://www.sphinx-doc.org/en/master", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True
autoclass_content = "class"
html_show_sourcelink = True
autodoc_inherit_docstrings = False
nbsphinx_allow_errors = True
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
    "exclude-members": "from_bytes,to_bytes",
}


def _strip_defaults_from_signature(sig: inspect.Signature) -> inspect.Signature:
    """Return signature with defaults removed to stabilize param-name matching.

    Sphinx can emit warnings like:
    "Parameter name 'foo' does not match any of the parameters defined in the signature:
    ['foo=<built-in function ...>']"
    when defaults are rendered into the signature string.
    """
    params = []
    for p in sig.parameters.values():
        if p.default is inspect._empty:
            params.append(p)
        else:
            params.append(p.replace(default=inspect._empty))
    return sig.replace(parameters=params)


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Normalize signatures to avoid docstring/param mismatch noise."""
    try:
        sig = inspect.signature(obj)
    except Exception:
        return (signature, return_annotation)

    try:
        sig = _strip_defaults_from_signature(sig)
    except Exception:
        return (signature, return_annotation)

    return (str(sig), return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", process_signature)


html_css_files = [
    "css/custom.css",
]

html_sidebars = {
    "**": [
        "fulltoc.html",
        "sourcelink.html",
        "relations.html",
        "searchbox.html",
        "logo-text.html",
        "globaltoc.html",
        "localtoc.html",
        "navigation.html",
    ]
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
extensions.append("sphinx_immaterial")
html_title = "Supervisely SDK for Python"
html_theme = "sphinx_immaterial"
html_favicon = "_static/images/favicon.ico"
html_logo = "_static/images/sly-top-logo-white.png"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://supervisely.com/",
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
