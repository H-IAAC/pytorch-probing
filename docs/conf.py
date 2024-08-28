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
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'Pytorch Probing'
copyright = '2024, H.IAAC'
author = 'EltonCN'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', #Docs modules
    'sphinx_mdinclude', #Markdown
    'sphinx.ext.napoleon', #NumPy/Google Docs Styles
    'sphinx.ext.viewcode', #Source code
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "README.md"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "collapse_navigation" : False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

source_suffix = ['.rst', '.md']

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):

    app.config.m2r_parse_relative_links = True
    app.connect("autodoc-skip-member", skip)


sys.path.insert(0, os.path.abspath('.'))

nbsphinx_execute = 'never'

# Parse Markdown files, creating copys with corrected links
#from markdown_parser import parse_files
#parse_files()

import shutil

print("Copy example notebooks into docs/_examples")

def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir,c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result

project_root = "../"

shutil.rmtree(os.path.join(project_root, "docs/_examples"), ignore_errors=True)
shutil.copytree(os.path.join(project_root, "examples"),
                os.path.join(project_root, "docs/_examples"),
                ignore=all_but_ipynb)

try:
    os.remove("README.md")
except:
    pass

shutil.copyfile(os.path.join(project_root, "README.md"), os.path.join(project_root, "docs/README.md"))
