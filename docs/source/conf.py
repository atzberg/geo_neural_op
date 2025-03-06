# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Go 2 directories up to your project root
sys.path.insert(0, os.path.abspath('../..'))

print("sys.path:", sys.path)  # Debug output to make sure your path is correct




project = 'Geometric Neural Operator'
copyright = '2025, Blaine Quackenbush, Paul J. Atzberger'
author = 'Blaine Quackenbush, Paul J. Atzberger'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]
autosummary_generate = True
napoleon_numpy_docstring = True

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
