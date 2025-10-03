# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os

# -- Project information -- #
# ------------------------- #
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TeMFpy"
copyright = "2025, Simon Hans Hille, Attila Szabó"
author = "Simon Hans Hille, Attila Szabó"
release = "2025"

# -- General configuration -- #
# --------------------------- #
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Note that it is somehow important that napoleon comes first
# see github.com/tox-dev/sphinx-autodoc-typehints
extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",  # Latex docstring style
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",  # extracts taglines from docstrings
    "sphinx.ext.extlinks",  # shortcuts to create hyperlinks
]

# -- sphinx.ext.autodoc -- #
autodoc_member_order = "bysource"

# -- sphinx.ext.intersphinx -- #
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),  # Link to Python standard library
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "tenpy": ("https://tenpy.readthedocs.io/en/latest/", None),
    "pfapack": ("https://pfapack.readthedocs.io/en/latest", None),
}


# -- sphinx.ext.extlinks -- #
# For later


# -- Something else -- #
# --------------------- #
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -- #
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,  # Keeps the sidebar expanded
    "style_external_links": True,  # Adds external link icons
}
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]


def autodoc_process_docstring(app, what, name, obj, options, lines):
    # doc_links = {
    #     "TeNpy": (
    #         "TeNpy",
    #         "https://tenpy.readthedocs.io/en/latest/index.html",
    #     ),
    # }
    # Add more here as needed
    for i in range(len(lines)):
        # lines[i] = lines[i].replace("np.", "numpy.") # For longer links
        lines[i] = lines[i].replace("np.", "~numpy.")  # For shorter links
        # lines[i] = lines[i].replace("List[", "~typing.List[")
        # A mapping from full class path to (display name, link)
        # for full_path, (short_name, url) in doc_links.items():
        #     lines[i] = lines[i].replace(full_path, f"`{short_name} <{url}>`_")


# -- example stubs  -=-----------------------------------------------------

# ensure parent folder is in sys.path to allow import of tenpy
REPO_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
GITHUBBASE = "https://github.com/temfpy/temfpy"


def create_example_stubs():
    """create stub files for examples to include them in the documentation."""
    folders = [
        (["examples"], ".py", []),
    ]
    for subfolders, extension, excludes in folders:
        outdir = os.path.join(os.path.dirname(__file__), *subfolders)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        files = os.listdir(os.path.join(REPO_PREFIX, *subfolders))
        files = sorted(
            [fn for fn in files if fn.endswith(extension) and fn not in excludes]
        )
        for fn in files:
            outfile = os.path.join(outdir, os.path.splitext(fn)[0] + ".rst")
            if os.path.exists(outfile):
                continue
            dirs = "/".join(["src"] + subfolders)
            sentence = (
                "`on github <{base}/blob/master/{dirs!s}/{fn!s}>`_ "
                "(`download <{base}/raw/master/{dirs!s}/{fn!s}>`_)."
            )
            sentence = sentence.format(dirs=dirs, fn=fn, base=GITHUBBASE)
            include = f".. literalinclude:: /../../{dirs}/{fn}"
            text = "\n".join([fn, "=" * len(fn), "", sentence, "", include, ""])
            with open(outfile, "w") as f:
                f.write(text)


create_example_stubs()


# -- Extension configuration ------------------------------------------
def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
