import os
import re
import requests
from pkg_resources import DistributionNotFound, get_distribution

from setuptools import find_packages, setup

# @TODO: change manifest location


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


response = requests.get(
    "https://api.github.com/repos/supervisely/supervisely/releases/latest"
)
version = response.json()["tag_name"]


INSTALL_REQUIRES = [
    "numpy>=1.19, <2.0.0",
    "opencv-python>=4.5.5.62, <5.0.0.0",
    "PTable>=0.9.2, <1.0.0",
    "pillow>=5.4.1, <10.0.0",
    "protobuf>=3.14.0, <=3.20.3",
    "python-json-logger>=0.1.11, <3.0.0",
    "requests>=2.27.1, <3.0.0",
    "requests-toolbelt>=0.9.1, <1.0.0",
    "Shapely>=1.7.1, <2.0.0",
    "bidict>=0.21.2, <1.0.0",
    "varname>=0.8.1, <1.0.0",
    "python-dotenv>=0.19.2, <1.0.0",
    "pynrrd>=0.4.2, <1.0.0",
    "SimpleITK>=2.1.1.2, <3.0.0.0",
    "pydicom>=2.3.0, <3.0.0",
    "stringcase>=1.2.0, <2.0.0",
    "python-magic>=0.4.25, <1.0.0",
    "trimesh>=3.11.2, <4.0.0",
    "scikit-video>=1.1.11, <2.0.0",
    "uvicorn[standard]>=0.18.2, <1.0.0",
    "fastapi>=0.79.0, <1.0.0",
    "websockets>=10.3, <11.0",
    "jinja2>=3.0.3, <4.0.0",
    "psutil>=5.9.0, <6.0.0",
    "jsonpatch>=1.32, <2.0",
    "MarkupSafe>=2.1.1, <3.0.0",
    "arel>=0.2.0, <1.0.0",
    "tqdm>=4.62.3, <5.0.0",
    "pandas>=1.1.3, <=1.5.2",  # For compatibility with Python3.7
    "async_asgi_testclient",
    "PyYAML",
    "distinctipy",
    "beautifulsoup4",
    "numerize",
    "ffmpeg-python==0.2.0",
    "python-multipart==0.0.5",
]

ALT_INSTALL_REQUIRES = {
    "opencv-python>=4.5.5.62, <5.0.0.0": ["opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"],
}


def check_alternative_installation(install_require, alternative_install_requires):
    """If some version version of alternative requirement installed, return alternative,
    else return main.
    """
    for alternative_install_require in alternative_install_requires:
        try:
            alternative_pkg_name = re.split(r"[ !<>=]", alternative_install_require)[0]
            get_distribution(alternative_pkg_name)
            return str(alternative_install_require)
        except DistributionNotFound:
            continue

    return str(install_require)


def get_install_requirements(main_requires, alternative_requires):
    """Iterates over all install requires
    If an install require has an alternative option, check if this option is installed
    If that is the case, replace the install require by the alternative to not install dual package"""
    install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires:
            main_require = check_alternative_installation(main_require, alternative_requires.get(main_require))
        install_requires.append(main_require)

    return install_requires


# Dependencies do not include PyTorch, so
# supervisely_lib.nn.hosted.pytorch will not work out of the box.
# If you need to invoke that part of the code, it is very likely you
# already have PyTorch installed.
setup(
    name="supervisely",
    version=version,
    python_requires=">=3.7.1",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(
        include=["supervisely_lib", "supervisely_lib.*", "supervisely", "supervisely.*"]
    ),
    description="Supervisely Python SDK.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/supervisely/supervisely",
    package_data={
        "": ["*.html", "*.css", "*.js"],
        "supervisely": ["video/*.sh", "app/development/*.sh", "imaging/colors.json.gz"],
    },
    install_requires=get_install_requirements(
        INSTALL_REQUIRES, ALT_INSTALL_REQUIRES
    ),
    extras_require={
        "extras": [
            "docker>=5.0.3, <6.0.0",
            "imagecorruptions>=1.1.2, <2.0.0",
            "scikit-image>=0.17.1, <1.0.0",
            "matplotlib>=3.3.2, <4.0.0",
            "pascal-voc-writer>=0.1.4, <1.0.0",
            "scipy>=1.5.2, <2.0.0",
            "sk-video>=1.1.10, <2.0.0",
            "pandas>=1.1.3, <1.4.0",
            "ruamel.yaml==0.17.21",
        ],
        "apps": [
            "uvicorn[standard]>=0.18.2, <1.0.0",
            "fastapi>=0.79.0, <1.0.0",
            "websockets>=10.3, <11.0",
            "jinja2>=3.0.3, <4.0.0",
            "psutil>=5.9.0, <6.0.0",
            "jsonpatch>=1.32, <2.0",
            "MarkupSafe>=2.1.1, <3.0.0",
            "arel>=0.2.0, <1.0.0",
            "tqdm>=4.62.3, <5.0.0",
            "pandas>=1.1.3, <1.4.0",
        ],
        "docs": [
            "sphinx==4.4.0",
            "jinja2==3.0.3",
            "sphinx-immaterial==0.4.0",
            "sphinx-copybutton==0.4.0",
            "sphinx-autodoc-typehints==1.15.3",
            "sphinxcontrib-details-directive==0.1.0",
            "myst-parser==0.18.0",
        ],
        "sdk-no-usages": [
            "grpcio>=1.34.1, <2.0.0",
            "plotly>=4.11.0, <6.0.0",
            "psutil>=5.4.5, <6.0.0",
        ],
        # legacy dependencies
        "plugins": [
            "jsonschema>=2.6.0,<3.0.0",
        ],
        "sdk-nn-plugins": [
            "flask-restful>=0.3.7, <1.0.0",
            "Werkzeug>=1.0.1, <3.0.0",
        ],
        "aug": [
            "imgaug>=0.4.0, <1.0.0",
        ]
    },
)
