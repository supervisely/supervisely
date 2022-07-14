import os
import requests

from setuptools import find_packages, setup

# @TODO: change manifest location


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


response = requests.get(
    "https://api.github.com/repos/supervisely/supervisely/releases/latest"
)
version = response.json()["tag_name"]

# Dependencies do not include PyTorch, so
# supervisely_lib.nn.hosted.pytorch will not work out of the box.
# If you need to invoke that part of the code, it is very likely you
# already have PyTorch installed.
setup(
    name="supervisely",
    version=version,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(
        include=["supervisely_lib", "supervisely_lib.*", "supervisely", "supervisely.*"]
    ),
    description="Supervisely Python SDK.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/supervisely/supervisely",
    package_data={"": ["*.html", "*.css"], "supervisely": ["video/*.sh"]},
    install_requires=[
        "numpy>=1.19",
        "opencv-python>=4.5.5.62",
        "PTable>=0.9.2",
        "pillow>=5.4.1",
        "protobuf>=3.14.0, <=3.20.1",
        "python-json-logger==0.1.11",
        "requests>=2.27.1",
        "requests-toolbelt>=0.9.1",
        "Shapely>=1.7.1",
        "bidict>=0.21.2",
        "varname>=0.8.1",
        "python-dotenv==0.19.2",
        "pynrrd==0.4.2",
        "imgaug==0.4.0",
        "SimpleITK==2.1.1.2",
        "pydicom==2.3.0",
        "stringcase==1.2.0",
        "python-magic==0.4.25",
        "trimesh==3.11.2",
        "scikit-video==1.1.11",
        "uvicorn[standard]==0.17.0",
        "fastapi==0.74.0",
        "websockets==10.1",
        "jinja2==3.0.3",
        "psutil==5.9.0",
        "jsonpatch==1.32",
        "MarkupSafe==2.0.1",
        "arel==0.2.0",
        "tqdm==4.62.3",
        "pandas==1.4.2",
    ],
    extras_require={
        "extras": [
            "docker==5.0.3",
            "imagecorruptions==1.1.2",
            "scikit-image>=0.17.1",
            "matplotlib>=3.3.2",
            "pascal-voc-writer>=0.1.4",
            "scipy>=1.5.2",
            "sk-video>=1.1.10",
            "pandas>=1.1.3",
        ],
        "apps": [
            "uvicorn[standard]==0.17.0",
            "fastapi==0.74.0",
            "websockets==10.1",
            "jinja2==3.0.3",
            "psutil==5.9.0",
            "jsonpatch==1.32",
            "MarkupSafe==2.0.1",
            "arel==0.2.0",
            "tqdm==4.62.3",
            "pandas==1.4.2",
        ],
        "docs": [
            "sphinx==4.4.0",
            "jinja2==3.0.3",
            "sphinx-immaterial==0.4.0",
            "sphinx-copybutton==0.4.0",
            "sphinx-autodoc-typehints==1.15.3",
            "sphinxcontrib-details-directive==0.1.0",
            "myst-parser==0.18.0"
        ],
        "sdk-no-usages": [
            "grpcio==1.34.1",
            "plotly>=4.11.0",
            "psutil>=5.4.5",
        ],
        # legacy dependencies
        "plugins": [
            "jsonschema>=2.6.0,<3.0.0",
        ],
        "sdk-nn-plugins": [
            "flask-restful>=0.3.7",
            "Werkzeug>=1.0.1",
        ],
    },
)
