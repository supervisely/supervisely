import os
import requests

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


response = requests.get("https://api.github.com/repos/supervisely/supervisely/releases/latest")
version = response.json()["tag_name"]

# Dependencies do not include PyTorch, so
# supervisely_lib.nn.hosted.pytorch will not work out of the box.
# If you need to invoke that part of the code, it is very likely you
# already have PyTorch installed.
setup(
    name="supervisely",
    version=version,
    packages=find_packages(include=['supervisely_lib', 'supervisely_lib.*', 'supervisely', 'supervisely.*']),
    description="Supervisely Python SDK.",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/supervisely/supervisely",
    install_requires=[
        "flask-restful>=0.3.7",
        "grpcio==1.34.1",
        "jsonschema>=2.6.0,<3.0.0",
        "matplotlib>=3.3.2",
        "numpy>=1.19",
        "opencv-python>=3.4.10.35",
        "pandas>=1.1.3",
        "pascal-voc-writer>=0.1.4",
        "PTable>=0.9.2",
        "pillow>=5.4.1",
        "protobuf>=3.14.0",
        # Higher python-json-logger versions are incompatible with
        # simplejson somehow, so for now prevent higher versions from
        # being installed.
        "python-json-logger==0.1.11",
        "requests>=2.24.0",
        "requests-toolbelt>=0.9.1",
        "scikit-image>=0.17.1",
        "scipy>=1.5.2",
        "Shapely>=1.7.1",
        #"simplejson>=3.17.2",
        "Werkzeug>=1.0.1",
        "bidict>=0.21.2",
        "sk-video>=1.1.10",
        "plotly>=4.11.0",
        "docker==5.0.3",
        "psutil>=5.4.5",
        "imgaug==0.4.0",
        "imagecorruptions==1.1.2",
        "python-dotenv==0.19.2"
    ],
    include_package_data=True
)
