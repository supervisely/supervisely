import os

from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


# Dependencies do not include PyTorch, so
# supervisely_lib.nn.hosted.pytorch will not work out of the box.
# If you need to invoke that part of the code, it is very likely you
# already have PyTorch installed.
setup(
    name="supervisely",
    version="6.1.43",
    packages=find_packages(include=['supervisely_lib', 'supervisely_lib.*']),
    description="Supervisely Python SDK.",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/supervisely/supervisely",
    install_requires=[
        "flask-restful>=0.3.7",
        "grpcio>=1.32.0",
        "jsonschema>=2.6.0,<3.0.0",
        "matplotlib>=3.0.0",
        "numpy>=1.19.1",
        "opencv-python==3.4.10.35,<4.0.0",
        "pandas>=1.1.3",
        "pascal-voc-writer>=0.1.4",
        "PTable>=0.9.2",
        "pillow>=6.2.1",
        "protobuf>=3.7.1",
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
        "Werkzeug>=0.15.1",
        "bidict",
        "sk-video",
        "plotly==4.11.0",
        "docker",
        "psutil>=5.4.5"
    ],
)
