import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="supervisely",
    version="0.0.1dev",
    packages=find_packages(),
    description="Supervisely library.",
    long_description=read("README.md"),
    url="https://github.com/supervisely/supervisely",
    install_requires=[
        "requests>=2.22.0",
        "requests-toolbelt>=0.9.1",
        "simplejson>=3.16.0",
        "python-json-logger>=0.1.11",
        "PrettyTable",
        "Shapely",
        "pascal-voc-writer",
    ],
)
