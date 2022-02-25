import os
import requests
from setuptools import find_packages, setup

#@TODO: change manifest location

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


response = requests.get("https://api.github.com/repos/supervisely/supervisely/releases/latest")
version = response.json()["tag_name"]

setup(
    name="supervisely",
    version=version,
    packages=find_packages(include=['supervisely_lib', 'supervisely_lib.*', 'supervisely', 'supervisely.*']),
    description="Supervisely Python SDK.",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/supervisely/supervisely",
    package_data={
        "": ["*.html", "*.css"], 
        "supervisely": ["video/*.sh"]
    },
    install_requires=[
        "numpy>=1.19",
        "opencv-python>=4.5.5.62",
        "PTable>=0.9.2",
        "pillow>=5.4.1",
        "protobuf>=3.14.0",
        "python-json-logger==0.1.11",
        "requests>=2.24.0",
        "requests-toolbelt>=0.9.1",
        "Shapely>=1.7.1",
        "bidict>=0.21.2",
        "varname>=0.8.1",
    ],
    extras_require={
        'extras': [
            "docker==5.0.3",
            "imgaug==0.4.0",
            "imagecorruptions==1.1.2",
            "scikit-image>=0.17.1",
            "matplotlib>=3.3.2",
            "pascal-voc-writer>=0.1.4",
            "scipy>=1.5.2",
            "sk-video>=1.1.10",
            "pandas>=1.1.3",
        ],
        'apps': [
            "uvicorn[standard]==0.17.0",
            "fastapi==0.74.0",
            "websockets==10.1",
            "jinja2==3.0.3",
            "psutil==5.9.0",
            "jsonpatch==1.32",
            "python-dotenv==0.19.2",
            "MarkupSafe==2.0.1",
            "arel==0.2.0",
        ],
        'sdk-no-usages': [
            "grpcio==1.34.1",
            "plotly>=4.11.0",
            "psutil>=5.4.5",
            "python-dotenv==0.19.2",
        ],
        # legacy dependencies
        'plugins':[
            "jsonschema>=2.6.0,<3.0.0",
        ],
        'sdk-nn-plugins': [
            "flask-restful>=0.3.7",
            "Werkzeug>=1.0.1",
        ],
    }
)
