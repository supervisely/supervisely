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
        "numpy>=1.19",
        "opencv-python>=3.4.10.35",
        "PTable>=0.9.2", # prettytable, jupyterlab_scripts, key_indexed_collection (maybe can reimplement)
        "pillow>=5.4.1",
        "protobuf>=3.14.0", # dtl-legacy, plugins/nn, worker_api, app_service
        "python-json-logger==0.1.11", # !!! legacy, sly_logger, maybe we can reimplement if
        "requests>=2.24.0",
        "requests-toolbelt>=0.9.1",
        "Shapely>=1.7.1", # dtl/legacy_supervisely/..., dtl/src, polygon-crop, polyline-crop
        "bidict>=0.21.2",
        "sk-video>=1.1.10",
        "imgaug==0.4.0",
        "imagecorruptions==1.1.2",
    ],
    extras_require={
        'nn': [
            "flask-restful>=0.3.7"
        ],
        'rare': [

        ],
        'legacy': [
            "jsonschema>=2.6.0,<3.0.0", # dtl, plugins
            "matplotlib>=3.3.2", # jupyterlab_scripts, tutorials_legacy, import, nn, !!!!imaging/font.py only font??? # image.draw_text
            "pandas>=1.1.3", # jupyterlab_scripts, plugins/nn--python--, lj_api.get_activity, 
            "scipy>=1.5.2", # plugins, annotation/annotation_transforms/--extract_labels_from_mask
            "Werkzeug>=1.0.1", # supervisely/nn/inferene 
            "scikit-image>=0.17.1", # dtl/legacy_..., dtl/src, nn/markrcnn, annotation/annotation_transforms/--extract_labels_from_mask, bitmap/--skeletonize and image/rotate, image/resize_inter_nearest
        ],
        'dev': [
            "python-dotenv==0.19.2",
            "grpcio==1.34.1",
            "psutil>=5.4.5", # no sources, maybe agent?
            "docker==5.0.3", # docker utils -> move to agent
            "plotly>=4.11.0", # plugins/python/16
            "pascal-voc-writer>=0.1.4", #sdk/export,

        ]
    },
    include_package_data=True
)
