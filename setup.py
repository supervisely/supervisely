import os
import re
import subprocess

import requests
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

# @TODO: change manifest location


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fin:
        return fin.read()


def get_common_commit_with_master():
    result = subprocess.run(["git", "merge-base", "HEAD", "master"], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip()


def get_previous_commit(sha: str):
    result = subprocess.run(["git", "rev-parse", sha + "^"], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip()


def get_branch():
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip()


def get_commit_tags(sha: str):
    result = subprocess.run(["git", "tag", "--points-at", sha], stdout=subprocess.PIPE)
    return [t for t in result.stdout.decode("utf-8").strip().split("\n") if t]


def get_github_releases():
    response = requests.get("https://api.github.com/repos/supervisely/supervisely/releases")
    response.raise_for_status()
    return response.json()


def get_release_commit(tag: str):
    response = requests.get(
        "https://api.github.com/repos/supervisely/supervisely/git/ref/tags/" + tag
    )
    response.raise_for_status()
    return response.json()["object"]["sha"]


def get_version():
    version = os.getenv("RELEASE_VERSION", None)
    if version is not None:
        return version
    branch_name = get_branch()
    gh_releases = get_github_releases()
    commit = get_common_commit_with_master()
    release_commits = {}
    while commit:
        if get_commit_tags(commit):
            for release in gh_releases:
                release_commit = release_commits.setdefault(
                    release["tag_name"], get_release_commit(release["tag_name"])
                )
                if release_commit == commit:
                    if branch_name != "master":
                        return release["tag_name"] + "+" + branch_name
                    return release["tag_name"]
        commit = get_previous_commit(commit)

    response = requests.get("https://api.github.com/repos/supervisely/supervisely/releases/latest")
    version = response.json()["tag_name"]
    return version


version = get_version()


INSTALL_REQUIRES = [
    "cachetools>=4.2.3, <=5.5.0",
    "numpy>=1.19, <=2.3.3",
    "opencv-python>=4.6.0.66, <5.0.0.0",
    "PTable>=0.9.2, <1.0.0",
    "pillow>=5.4.1, <=10.4.0",
    "python-json-logger>=0.1.11, <3.0.0",
    "requests>=2.27.1, <3.0.0",
    "requests-toolbelt>=0.9.1",  # , <1.0.0
    "Shapely>=1.7.1, <=2.1.2",
    "bidict>=0.21.2, <1.0.0",
    "varname>=0.8.1, <1.0.0",
    "python-dotenv>=0.19.2, <=1.0.1",
    "pynrrd>=0.4.2, <1.0.0",
    "SimpleITK>=2.1.1.2, <=2.4.1.0",  # 2.5.0 does not have packaging for python 3.8
    "pydicom>=2.3.0, <3.0.0",
    "stringcase>=1.2.0, <2.0.0",
    "python-magic>=0.4.25, <1.0.0",
    "trimesh>=3.11.2, <=4.5.0",
    "uvicorn[standard]>=0.18.2, <1.0.0",
    "starlette<=0.47.3",  # if update to 0.48.0+ change supervisely/app/fastapi/custom_static_files.py line 45
    "pydantic>=1.7.4, <=2.12.3",
    "fastapi>=0.103.1, <=0.119.1",
    "websockets>=10.3, <=13.1",
    "jinja2>=3.0.3, <4.0.0",
    "psutil>=5.9.0, <6.0.0",
    "jsonpatch>=1.32, <2.0",
    "MarkupSafe>=2.1.1, <3.0.0",
    "arel>=0.2.0, <1.0.0",
    "tqdm>=4.62.3, <5.0.0",
    "pandas>=1.1.3, <=2.3.3",
    "async_asgi_testclient",
    "PyYAML>=5.4.0",
    "distinctipy",
    "beautifulsoup4",
    "numerize",
    "ffmpeg-python==0.2.0",
    "python-multipart>=0.0.5, <=0.0.12",
    "GitPython",
    "giturlparse",
    "rich",
    "click",
    "imutils==0.5.4",
    "urllib3>=1.26.15, <=2.2.3",
    "cacheout==0.14.1",
    "jsonschema>=2.6.0,<=4.23.0",
    "pyjwt>=2.1.0,<3.0.0",
    "zstd",
    "aiofiles",
    "httpx[http2]==0.27.2",
    "debugpy",
    "setuptools<81.0.0",
]

ALT_INSTALL_REQUIRES = {
    "opencv-python>=4.6.0.66, <5.0.0.0": [
        "opencv-python-headless>=4.8.1.78",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ],
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
    If that is the case, replace the install require by the alternative to not install dual package
    """
    install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires:
            main_require = check_alternative_installation(
                main_require, alternative_requires.get(main_require)
            )
        install_requires.append(main_require)

    return install_requires


# Dependencies do not include PyTorch, so
# supervisely_lib.nn.hosted.pytorch will not work out of the box.
# If you need to invoke that part of the code, it is very likely you
# already have PyTorch installed.
setup(
    name="supervisely",
    maintainer="Max Kolomeychenko",
    version=version,
    description="Supervisely Python SDK.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Supervisely",
    author_email="support@supervisely.com",
    url="https://github.com/supervisely/supervisely",
    packages=find_packages(
        include=["supervisely_lib", "supervisely_lib.*", "supervisely", "supervisely.*"]
    ),
    package_data={
        "": [
            "*.html",
            "*.jinja",
            "*.css",
            "*.js",
            "*.md",
        ],
        "supervisely": [
            "versions.json",
            "video/*.sh",
            "app/development/*.sh",
            "imaging/colors.json.gz",
            "nn/benchmark/*/*.yaml",
            "nn/tracker/botsort/botsort_config.yaml",
        ],
    },
    entry_points={
        "console_scripts": [
            "sly-release=supervisely.release.run:cli_run",
            "supervisely=supervisely.cli.cli:cli",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=get_install_requirements(INSTALL_REQUIRES, ALT_INSTALL_REQUIRES),
    extras_require={
        "extras": [
            "docker>=5.0.3, <6.0.0",
            "imagecorruptions>=1.1.2, <2.0.0",
            "scikit-image>=0.17.1, <1.0.0",
            "matplotlib>=3.3.2, <4.0.0",
            "pascal-voc-writer>=0.1.4, <1.0.0",
            "scipy>=1.8.0, <2.0.0",
            "pandas>=1.1.3, <=2.3.3",
            "ruamel.yaml==0.17.21",
        ],
        "apps": [
            "uvicorn[standard]>=0.18.2, <1.0.0",
            "fastapi>=0.79.0, <1.0.0",
            "websockets>=10.3, <=13.1",
            "jinja2>=3.0.3, <4.0.0",
            "psutil>=5.9.0, <6.0.0",
            "jsonpatch>=1.32, <2.0",
            "MarkupSafe>=2.1.1, <3.0.0",
            "arel>=0.2.0, <1.0.0",
            "tqdm>=4.62.3, <5.0.0",
            "pandas>=1.1.3, <=2.3.3",
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
            "grpcio>=1.53.2, <2.0.0",
            "plotly>=4.11.0, <6.0.0",
            "psutil>=5.4.5, <6.0.0",
        ],
        "tracking": [
            "yacs",
            "matplotlib>=3.3.2, <4.0.0",
            "scipy>=1.8.0, <2.0.0",
            "lap",
            "cython_bbox",
            "termcolor",
            "scikit-learn",
            "faiss-gpu",  # Not supported in Python 3.11+
            "tabulate",
            "tensorboard",
            "decord",
            "gdown",
            "torch",
            "motmetrics",
        ],
        "model-benchmark": [
            "pycocotools",
            "scikit-learn",
            "plotly==5.22.0",
            "torch",
            "torchvision",
        ],
        "training": [
            "pycocotools",
            "scikit-learn",
            "plotly==5.22.0",
            "torch",
            "torchvision",
            "tensorboardX",
            "markdown",
            "pymdown-extensions",
            "tbparse",
            "kaleido==0.2.1",
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
            "imagecorruptions>=1.1.2, <2.0.0",
            "numpy>=1.19, <2.0.0",
        ],
        "agent": [
            "protobuf>=3.19.5, <=3.20.3",
        ],
    },
)
