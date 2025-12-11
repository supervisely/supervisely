import importlib
import platform
import re
import importlib

from supervisely.sly_logger import logger

MACOS, LINUX, WINDOWS = (
    platform.system() == x for x in ["Darwin", "Linux", "Windows"]
)  # environment booleans


def check_version(
    package: str,
    required: str = "0.0.0",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    try:
        current = importlib.metadata.version(package)  # get version string from package name
    except importlib.metadata.PackageNotFoundError as e:
        if hard:
            raise ModuleNotFoundError(f"{current} package is required but not installed") from e
        else:
            return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(
            r"([^0-9]*)([\d.]+)", r
        ).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"{package}{required} is required, but {package}=={current} is currently installed. {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            logger.warning(warning)
    return result


def parse_version(version: str = "0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        logger.warning(f"failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def install_dependency(dependency: str, version: str = ""):
    try:
        import subprocess

        package = f"{dependency}"
        if version:
            package += f"=={version}"
        subprocess.check_output(
            f"pip install --no-cache-dir {package}",
            shell=True,
            stderr=subprocess.STDOUT,
            text=True,
        )

    except Exception as e:
        logger.error(f"Failed to install dependency {dependency}: {e}")
        raise
