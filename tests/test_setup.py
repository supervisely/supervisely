import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime


class SetupTests(unittest.TestCase):
    """Test installation of supervisely package with different extras_require options.

    SAFETY GUARANTEES:
    - All tests run in ISOLATED temporary virtual environments
    - System Python and its packages are NEVER modified
    - All virtual environments are created in /tmp with unique names
    - Virtual environments are automatically cleaned up after tests
    - Package installation uses -e (editable) mode for local development
    """

    # List of all extras_require options from setup.py
    EXTRAS_OPTIONS = [
        "extras",
        "apps",
        "docs",
        "sdk-no-usages",
        "tracking",
        "model-benchmark",
        "training",
        "plugins",
        "sdk-nn-plugins",
        "aug",
        "agent",
    ]

    # Python versions to test (can be overridden by command line argument)
    PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]

    # Filtered versions for testing (set by command line arguments)
    TEST_VERSIONS = None

    # Results storage
    test_results = {
        "start_time": None,
        "end_time": None,
        "python_versions": {},
        "tests": [],
        "summary": {"total": 0, "passed": 0, "failed": 0, "errors": 0},
    }

    @classmethod
    def _save_results(cls):
        """Save test results to a JSON file."""
        cls.test_results["end_time"] = datetime.now().isoformat()

        # Create results directory if it doesn't exist
        results_dir = os.path.join(cls.project_root, "test_results")
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"setup_test_results_{timestamp}.json")

        # Save JSON results
        with open(results_file, "w") as f:
            json.dump(cls.test_results, f, indent=2)

        # Also create a human-readable report
        report_file = os.path.join(results_dir, f"setup_test_report_{timestamp}.txt")
        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SUPERVISELY SDK INSTALLATION TEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Date: {cls.test_results['start_time']}\n")
            f.write(f"Test Duration: {cls.test_results['end_time']}\n\n")

            f.write("Python Versions Tested:\n")
            f.write("-" * 80 + "\n")
            for version, info in cls.test_results["python_versions"].items():
                f.write(f"  Python {version}: {info}\n")
            f.write("\n")

            f.write("Test Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Tests: {cls.test_results['summary']['total']}\n")
            f.write(f"  Passed: {cls.test_results['summary']['passed']}\n")
            f.write(f"  Failed: {cls.test_results['summary']['failed']}\n")
            f.write(f"  Errors: {cls.test_results['summary']['errors']}\n\n")

            f.write("Detailed Results:\n")
            f.write("-" * 80 + "\n")
            for test in cls.test_results["tests"]:
                status_symbol = "✓" if test["status"] == "PASS" else "✗"
                f.write(f"{status_symbol} {test['name']} (Python {test['python_version']})\n")
                f.write(f"   Status: {test['status']}\n")
                if test.get("error"):
                    f.write(f"   Error: {test['error']}\n")
                f.write("\n")

        print(f"\n{'='*80}")
        print(f"Test results saved to:")
        print(f"  JSON: {results_file}")
        print(f"  Report: {report_file}")
        print(f"{'='*80}\n")

        return results_file, report_file

    @classmethod
    def _check_pyenv(cls):
        """Check if pyenv is available."""
        try:
            result = subprocess.run(["pyenv", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @classmethod
    def _version_key(cls, version_string):
        """Convert version string to tuple of integers for proper sorting.

        Examples:
            '3.8.9' -> (3, 8, 9)
            '3.8.20' -> (3, 8, 20)
            '3.10.1' -> (3, 10, 1)
        """
        # Remove any leading/trailing whitespace
        version_string = version_string.strip()

        # Split by dots and convert to integers
        parts = version_string.split(".")
        try:
            return tuple(int(p) for p in parts if p.isdigit())
        except ValueError:
            # If conversion fails, return original string (will be sorted lexicographically)
            return (0,)  # Put invalid versions at the beginning

    @classmethod
    def _get_pyenv_versions(cls):
        """Get list of installed Python versions from pyenv."""
        try:
            result = subprocess.run(["pyenv", "versions", "--bare"], capture_output=True, text=True)
            if result.returncode == 0:
                return [v.strip() for v in result.stdout.strip().split("\n") if v.strip()]
            return []
        except FileNotFoundError:
            return []

    @classmethod
    def _install_python_with_pyenv(cls, version):
        """Install Python version using pyenv."""
        print(f"\nInstalling Python {version} using pyenv...")
        print(f"This may take several minutes...\n")

        result = subprocess.run(
            ["pyenv", "install", "-s", version], stdout=sys.stdout, stderr=sys.stderr
        )

        return result.returncode == 0

    @classmethod
    def _find_pyenv_python_version(cls, target_version):
        """Find matching Python version in pyenv for target version (e.g., '3.8' -> '3.8.18').
        Returns only INSTALLED versions, not available for installation."""
        installed_versions = cls._get_pyenv_versions()

        # Find all versions matching the target, excluding virtual environments
        matching = [
            v
            for v in installed_versions
            if v.startswith(target_version + ".")
            and "/" not in v  # Exclude virtual environments like "3.9.22/envs/.venv"
            and not any(x in v for x in ["-", "a", "b", "rc", "dev"])  # Exclude dev versions
        ]

        if matching:
            # Return the latest patch version using proper version comparison
            return sorted(matching, key=cls._version_key)[-1]

        # Version not installed
        return None

    @classmethod
    def _find_installable_python_version(cls, target_version):
        """Find the latest installable Python version for target version.
        Returns version string that can be installed via pyenv."""
        try:
            result = subprocess.run(["pyenv", "install", "--list"], capture_output=True, text=True)
            if result.returncode == 0:
                available = [
                    v.strip()
                    for v in result.stdout.strip().split("\n")
                    if v.strip().startswith(target_version + ".")
                    and not any(x in v for x in ["-", "a", "b", "rc", "dev", "/"])
                ]
                if available:
                    latest = sorted(available, key=cls._version_key)[-1]
                    return latest
        except Exception as e:
            print(f"Error finding available versions: {e}")

        return None

    @classmethod
    def _get_project_version(cls):
        """Get project version from git branch and latest tag."""
        try:
            # Get current branch
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=cls.project_root,
            ).stdout.strip()

            # Get latest tag
            tag = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=cls.project_root,
            ).stdout.strip()

            if branch and branch != "master" and branch != "main":
                return f"{tag}+{branch}" if tag else f"0.0.0+{branch}"
            return tag if tag else "0.0.0"
        except Exception as e:
            print(f"Warning: Could not determine version from git: {e}")
            return "0.0.0+test"

    @classmethod
    def setUpClass(cls):
        """Create test virtual environments for all Python versions using pyenv."""
        cls.test_results["start_time"] = datetime.now().isoformat()
        cls.test_venv_dirs = {}
        cls.test_pythons = {}
        cls.test_pips = {}
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Get project version to avoid GitHub API calls
        cls.release_version = cls._get_project_version()
        print(f"\nUsing version: {cls.release_version}\n")

        # Determine which versions to test
        versions_to_test = cls.TEST_VERSIONS if cls.TEST_VERSIONS else cls.PYTHON_VERSIONS
        print(f"Requested Python versions: {', '.join(versions_to_test)}")

        print(f"\n{'='*80}")
        print(f"Setting up test environments for Python versions: {', '.join(versions_to_test)}")
        print(f"{'='*80}\n")

        # Check if pyenv is available
        if not cls._check_pyenv():
            print("⚠ pyenv is not installed or not in PATH")
            print("Falling back to system Python versions...\n")
            use_pyenv = False
        else:
            pyenv_version = subprocess.run(["pyenv", "--version"], capture_output=True, text=True)
            print(f"✓ Found {pyenv_version.stdout.strip()}\n")
            use_pyenv = True

        # Find available Python versions
        cls.available_pythons = {}

        if use_pyenv:
            # Use pyenv to find or install Python versions
            for version in versions_to_test:
                print(f"Looking for Python {version} in pyenv...")

                # Check if version is already installed
                full_version = cls._find_pyenv_python_version(version)

                if full_version:
                    print(f"✓ Found Python {full_version} in pyenv")
                    cls.available_pythons[version] = full_version
                    cls.test_results["python_versions"][version] = f"pyenv {full_version}"
                else:
                    # Version not installed, try to find and install it
                    print(f"Python {version} not installed, searching for available versions...")

                    installable_version = cls._find_installable_python_version(version)

                    if installable_version:
                        print(f"Found installable version: {installable_version}")
                        print(
                            f"Installing Python {installable_version} with pyenv (this may take several minutes)..."
                        )

                        install_result = subprocess.run(
                            ["pyenv", "install", "-s", installable_version],
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                        )

                        if install_result.returncode == 0:
                            print(f"✓ Successfully installed Python {installable_version}")
                            cls.available_pythons[version] = installable_version
                            cls.test_results["python_versions"][
                                version
                            ] = f"pyenv {installable_version} (installed)"
                        else:
                            print(f"✗ Failed to install Python {installable_version}")
                    else:
                        print(f"✗ No installable version found for Python {version}")
        else:
            # Fallback to system Python
            for version in versions_to_test:
                python_cmd = f"python{version}"
                try:
                    result = subprocess.run(
                        [python_cmd, "--version"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        cls.available_pythons[version] = python_cmd
                        cls.test_results["python_versions"][
                            version
                        ] = f"system {result.stdout.strip()}"
                        print(f"✓ Found {result.stdout.strip()}")
                    else:
                        print(f"✗ Python {version} not found")
                except FileNotFoundError:
                    print(f"✗ Python {version} not found")

        if not cls.available_pythons:
            raise RuntimeError(
                "No Python versions found! Please install Python 3.8-3.12 using pyenv or system package manager"
            )

        print(f"\n{'='*80}")
        print(f"Creating virtual environments...")
        print(f"{'='*80}\n")

        # Create virtual environments for each available Python version
        for version, python_identifier in cls.available_pythons.items():
            venv_dir = tempfile.mkdtemp(prefix=f"test_venv_py{version}_")

            # SAFETY CHECK: Ensure we're creating venv in /tmp
            if not venv_dir.startswith("/tmp/") and not venv_dir.startswith(tempfile.gettempdir()):
                raise RuntimeError(
                    f"SAFETY ERROR: Virtual environment must be in /tmp, got: {venv_dir}"
                )

            print(f"Creating venv for Python {version} at: {venv_dir}")

            try:
                if use_pyenv:
                    # Use pyenv to get the Python executable path
                    result_prefix = subprocess.run(
                        ["pyenv", "prefix", python_identifier], capture_output=True, text=True
                    )

                    if result_prefix.returncode != 0:
                        print(f"✗ Failed to get pyenv prefix for {python_identifier}")
                        print(f"Error: {result_prefix.stderr}")
                        shutil.rmtree(venv_dir, ignore_errors=True)
                        continue

                    python_path = result_prefix.stdout.strip()

                    # Ensure we have an absolute path
                    if not os.path.isabs(python_path):
                        python_path = os.path.abspath(python_path)

                    python_bin = os.path.join(python_path, "bin", "python")

                    if not os.path.exists(python_bin):
                        print(f"✗ Python executable not found at: {python_bin}")
                        shutil.rmtree(venv_dir, ignore_errors=True)
                        continue

                    print(f"Using Python: {python_bin}")

                    result = subprocess.run(
                        [python_bin, "-m", "venv", venv_dir], stdout=sys.stdout, stderr=sys.stderr
                    )
                else:
                    result = subprocess.run(
                        [python_identifier, "-m", "venv", venv_dir],
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                    )

                if result.returncode != 0:
                    print(f"✗ Failed to create virtual environment for Python {version}")
                    shutil.rmtree(venv_dir, ignore_errors=True)
                    continue

                # Successfully created venv, add to tracking
                cls.test_venv_dirs[version] = venv_dir

            except Exception as e:
                print(f"✗ Exception creating venv for Python {version}: {e}")
                shutil.rmtree(venv_dir, ignore_errors=True)
                continue

            # Get paths for the test virtual environment
            if sys.platform.startswith("win"):
                cls.test_pythons[version] = os.path.join(venv_dir, "Scripts", "python.exe")
                cls.test_pips[version] = os.path.join(venv_dir, "Scripts", "pip.exe")
            else:
                cls.test_pythons[version] = os.path.join(venv_dir, "bin", "python")
                cls.test_pips[version] = os.path.join(venv_dir, "bin", "pip")

            # Upgrade pip
            print(f"Upgrading pip for Python {version}...")
            subprocess.run(
                [cls.test_pips[version], "install", "--upgrade", "pip", "-q"],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            print(f"✓ Environment ready for Python {version}\n")

    @classmethod
    def tearDownClass(cls):
        """Remove all test virtual environments."""
        print(f"\n{'='*80}")
        print(f"TEARDOWN: Cleaning up test virtual environments")
        print(f"Total tests run: {cls.test_results['summary']['total']}")
        print(f"{'='*80}\n")
        for version, venv_dir in cls.test_venv_dirs.items():
            print(f"Removing test environment for Python {version}: {venv_dir}")
            if os.path.exists(venv_dir):
                shutil.rmtree(venv_dir)
            else:
                print(f"  WARNING: venv already removed: {venv_dir}")

        # Save test results
        cls._save_results()

    def _test_install_with_extra(self, extra_name, python_version):
        """Helper method to test installation with a specific extra for a specific Python version."""
        test_name = f"test_install_{extra_name}_py{python_version}"
        test_result = {
            "name": extra_name,
            "python_version": python_version,
            "status": "PASS",
            "error": None,
        }

        try:
            print(f"\n{'='*80}")
            print(f"Testing installation with extra: [{extra_name}] for Python {python_version}")
            print(f"{'='*80}\n")

            # Check if Python version is available
            if python_version not in self.test_pips or python_version not in self.test_pythons:
                raise RuntimeError(
                    f"Python {python_version} environment not available. "
                    f"Available versions: {list(self.test_pips.keys())}"
                )

            test_pip = self.test_pips[python_version]
            test_python = self.test_pythons[python_version]

            # SAFETY CHECK: Ensure venv still exists
            venv_dir = self.test_venv_dirs[python_version]
            if not os.path.exists(venv_dir):
                raise RuntimeError(
                    f"Virtual environment for Python {python_version} does not exist at: {venv_dir}"
                )

            # SAFETY CHECK: Ensure pip and python executables still exist
            if not os.path.exists(test_pip):
                raise RuntimeError(f"pip executable does not exist at: {test_pip}")
            if not os.path.exists(test_python):
                raise RuntimeError(f"python executable does not exist at: {test_python}")

            print(f"✓ Environment validated:")
            print(f"  venv: {venv_dir}")
            print(f"  python: {test_python}")
            print(f"  pip: {test_pip}\n")

            # SAFETY CHECK: Ensure we're using pip from isolated venv
            if not test_pip.startswith(venv_dir):
                raise RuntimeError(
                    f"SAFETY ERROR: pip must be from isolated venv!\n"
                    f"  Expected prefix: {venv_dir}\n"
                    f"  Got: {test_pip}"
                )

            if not test_python.startswith(venv_dir):
                raise RuntimeError(
                    f"SAFETY ERROR: python must be from isolated venv!\n"
                    f"  Expected prefix: {venv_dir}\n"
                    f"  Got: {test_python}"
                )

            # Install the package with the specific extra
            install_cmd = [
                test_pip,
                "install",
                "-e",
                f"{self.project_root}[{extra_name}]",
            ]

            print(f"Running: {' '.join(install_cmd)}\n")

            # Set environment variable to avoid GitHub API calls in setup.py
            env = os.environ.copy()
            env["RELEASE_VERSION"] = self.release_version

            install_result = subprocess.run(
                install_cmd, stdout=sys.stdout, stderr=sys.stderr, text=True, env=env
            )

            # Check if the installation was successful
            self.assertEqual(
                install_result.returncode,
                0,
                msg=f"Installation with [{extra_name}] for Python {python_version} failed with return code {install_result.returncode}",
            )

            # Verify supervisely can be imported
            print(f"\n{'='*80}")
            print(f"Verifying import for [{extra_name}] (Python {python_version})")
            print(f"{'='*80}\n")

            import_cmd = [
                test_python,
                "-c",
                "import supervisely; print(f'Successfully imported supervisely {supervisely.__version__}')",
            ]
            import_result = subprocess.run(
                import_cmd, stdout=sys.stdout, stderr=sys.stderr, text=True
            )

            self.assertEqual(
                import_result.returncode,
                0,
                msg=f"Failed to import supervisely with [{extra_name}] (Python {python_version})",
            )

            # List all installed packages and their versions
            print(f"\n{'='*80}")
            print(f"Installed packages for [{extra_name}] (Python {python_version})")
            print(f"{'='*80}\n")

            list_packages_cmd = [test_pip, "list"]
            subprocess.run(list_packages_cmd, stdout=sys.stdout, stderr=sys.stderr)

            print(f"\n{'='*80}")
            print(
                f"✓ Successfully tested installation with [{extra_name}] for Python {python_version}"
            )
            print(f"{'='*80}\n")

            test_result["status"] = "PASS"
            self.test_results["summary"]["passed"] += 1

        except Exception as e:
            test_result["status"] = "FAIL"
            test_result["error"] = str(e)
            self.test_results["summary"]["failed"] += 1
            raise
        finally:
            self.test_results["tests"].append(test_result)
            self.test_results["summary"]["total"] += 1

    def test_01_base_installation(self):
        """Test base installation without any extras for all Python versions."""
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                test_result = {
                    "name": "base_installation",
                    "python_version": version,
                    "status": "PASS",
                    "error": None,
                }

                try:
                    print(f"\n{'='*80}")
                    print(f"Testing base installation (no extras) for Python {version}")
                    print(f"{'='*80}\n")

                    # Check if Python version is available
                    if version not in self.test_pips or version not in self.test_pythons:
                        raise RuntimeError(
                            f"Python {version} environment not available. "
                            f"Available versions: {list(self.test_pips.keys())}"
                        )

                    test_pip = self.test_pips[version]
                    test_python = self.test_pythons[version]

                    # SAFETY CHECK: Ensure venv still exists
                    venv_dir = self.test_venv_dirs[version]
                    if not os.path.exists(venv_dir):
                        raise RuntimeError(
                            f"Virtual environment for Python {version} does not exist at: {venv_dir}"
                        )

                    # SAFETY CHECK: Ensure pip and python executables still exist
                    if not os.path.exists(test_pip):
                        raise RuntimeError(f"pip executable does not exist at: {test_pip}")
                    if not os.path.exists(test_python):
                        raise RuntimeError(f"python executable does not exist at: {test_python}")

                    print(f"✓ Environment validated:")
                    print(f"  venv: {venv_dir}")
                    print(f"  python: {test_python}")
                    print(f"  pip: {test_pip}\n")

                    # SAFETY CHECK: Ensure we're using pip/python from isolated venv
                    if not test_pip.startswith(venv_dir):
                        raise RuntimeError(
                            f"SAFETY ERROR: pip must be from isolated venv!\n"
                            f"  Expected prefix: {venv_dir}\n"
                            f"  Got: {test_pip}"
                        )

                    if not test_python.startswith(venv_dir):
                        raise RuntimeError(
                            f"SAFETY ERROR: python must be from isolated venv!\n"
                            f"  Expected prefix: {venv_dir}\n"
                            f"  Got: {test_python}"
                        )

                    install_cmd = [
                        test_pip,
                        "install",
                        "--no-cache-dir",
                        "-e",
                        self.project_root,
                    ]

                    print(f"Running: {' '.join(install_cmd)}\n")

                    # Set environment variable to avoid GitHub API calls in setup.py
                    env = os.environ.copy()
                    env["RELEASE_VERSION"] = self.release_version

                    install_result = subprocess.run(
                        install_cmd, stdout=sys.stdout, stderr=sys.stderr, text=True, env=env
                    )

                    self.assertEqual(
                        install_result.returncode,
                        0,
                        msg=f"Base installation failed for Python {version}",
                    )

                    # Check if supervisely is installed
                    print(f"\n{'='*80}")
                    print(f"Checking installed packages for Python {version}")
                    print(f"{'='*80}\n")

                    list_cmd = [test_pip, "list"]
                    list_result = subprocess.run(list_cmd, capture_output=True, text=True)

                    self.assertIn(
                        "supervisely",
                        list_result.stdout,
                        msg=f"supervisely is not in the list of installed packages for Python {version}",
                    )
                    print(f"✓ supervisely found in installed packages for Python {version}")

                    # Check if the supervisely package can be imported
                    print(f"\n{'='*80}")
                    print(f"Verifying import for Python {version}")
                    print(f"{'='*80}\n")

                    import_cmd = [
                        test_python,
                        "-c",
                        "import supervisely; print(f'Successfully imported supervisely {supervisely.__version__}')",
                    ]
                    import_result = subprocess.run(
                        import_cmd, stdout=sys.stdout, stderr=sys.stderr, text=True
                    )

                    self.assertEqual(
                        import_result.returncode,
                        0,
                        msg=f"Failed to import supervisely package for Python {version}",
                    )

                    # List all installed packages and their versions
                    print(f"\n{'='*80}")
                    print(f"Installed packages for Python {version}")
                    print(f"{'='*80}\n")

                    list_packages_cmd = [test_pip, "list"]
                    subprocess.run(list_packages_cmd, stdout=sys.stdout, stderr=sys.stderr)

                    print(f"\n{'='*80}")
                    print(f"✓ Base installation test passed for Python {version}")
                    print(f"{'='*80}\n")

                    test_result["status"] = "PASS"
                    self.test_results["summary"]["passed"] += 1

                except Exception as e:
                    test_result["status"] = "FAIL"
                    test_result["error"] = str(e)
                    self.test_results["summary"]["failed"] += 1
                    raise
                finally:
                    self.test_results["tests"].append(test_result)
                    self.test_results["summary"]["total"] += 1

    def test_02_extras(self):
        """Test installation with 'extras' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 02: EXTRAS")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("extras", version)

    def test_03_apps(self):
        """Test installation with 'apps' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 03: APPS")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("apps", version)

    def test_04_docs(self):
        """Test installation with 'docs' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 04: DOCS")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("docs", version)

    def test_05_sdk_no_usages(self):
        """Test installation with 'sdk-no-usages' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 05: SDK-NO-USAGES")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("sdk-no-usages", version)

    def test_06_tracking(self):
        """Test installation with 'tracking' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 06: TRACKING")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("tracking", version)

    def test_07_model_benchmark(self):
        """Test installation with 'model-benchmark' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 07: MODEL-BENCHMARK")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("model-benchmark", version)

    def test_08_training(self):
        """Test installation with 'training' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 08: TRAINING")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("training", version)

    def test_09_plugins(self):
        """Test installation with 'plugins' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 09: PLUGINS")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("plugins", version)

    def test_10_sdk_nn_plugins(self):
        """Test installation with 'sdk-nn-plugins' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 10: SDK-NN-PLUGINS")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("sdk-nn-plugins", version)

    def test_11_aug(self):
        """Test installation with 'aug' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 11: AUG")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("aug", version)

    def test_12_agent(self):
        """Test installation with 'agent' option for all Python versions."""
        print(f"\n{'#'*80}")
        print(f"# TEST 12: AGENT")
        print(f"{'#'*80}\n")
        for version in self.available_pythons.keys():
            with self.subTest(python_version=version):
                self._test_install_with_extra("agent", version)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test Supervisely SDK installation with different extras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for all Python versions
  python3 tests/test_setup.py
  
  # Run tests only for Python 3.11 and 3.12
  python3 tests/test_setup.py --python-versions 3.11 3.12
  
  # Run specific test for specific Python versions
  python3 tests/test_setup.py --python-versions 3.11 3.12 SetupTests.test_06_tracking
  
  # Run with verbose output
  python3 tests/test_setup.py --python-versions 3.11 -v
        """,
    )
    parser.add_argument(
        "--python-versions",
        nargs="+",
        metavar="VERSION",
        help="Python versions to test (e.g., 3.8 3.9 3.10 3.11 3.12)",
    )

    # Parse known args to allow unittest args to pass through
    args, remaining = parser.parse_known_args()

    # Set the test versions if specified
    if args.python_versions:
        # Separate Python versions from test names
        # Python versions are in format X.Y or X.YY
        python_versions = []
        test_names = []

        for arg in args.python_versions:
            # Check if it looks like a Python version (e.g., "3.8", "3.11")
            if arg.replace(".", "").isdigit() and arg.count(".") == 1:
                python_versions.append(arg)
            else:
                # Treat as test name
                test_names.append(arg)

        # Validate Python versions
        if python_versions:
            invalid_versions = [v for v in python_versions if v not in SetupTests.PYTHON_VERSIONS]
            if invalid_versions:
                print(f"Error: Invalid Python version(s): {', '.join(invalid_versions)}")
                print(f"Valid versions are: {', '.join(SetupTests.PYTHON_VERSIONS)}")
                sys.exit(1)

            SetupTests.TEST_VERSIONS = python_versions
            print(f"Testing only Python versions: {', '.join(python_versions)}\n")

        # Add test names to remaining args for unittest
        if test_names:
            remaining = test_names + remaining
    else:
        print(f"Testing all Python versions: {', '.join(SetupTests.PYTHON_VERSIONS)}\n")

    # Run unittest with remaining arguments
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
