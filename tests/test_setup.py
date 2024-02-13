import os
import subprocess
import sys
import unittest


class SetupTests(unittest.TestCase):
    def test_installation(self):
        # Create a new virtual environment
        subprocess.run([sys.executable, "-m", "venv", "test_venv"])

        # Activate the virtual environment
        if sys.platform.startswith("win"):
            activate_script = os.path.join("test_venv", "Scripts", "activate")
        else:
            activate_script = os.path.join("test_venv", "bin", "activate")

        # Install the package in the virtual environment
        install_cmd = f"bash -c 'source {activate_script} && python -m pip install ../supervisely[extras,sdk-nn-plugins,aug]'"
        install_result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)

        # Check if the installation was successful
        self.assertEqual(
            install_result.returncode, 0, msg=f"Installation failed: {install_result.stderr}"
        )

        # Check if supervisely is in the list of installed packages
        list_cmd = f"bash -c 'source {activate_script} && pip list'"
        list_result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True)
        self.assertIn(
            "supervisely",
            list_result.stdout,
            msg=f"supervisely is not in the list of installed packages: {list_result.stdout}",
        )

        # Check if the supervisely package can be imported
        import_cmd = f"bash -c 'source {activate_script} && python -c \"import supervisely\"'"
        import_result = subprocess.run(import_cmd, shell=True, capture_output=True, text=True)
        self.assertEqual(
            import_result.returncode,
            0,
            msg=f"supervisely package can not be imported: {import_result.stderr}",
        )

        # Deactivate the virtual environment
        deactivate_cmd = f"bash -c 'source {activate_script} && deactivate'"
        subprocess.run(deactivate_cmd, shell=True)

        # Remove the virtual environment
        subprocess.run(["rm", "-rf", "test_venv"])


if __name__ == "__main__":
    unittest.main()
