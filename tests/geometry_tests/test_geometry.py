import os

import pytest


def run_geometry_test():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytest.main([test_dir])


if __name__ == "__main__":
    run_geometry_test()
