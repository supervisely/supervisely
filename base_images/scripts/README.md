# Base Images Scripts

This directory contains scripts related to building and testing Docker base images for the Supervisely platform.

## Scripts

### migrate_to_ubuntu24_py312.sh
Automated migration script for upgrading to Ubuntu 24.04 + Python 3.12.

**Usage:**
```bash
# Run from the project root directory
./base_images/scripts/migrate_to_ubuntu24_py312.sh
```

**Features:**
- Builds new base images with Ubuntu 24.04 and Python 3.12
- Runs compatibility tests
- Tests GPU support (if available)
- Generates migration report

### test_python312_compatibility.sh
Compatibility testing script for Python 3.12 migration.

**Usage:**
```bash
# Usually called by the migration script, but can be run manually in a container:
docker run --rm -v $(pwd)/base_images/scripts:/scripts supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12 /scripts/test_python312_compatibility.sh
```

**Features:**
- Tests core Python libraries compatibility
- Tests Supervisely SDK functionality
- Provides system information
- Reports any compatibility issues

## Docker Compose

The `docker-compose.test-py312.yml` file (located in the parent directory) provides an easy way to test the migration:

```bash
# Run from base_images directory
cd base_images
docker-compose -f docker-compose.test-py312.yml up
```

## Directory Structure

```
base_images/
├── docker-compose.test-py312.yml        # Docker Compose for testing
├── scripts/
│   ├── migrate_to_ubuntu24_py312.sh    # Main migration script
│   ├── test_python312_compatibility.sh  # Compatibility testing
│   └── README.md                        # This file
├── py/
│   └── Dockerfile.ubuntu24.04-py3.12   # Base Python image
└── py_sdk/
    └── Dockerfile.ubuntu24.04-py3.12   # SDK image
```

## Notes

- Scripts should be run from the project root directory
- Docker must be installed and running
- NVIDIA Docker runtime is optional but recommended for GPU testing
