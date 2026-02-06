#!/bin/bash

# Automated migration script to Ubuntu 24.04 + Python 3.12
# migrate_to_ubuntu24_py312.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function for logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Docker check
check_docker() {
    log "Checking Docker..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed!"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running or no access permissions!"
    fi
    log "✅ Docker ready"
}

# Check NVIDIA runtime availability (optional)
check_nvidia() {
    log "Checking NVIDIA Docker runtime..."
    if docker info 2>/dev/null | grep -q nvidia; then
        log "✅ NVIDIA Docker runtime available"
        return 0
    else
        warn "NVIDIA Docker runtime unavailable. GPU functions may not work."
        return 1
    fi
}

# Get current SDK version
get_sdk_version() {
    log "Determining current Supervisely SDK version..."
    
    # Try to get version from different sources
    if [ -f "supervisely/__init__.py" ]; then
        SDK_VERSION=$(grep -oP "__version__ = ['\"]([^'\"]*)['\"]" supervisely/__init__.py | grep -oP "[0-9]+\.[0-9]+\.[0-9]+" || echo "")
    fi
    
    if [ -z "$SDK_VERSION" ]; then
        SDK_VERSION="6.73.137"  # Fallback version
        warn "Could not automatically determine SDK version. Using: $SDK_VERSION"
    fi
    
    log "SDK Version: $SDK_VERSION"
}

# Build base image
build_base_image() {
    log "🔨 Building base image base-py..."
    
    local image_name="supervisely/base-py:7.0.0-ubuntu24.04-py3.12"
    local dockerfile="base_images/py/Dockerfile.ubuntu24.04-py3.12"
    
    if [ ! -f "$dockerfile" ]; then
        error "Dockerfile not found: $dockerfile"
    fi
    
    docker build -f "$dockerfile" -t "$image_name" base_images/py/ || error "Error building base-py image"
    log "✅ Base image built: $image_name"
}

# Build SDK image
build_sdk_image() {
    log "🔨 Building SDK image base-py-sdk..."
    
    local image_name="supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12"  
    local dockerfile="base_images/py_sdk/Dockerfile.ubuntu24.04-py3.12"
    
    if [ ! -f "$dockerfile" ]; then
        error "Dockerfile not found: $dockerfile"
    fi
    
    docker build -f "$dockerfile" --build-arg tag_ref_name="$SDK_VERSION" -t "$image_name" base_images/py_sdk/ || error "Error building SDK image"
    log "✅ SDK image built: $image_name"
}

# Run compatibility tests
run_compatibility_tests() {
    log "🧪 Running compatibility tests..."
    
    local test_image="supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12"
    local test_script="/scripts/test_python312_compatibility.sh"
    
    # Check for test script existence
    if [ ! -f "base_images/scripts/test_python312_compatibility.sh" ]; then
        error "Test script not found!"
    fi
    
    # Run tests in container
    log "Starting container for testing..."
    docker run --rm \
        -v "$(pwd)/base_images/scripts:/scripts" \
        -v "$(pwd)/supervisely:/workspace/supervisely" \
        --workdir /workspace \
        "$test_image" \
        "$test_script" || error "Compatibility tests failed!"
        
    log "✅ Compatibility tests passed successfully!"
}

# Test GPU support (if available)
test_gpu_support() {
    if check_nvidia; then
        log "🎮 Testing GPU support..."
        
        local test_image="supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12"
        
        docker run --rm --gpus all "$test_image" \
            python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch not installed (normal for base image)')
" || warn "GPU tests failed"
        
        log "✅ GPU testing completed"
    fi
}

# Create migration report
create_migration_report() {
    log "📋 Creating migration report..."
    
    local report_file="migration_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# MIGRATION REPORT TO UBUNTU 24.04 + PYTHON 3.12
========================================

**Date:** $(date)  
**SDK version:** $SDK_VERSION
**Status:** Success ✅

## Created images:
- supervisely/base-py:7.0.0-ubuntu24.04-py3.12
- supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12

## Image sizes:
$(docker images | grep "ubuntu24.04-py3.12")

## Component versions:
\`\`\`
$(docker run --rm supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12 python -c "
import sys, numpy, cv2, pandas, matplotlib, supervisely
print(f'Python: {sys.version.split()[0]}')
print(f'NumPy: {numpy.__version__}')  
print(f'OpenCV: {cv2.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Supervisely: {supervisely.__version__}')
")
\`\`\`

## Next steps:
1. Test images in dev environment
2. Update dependent images (pytorch, tensorflow, etc.)
3. Update CI/CD pipelines
4. Plan production rollout

## Usage commands:
\`\`\`bash
# Using new base image
docker run -it supervisely/base-py:7.0.0-ubuntu24.04-py3.12 bash

# Using SDK image  
docker run -it supervisely/base-py-sdk:7.0.0-ubuntu24.04-py3.12 bash
\`\`\`
EOF

    log "✅ Report created: $report_file"
}

# Main function
main() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "🚀 MIGRATION TO UBUNTU 24.04 + PYTHON 3.12"
    echo "========================================"
    echo -e "${NC}"
    
    check_docker
    get_sdk_version
    
    echo ""
    read -p "Continue with image building? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Migration cancelled by user"
        exit 0
    fi
    
    build_base_image
    build_sdk_image
    
    echo ""
    read -p "Run compatibility tests? (y/N): " -n 1 -r  
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_compatibility_tests
        test_gpu_support
    fi
    
    create_migration_report
    
    echo ""
    echo -e "${GREEN}🎉 MIGRATION COMPLETED SUCCESSFULLY!${NC}"
    echo "Images are ready for testing in dev environment"
}

# Signal handling
trap 'error "Migration interrupted"' INT TERM

# Check that script is run from project root
if [ ! -f "../../setup.py" ] || [ ! -d "../../supervisely" ]; then
    error "Run the script from the supervisely project root directory or ensure the project structure is correct"
fi

# Change to project root directory
cd "$(dirname "$0")/../.."

main "$@"
