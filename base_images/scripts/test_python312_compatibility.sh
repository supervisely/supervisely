#!/bin/bash

# Script for testing compatibility after migration to Python 3.12
# test_python312_compatibility.sh

set -e

echo "🔍 TESTING COMPATIBILITY WITH PYTHON 3.12"
echo "=============================================="

# Check Python version
echo "📍 Checking Python version:"
python --version
echo ""

# Test importing core libraries
echo "📚 Testing import of core libraries:"

test_library() {
    local lib_name=$1
    local import_name=${2:-$1}
    
    echo -n "   $lib_name: "
    if python -c "import $import_name; print(f'✅ {$import_name.__version__}' if hasattr($import_name, '__version__') else '✅ OK')" 2>/dev/null; then
        echo ""
    else
        echo "❌ ERROR"
        return 1
    fi
}

# List of libraries to test
test_library "NumPy" "numpy"
test_library "OpenCV" "cv2" 
test_library "SciPy" "scipy"
test_library "Pandas" "pandas"
test_library "Matplotlib" "matplotlib"
test_library "Pillow" "PIL"
test_library "Requests" "requests"
test_library "NetworkX" "networkx"
test_library "JsonSchema" "jsonschema"
test_library "Shapely" "shapely"
test_library "scikit-image" "skimage"
test_library "Grpcio" "grpc"
test_library "Protobuf" "google.protobuf"

echo ""
echo "🧪 TESTING BASIC OPERATIONS:"

# Test NumPy operations
echo "   NumPy operations:"
python -c "
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)
print(f'   ✅ NumPy mean test: {result}')
"

# Test OpenCV operations  
echo "   OpenCV operations:"
python -c "
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'   ✅ OpenCV conversion test: {gray.shape}')
"

# Test Pandas operations
echo "   Pandas operations:"
python -c "
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = df.sum().sum()
print(f'   ✅ Pandas operations test: {result}')
"

# Test Matplotlib (without display)
echo "   Matplotlib operations:"
python -c "
import matplotlib
matplotlib.use('Agg')  # No GUI
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure()
plt.plot(x, y)
print('   ✅ Matplotlib plot test: OK')
"

echo ""
echo "🎯 TESTING SUPERVISELY SDK:"

# Test Supervisely import
python -c "
try:
    import supervisely as sly
    print(f'   ✅ Supervisely SDK version: {sly.__version__}')
except Exception as e:
    print(f'   ❌ Supervisely SDK error: {e}')
"

# Test core SDK components
python -c "
try:
    import supervisely as sly
    # Test creating simple objects
    project_meta = sly.ProjectMeta()
    print('   ✅ ProjectMeta creation: OK')
    
    # Test geometry
    rect = sly.Rectangle(0, 0, 100, 100)
    print('   ✅ Rectangle geometry: OK')
    
    # Test annotations (basic)
    ann = sly.Annotation((1000, 1000))
    print('   ✅ Annotation creation: OK')
    
except Exception as e:
    print(f'   ❌ Supervisely components error: {e}')
"

echo ""
echo "🔧 SYSTEM INFORMATION:"
echo "   Ubuntu version: $(lsb_release -d | cut -f2)"
echo "   CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'CUDA not found')"
echo "   Available memory: $(free -h | grep Mem | awk '{print $2}')"

echo ""
echo "✅ TESTING COMPLETED"
echo "===================="
