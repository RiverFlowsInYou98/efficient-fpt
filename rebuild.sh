#!/bin/bash
set -euo pipefail

echo "🧼 Cleaning previous builds..."
rm -rf build/ dist/
find src/efpt -name "*.c" -delete
find src/efpt -name "*.so" -delete
find src/efpt -name "*.pyd" -delete

echo "🔧 Building Cython extensions..."
python setup.py build_ext --inplace

echo "📦 Installing in editable mode..."
python -m pip install -e .

echo "✅ Done."