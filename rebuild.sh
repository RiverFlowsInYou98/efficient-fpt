#!/bin/bash

echo "🧼 Cleaning previous builds..."
rm -rf build/ dist/
find src/efficient_fpt -name "*.c" -delete
find src/efficient_fpt -name "*.so" -delete
find src/efficient_fpt -name "*.pyd" -delete  # For Windows support

echo "🔧 Building Cython extensions..."
python setup.py build_ext --inplace

echo "📦 Installing in editable mode..."
pip install -e .

echo "✅ Done."