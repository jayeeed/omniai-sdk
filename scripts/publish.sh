#!/usr/bin/env bash
# ==============================================================================
# OmniAI SDK: PyPI Build & Publish Script
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "  🚀 OmniAI Python SDK (omniai) PyPI Packaging Pipeline"
echo "=========================================================="

# 1. Clean previous build artifacts
echo "🧹 Cleaning previous build artifacts..."
rm -rf build dist *.egg-info

# 2. Check build tools
echo "📦 Verifying packaging dependencies (build, twine)..."
python3 -m pip install --quiet --upgrade build twine

# 3. Build Source Distribution and Binary Wheel
echo "🔨 Building sdist and bdist_wheel..."
python3 -m build

# 4. Validate distributions with twine check
echo "🔍 Validating built distributions with twine check..."
python3 -m twine check dist/*

echo ""
echo "✅ Build and validation succeeded!"
echo "Artifacts generated in dist/:"
ls -lh dist/

echo ""
echo "To publish to PyPI:"
echo "  1) TestPyPI: python3 -m twine upload --repository testpypi dist/*"
echo "  2) Production PyPI: python3 -m twine upload dist/*"
echo "  (Username: __token__, Password: your-pypi-api-token)"
echo ""
