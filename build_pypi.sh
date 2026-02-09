#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*

cat <<'MSG'
Build complete.

Upload to TestPyPI:
  python -m twine upload --repository testpypi dist/*

Upload to PyPI:
  python -m twine upload dist/*
MSG
