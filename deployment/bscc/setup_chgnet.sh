#!/bin/bash
# Run with a Python >=3.10 interpreter; never modifies the parent environment.
set -euo pipefail
NAGEN_BASE_PYTHON=${NAGEN_BASE_PYTHON:-/data02/home/scv7eyx/.conda/envs/flowdesign-uq/bin/python}
NAGEN_ENV=${NAGEN_ENV:-/data02/home/scv7eyx/.conda/envs/nagen-chgnet}
"$NAGEN_BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
if test ! -x "$NAGEN_ENV/bin/python"; then
  "$NAGEN_BASE_PYTHON" -m venv --system-site-packages "$NAGEN_ENV"
fi
"$NAGEN_ENV/bin/python" -m pip install \
  'chgnet==0.4.2' 'torch==2.4.1' 'pymatgen==2025.10.7' \
  'spglib==2.5.0' 'numpy>=1.26,<2' --only-binary=numpy,spglib,pymatgen
"$NAGEN_ENV/bin/python" -c 'import torch,chgnet,spglib; print("torch",torch.__version__,"CHGNet",chgnet.__version__,"spglib",spglib.__version__)'
