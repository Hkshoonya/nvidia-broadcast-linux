#!/usr/bin/env bash
# Run only inside the non-release builder with networking disabled.
set -euo pipefail
umask 022

NVB_PYTHON=/opt/nvbroadcast/.venv/bin/python
export PATH="/opt/nvbroadcast/.venv/bin:$PATH"
: "${SOURCE_DATE_EPOCH:?set from inputs.json application.source_date_epoch}"
export SOURCE_DATE_EPOCH
export CFLAGS='-O2 -g0'

"$NVB_PYTHON" -I /prototype/prepare.py fetch --offline --directory /work
"$NVB_PYTHON" -I -m pip --isolated install --no-index --no-cache-dir \
    --no-deps --require-hashes --find-links=/work/bootstrap/wheels \
    -r /work/bootstrap/requirements.txt

mkdir -p /work/wheels
"$NVB_PYTHON" -I -m pip --isolated wheel --no-index --no-cache-dir \
    --no-deps --no-build-isolation --wheel-dir=/work/wheels \
    /work/inputs/pycairo-1.29.0.tar.gz
"$NVB_PYTHON" -I -m pip --isolated install --no-index --no-cache-dir --no-deps \
    /work/wheels/pycairo-1.29.0-cp313-cp313-linux_x86_64.whl
"$NVB_PYTHON" -I -m pip --isolated wheel --no-index --no-cache-dir \
    --no-deps --no-build-isolation --wheel-dir=/work/wheels \
    /work/inputs/pygobject-3.48.2.tar.gz

# setuptools writes build metadata beside the source. Use a disposable copy.
cp -a /work/app-source /tmp/nvbroadcast-app
"$NVB_PYTHON" -I -m pip --isolated wheel --no-index --no-cache-dir \
    --no-deps --no-build-isolation --wheel-dir=/work/wheels /tmp/nvbroadcast-app
sha256sum /work/wheels/*.whl > /work/built-wheel-sha256.txt
dpkg-query -W > /work/binding-builder-packages.txt
