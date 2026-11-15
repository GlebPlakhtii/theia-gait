#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Pinned upstream code and checkpoint; uses a workspace-local environment.
revision=7e5bae88456ac418ff0e58e74106c9fe192055d4
checkpoint=models/sapiens2-5b/sapiens2_5b_pose.safetensors
if [[ ! -d vendor/sapiens2/.git ]]; then
    git clone https://github.com/facebookresearch/sapiens2.git vendor/sapiens2
    git -C vendor/sapiens2 checkout "$revision"
fi
[[ "$(git -C vendor/sapiens2 rev-parse HEAD)" == "$revision" ]]
if [[ ! -d .venv-pose ]]; then uv venv .venv-pose --python 3.13; fi
UV_HTTP_TIMEOUT=300 uv pip install --python .venv-pose/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
UV_HTTP_TIMEOUT=300 uv pip install --python .venv-pose/bin/python -e vendor/sapiens2 av
python3 scripts/download_pose_checkpoint.py
.venv-pose/bin/python - <<'PY'
import hashlib
from pathlib import Path
p=Path('models/sapiens2-5b/sapiens2_5b_pose.safetensors')
target=p if p.exists() else p.with_suffix('.safetensors.part')
assert target.stat().st_size==20480899148
assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()=='b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52'
if target!=p:target.rename(p)
import torch
assert torch.cuda.is_available()
print(torch.__version__,torch.cuda.get_device_name())
PY
