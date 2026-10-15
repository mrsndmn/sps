"""Extract matching Ubuntu CPython headers without replacing the interpreter."""
import hashlib
from pathlib import Path
import subprocess

from experiments import ARTIFACTS_ROOT

HEADER_ROOT = ARTIFACTS_ROOT / 'toolchain' / 'python310'
HEADER_URL = ('https://archive.ubuntu.com/ubuntu/pool/main/p/python3.10/'
              'libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb')
HEADER_SHA256 = 'c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9'


def prepare():
    import requests
    if (HEADER_ROOT / 'usr/include/python3.10/Python.h').is_file():
        return
    HEADER_ROOT.mkdir(parents=True, exist_ok=True)
    response = requests.get(HEADER_URL, timeout=90)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != HEADER_SHA256:
        raise RuntimeError('Python headers checksum mismatch')
    archive = HEADER_ROOT / 'headers.deb'
    archive.write_bytes(response.content)
    subprocess.run(['dpkg-deb', '-x', str(archive), str(HEADER_ROOT)], check=True)


if __name__ == '__main__':
    prepare()
