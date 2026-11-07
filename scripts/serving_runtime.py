"""Prepare the isolated, commit-pinned vLLM SPS runtime through MLSpace."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
from pathlib import Path

from scripts import mlspace_launch as launch
from scripts import vllm_sps_config as c

ENV_ROOT = Path("/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/envs")
RUNTIME = ENV_ROOT / "vllm_sps_main_py312_cu130"
ARTIFACTS = c.ROOT / "runtime"
UV = ENV_ROOT / ".tools/uv-0.9.11"
UV_BOOTSTRAP = Path("/home/jovyan/.local/bin/uv")
WHEEL = (
    f"https://wheels.vllm.ai/{c.VLLM_BASE}/"
    "vllm-0.33.0rc1.dev930%2Bg3fe0d4421-cp38-abi3-manylinux_2_28_x86_64.whl"
)
OVERLAY = (
    "vllm/model_executor/models/qwen3_sps.py",
    "vllm/model_executor/models/registry.py",
    "vllm/v1/attention/ops/sps_prediction.py",
    "vllm/v1/spec_decode/sps_proposer.py",
)
experiment_configs = [{"name": "vllm-sps-main-py312-cu130", "num_gpus": 1, "hours": 2}]


def identity():
    return {
        "runtime": str(RUNTIME),
        "python": "3.12.12",
        "wheel": WHEEL,
        "upstream": c.VLLM_BASE,
        "implementation": c.VLLM_COMMIT,
        "overlay": list(OVERLAY),
        "extra_packages": ["datasets==5.1.0"],
    }


def signature():
    return hashlib.sha256(json.dumps(identity(), sort_keys=True).encode()).hexdigest()


def run(command, **kwargs):
    print(shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def replace_overlay(source, target):
    """Replace the inode: an installed file might share uv's cached wheel."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(source.read_bytes())
        temporary.chmod(0o644)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_snapshot():
    """Independent immutable checkout, without copying repository credentials."""
    target = ARTIFACTS / "source" / c.VLLM_COMMIT
    if target.exists():
        if (target / "source-commit.txt").read_text().strip() != c.VLLM_COMMIT:
            raise RuntimeError("Snapshot identity mismatch")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive_path = Path(tmp) / "source.tar"
        run(
            [
                "git",
                "-C",
                c.VLLM_WORKTREE,
                "archive",
                "--output",
                archive_path,
                c.VLLM_COMMIT,
            ]
        )
        checkout = Path(tmp) / "checkout"
        checkout.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(checkout)
        (checkout / "source-commit.txt").write_text(c.VLLM_COMMIT + "\n")
        checkout.rename(target)
    return target


def validate_overlay():
    changed = set(
        launch.git(
            "-C",
            str(c.VLLM_WORKTREE),
            "diff",
            "--name-only",
            c.VLLM_BASE,
            c.VLLM_COMMIT,
        ).splitlines()
    )
    runtime_changes = {
        p for p in changed if not p.startswith(("docs/", "tests/")) and p != "AGENTS.md"
    }
    if runtime_changes != set(OVERLAY):
        raise RuntimeError(f"Unexpected compiled/runtime changes: {runtime_changes}")


def worker():
    owner = RUNTIME / ".sps-runtime-owner.json"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if owner.exists():
        if json.loads(owner.read_text()) != identity():
            raise RuntimeError("Refusing to modify an unrelated runtime")
    elif any(RUNTIME.iterdir()):
        raise RuntimeError("Refusing to modify a pre-existing unowned environment")
    else:
        launch.store_json(owner, identity())
    env = dict(
        os.environ,
        UV_CACHE_DIR=str(ENV_ROOT / ".uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(ENV_ROOT / ".python"),
        UV_HTTP_TIMEOUT="120",
    )
    # The orchestration environment supplies Python3.10 headers; the runtime
    # interpreter comes with its own matching headers for Triton compilation.
    env.pop("C_INCLUDE_PATH", None)
    env.pop("CPLUS_INCLUDE_PATH", None)
    if not (RUNTIME / "pyvenv.cfg").exists():
        run(
            [UV, "venv", "--python", identity()["python"], "--allow-existing", RUNTIME],
            env=env,
        )
    python = RUNTIME / "bin/python"
    run(
        [
            UV,
            "pip",
            "install",
            "--python",
            python,
            "--link-mode",
            "copy",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
            WHEEL,
            *identity()["extra_packages"],
        ],
        env=env,
    )
    # Restore upstream wheel files before applying the reviewed Python delta.
    # Only these files differ; the compiled ABI is exactly the upstream wheel's.
    site = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                'import sysconfig; print(sysconfig.get_paths()["purelib"])',
            ],
            text=True,
        ).strip()
    )
    if not site.is_relative_to(RUNTIME):
        raise RuntimeError("Installer selected an unrelated environment")
    snapshot = ARTIFACTS / "source" / c.VLLM_COMMIT
    hashes = {}
    for name in OVERLAY:
        source = snapshot / name
        target = site / name
        replace_overlay(source, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    smoke = """
import json, torch, vllm
import vllm._C
from vllm.model_executor.models.qwen3_sps import Qwen3SPSForCausalLM
from vllm.v1.spec_decode.sps_proposer import SPSProposer
from vllm.v1.attention.ops.sps_prediction import sps_prediction_attention
assert torch.__version__.split('+')[0] == '2.14.0', torch.__version__
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
x = torch.randn(4, 16, device='cuda', dtype=torch.bfloat16)
out = torch.empty(4, 8, device='cuda', dtype=torch.bfloat16)
torch.ops._C.silu_and_mul(out, x)
torch.testing.assert_close(out, torch.nn.functional.silu(x[:, :8]) * x[:, 8:], atol=.02, rtol=.02)
torch.cuda.synchronize()
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                     vllm=vllm.__version__, imported_from=vllm.__file__,
                     gpu=torch.cuda.get_device_name())))
"""
    checked = subprocess.check_output([str(python), "-c", smoke], env=env, text=True)
    print(checked, flush=True)
    (ARTIFACTS / "smoke.log").write_text(checked)
    run([UV, "pip", "check", "--python", python], env=env)
    frozen = subprocess.check_output(
        [str(UV), "pip", "freeze", "--python", str(python)], env=env, text=True
    )
    (ARTIFACTS / "requirements.lock.txt").write_text(frozen)
    manifest = dict(
        identity(),
        design_signature=signature(),
        overlay_sha256=hashes,
        site_packages=str(site),
        smoke=checked,
        packages=frozen,
    )
    launch.store_json(RUNTIME / "sps-runtime.json", manifest)
    launch.store_json(ARTIFACTS / "prepared.json", manifest)


def payload(staged):
    exp = experiment_configs[0]
    body = launch.payload("prepare", exp["name"], staged, "d.tarasov", "mrsndmn")
    body.update(
        instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[exp["num_gpus"]],
        job_desc=f"sps-serving-runtime prepare {exp['name']} #d.tarasov @mrsndmn",
    )
    body["script"] = f"cd {shlex.quote(str(staged))} && " + shlex.join(
        [str(launch.PYTHON), "-m", "scripts.serving_runtime", "--worker"]
    )
    return body


def submit(dry):
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    if dry:
        print(json.dumps(payload(staged), indent=2))
        return
    validate_overlay()
    from mls.manager.job.utils import training_job_api_from_profile

    client, options = training_job_api_from_profile("default")
    launch.disable_submission_retries(client)
    active = {}
    # Reuse the authenticated client, avoiding repeated service_auth requests.
    for status in ("Pending", "Running"):
        response = client.get_list_jobs(options["region"], None, status, 1000, 0)
        if "jobs" not in response:
            raise RuntimeError("Cannot establish active jobs")
        for job in response["jobs"]:
            active[launch.normalize(job["job_desc"])] = job["job_name"]
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    marker = ARTIFACTS / "prepared.json"
    if marker.exists():
        if json.loads(marker.read_text())["design_signature"] != signature():
            raise RuntimeError("Runtime configuration changed")
        print("Runtime already prepared:", RUNTIME)
        return
    body = payload(staged)
    job_identity = launch.normalize(body["job_desc"])
    if job_identity in active:
        print(json.dumps({"job_name": active[job_identity], "reused": True}))
        return
    staged = launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
    source_snapshot()
    UV.parent.mkdir(parents=True, exist_ok=True)
    if not UV.exists():
        shutil.copyfile(UV_BOOTSTRAP, UV)
        UV.chmod(0o755)
    body = payload(staged)
    body["region"] = options["region"]
    job = launch.submit_payload(
        client, body, "prepare", experiment_configs[0]["name"], commit
    )
    print(json.dumps(job), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--internal-worker", action="store_true")
    args = parser.parse_args()
    if args.internal_worker:
        return worker()
    if args.worker:
        process = subprocess.Popen(
            [str(launch.PYTHON), "-m", "scripts.serving_runtime", "--internal-worker"],
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=experiment_configs[0]["hours"] * 3600)
            if code:
                raise RuntimeError(f"Runtime worker exited {code}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        return
    launch.ARTIFACTS_ROOT = ARTIFACTS
    launch.RUN_ROOT = ARTIFACTS / "runs"
    if args.dry:
        return submit(True)
    with launch.submission_lock():
        return submit(False)


if __name__ == "__main__":
    main()
