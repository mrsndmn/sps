"""Submit the native-readout arm of the paired Qwen MTP experiment."""
import argparse
import json
from pathlib import Path
import shlex

from orchestration import mlspace_launch as launch
from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c
from experiments.real_readout_run import family_config as real_readout_run_settings
from experiments.real_readout_run import experiment_configs
ARM, = experiment_configs
ROOT = real_readout_run_settings.ROOT
PARENT = real_readout_run_settings.PARENT
OUT = real_readout_run_settings.OUT
LEGACY_HORIZONS = real_readout_run_settings.LEGACY_HORIZONS


def trained():
    marker = json.loads((OUT / "trained.json").read_text())
    if (marker["checkpoint"] != str(OUT / "final.pt") or marker["parent"] != str(PARENT)
            or marker["step"] != c.family_config.TRAIN_STEPS or not marker["checkpoint_sha256"]
            or marker.get("horizons", LEGACY_HORIZONS) != c.family_config.HORIZONS
            or common.recorded_schedule(marker) != common.training_schedule()):
        raise ValueError("Native readout checkpoint does not match the configured run")
    return marker


def probe_report():
    path = ROOT / "probe/prepared.json"
    return path


def check_probe():
    projected = common.probe_seconds(probe_report(), num_gpus=ARM.num_gpus,
                                     legacy_horizons=LEGACY_HORIZONS)
    if projected >= c.family_config.TRAIN_HOURS * 3600:
        raise RuntimeError("Training projection exceeds allocation")


def submit(phase, dry):
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    output = ROOT / "probe" if phase == "probe" else OUT
    if not dry and not launch.pending(output):
        return
    script = ["timeout", f"{c.family_config.PROBE_HOURS if phase == 'probe' else c.family_config.TRAIN_HOURS}h",
              str(launch.PYTHON), "-m", "torch.distributed.run", "--standalone",
              f"--nproc_per_node={ARM.num_gpus}", "scripts/readout/real_readout_train.py", phase]
    body = launch.payload("train", ARM.name, staged, "d.tarasov", "mrsndmn")
    body.update(script=f"cd {shlex.quote(str(staged))} && " + shlex.join(script),
                instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[ARM.num_gpus],
                job_desc=f"sps-real-readout {phase} {ARM.name} #d.tarasov @mrsndmn")
    body["env_variables"].update(SPS_EXPERIMENT_SUITE="ar-sps-qwen3-1.7b-instruct",
                                 HF_HUB_DISABLE_XET="1")
    if dry:
        launch.print_preview(body, output, staged, phase)
        return
    if phase == "train":
        check_probe()
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, extra = training_job_api_from_profile("default")
    launch.disable_submission_retries(client)
    active = {launch.normalize(j["job_desc"]): j["job_name"]
              for j in get_in_progress_jobs(client_profile="default")}
    launch.reconcile_for_outputs(client, active, [output])
    identity = launch.output_identity(body["job_desc"], output)
    if identity in active:
        print(active[identity])
        return
    launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
    body["region"] = extra["region"]
    print(json.dumps(launch.submit_payload(client, body, "prepare" if phase == "probe" else phase,
                                          ARM.name, commit, output_dir=output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "train"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    launch.ARTIFACTS_ROOT, launch.RUN_ROOT = ROOT, ROOT / "runs"
    if args.dry:
        return submit(args.phase, True)
    with launch.submission_lock():
        submit(args.phase, False)
