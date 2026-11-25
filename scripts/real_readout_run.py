"""Submit the native-readout arm of the paired Qwen MTP experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

from scripts import mlspace_launch as launch
from scripts import vllm_sps_common as common
from scripts import vllm_sps_config as c

ROOT = c.ROOT.parent / "qwen3-instruct-real-readout-mtp"
PARENT = c.PARENT.parents[1] / "qwen3_instruct_continued_ar_ce" / "final.pt"
experiment_configs = [c.Experiment("qwen3_real_readout_mtp", "train",
                                    num_gpus=c.experiment_configs[0].num_gpus)]
ARM = experiment_configs[0]
OUT = ROOT / "runs/out" / ARM.name


def signature():
    files = ("real_readout_model.py", "real_readout_train.py", "real_readout_run.py")
    hashes = [common.file_hash(Path(__file__).with_name(name)) for name in files]
    return hashlib.sha256(json.dumps([*hashes, common.signature()]).encode()).hexdigest()


def submit(phase, dry):
    commit = launch.git("rev-parse", "HEAD")
    staged = launch.STAGING_ROOT / commit
    marker = ROOT / "prepared.json" if phase == "probe" else OUT / "trained.json"
    if marker.exists():
        if json.loads(marker.read_text())["design_signature"] != signature():
            raise ValueError(f"Stale result {marker}")
        print(f"Completed {phase}")
        return
    script = ["timeout", f"{c.PROBE_HOURS if phase == 'probe' else c.TRAIN_HOURS}h",
              str(launch.PYTHON), "-m", "torch.distributed.run", "--standalone",
              f"--nproc_per_node={ARM.num_gpus}", "-m", "scripts.real_readout_train", phase]
    body = launch.payload("train", ARM.name, staged, "d.tarasov", "mrsndmn")
    body.update(script=f"cd {shlex.quote(str(staged))} && " + shlex.join(script),
                instance_type=launch.INSTANCE_TYPES_BY_NUM_GPUS[ARM.num_gpus],
                job_desc=f"sps-real-readout {phase} {ARM.name} #d.tarasov @mrsndmn")
    body["env_variables"].update(SPS_EXPERIMENT_SUITE="ar-sps-qwen3-1.7b-instruct",
                                 HF_HUB_DISABLE_XET="1")
    if dry:
        print(json.dumps(body))
        return
    if phase == "train":
        probe = json.loads((ROOT / "prepared.json").read_text())
        if probe["design_signature"] != signature():
            raise ValueError("Probe design mismatch")
        if probe["projected_hours"] >= c.TRAIN_HOURS:
            raise RuntimeError("Training projection exceeds allocation")
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    client, extra = training_job_api_from_profile("default")
    launch.disable_submission_retries(client)
    active = {launch.normalize(j["job_desc"]): j["job_name"]
              for j in get_in_progress_jobs(client_profile="default")}
    launch.reconcile_receipts(client, active)
    launch.reconcile_intents(active)
    identity = launch.normalize(body["job_desc"])
    if identity in active:
        print(active[identity])
        return
    launch.stage_repo(Path(launch.git("rev-parse", "--show-toplevel")), commit)
    body["region"] = extra["region"]
    print(json.dumps(launch.submit_payload(client, body, "prepare" if phase == "probe" else phase,
                                          ARM.name, commit)), flush=True)


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


if __name__ == "__main__":
    main()
