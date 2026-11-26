"""MLSpace entry point for the sched instruction experiment."""
from workloads.instruction import sched_common as sc
from experiments import sched_config as c
from workloads.instruction.rerun_jobs import Rerun

_runner = Rerun("sched", sc, c, "cosine")
experiments = _runner.experiments
payload = _runner.payload
hours = _runner.hours
reserve = _runner.reserve
submit = _runner.submit
worker = _runner.worker
main = _runner.main
