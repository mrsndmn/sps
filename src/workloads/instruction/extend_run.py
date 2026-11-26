"""MLSpace entry point for the extend instruction experiment."""
from workloads.instruction import extend_common as sc
from experiments import extend_config as c
from workloads.instruction.rerun_jobs import Rerun

_runner = Rerun("extend", sc, c, str(c.TRAIN_STEPS), continuation=True)
experiments = _runner.experiments
payload = _runner.payload
hours = _runner.hours
reserve = _runner.reserve
submit = _runner.submit
worker = _runner.worker
main = _runner.main
