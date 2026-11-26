"""MLSpace entry point for the extend instruction experiment."""
from scripts import extend_common as sc, extend_config as c
from scripts.instruction.rerun_jobs import Rerun

_runner = Rerun("extend", sc, c, str(c.TRAIN_STEPS), continuation=True)
experiments = _runner.experiments
payload = _runner.payload
hours = _runner.hours
reserve = _runner.reserve
submit = _runner.submit
worker = _runner.worker
main = _runner.main


if __name__ == "__main__":
    main()
