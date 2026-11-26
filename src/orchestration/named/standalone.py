"""Named plans for families with dedicated MLSpace submission protocols."""

from dataclasses import asdict, dataclass, is_dataclass
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class Job:
    name: str
    phase: str
    module: str
    args: tuple
    settings: dict
    submit_flag: str | None = None

    @property
    def script(self):
        if self.module == "orchestration.named.cap_jobs":
            return "scripts/evaluation/cap_jobs.py"
        family, name = self.module.removeprefix("workloads.").split(".")
        return f"scripts/{family}/{name}.py"



def values(module):
    return {
        k: v
        for k, v in vars(module).items()
        if k.isupper()
        and isinstance(v, (str, int, float, bool, dict, tuple, list, Path))
    }


def job(
    module, phase, run, *args, config=None, name=None, submit_flag=None, parameters=None
):
    sources = config if isinstance(config, tuple) else (config or module,)
    settings = {}
    for source in sources:
        settings.update(values(source))
    # The launcher can own phase-specific timeouts and benchmark settings.
    settings.update(values(module))
    if parameters:
        settings.update(parameters)
    if is_dataclass(run):
        settings["run"] = asdict(run)
    return Job(
        name or (run if isinstance(run, str) else run.name),
        phase,
        module.__name__,
        tuple(map(str, args)),
        settings,
        submit_flag,
    )


def load(name):
    return import_module(name)


def builds(module_name, phase, selector=None, *, config=None):
    module = load(module_name)
    for run in getattr(module, "job_configs", None) or module.experiment_configs:
        args = (selector, run.name) if selector else ()
        yield job(module, phase, run, *args, config=config)
