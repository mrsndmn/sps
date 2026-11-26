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


def values(module):
    return {
        k: v
        for k, v in vars(module).items()
        if k.isupper()
        and isinstance(v, (str, int, float, bool, dict, tuple, list, Path))
    }


def job(module, phase, run, *args, config=None, name=None):
    sources = config if isinstance(config, tuple) else (config or module,)
    settings = {}
    for source in sources:
        settings.update(values(source))
    # The launcher can own phase-specific timeouts and benchmark settings.
    settings.update(values(module))
    if is_dataclass(run):
        settings["run"] = asdict(run)
    return Job(
        name or (run if isinstance(run, str) else run.name),
        phase,
        module.__name__,
        tuple(map(str, args)),
        settings,
    )


def load(name):
    return import_module("scripts." + name)


def builds(module_name, phase, selector=None):
    module = load(module_name)
    for run in getattr(module, "job_configs", None) or module.experiment_configs:
        args = (selector, run.name) if selector else ()
        yield job(module, phase, run, *args)
