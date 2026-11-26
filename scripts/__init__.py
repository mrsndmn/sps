"""Experiment scripts, including the original flat module names used by jobs."""
from importlib import import_module
from importlib.util import find_spec, spec_from_file_location, spec_from_loader
from pathlib import Path
import sys


_GROUPS = ('adaptation', 'distillation', 'evaluation', 'instruction', 'pilot',
           'provenance', 'readout', 'serving', 'speculative')
_ROOT = Path(__file__).parent
_MOVED_CONFIGS = {
    'ar_sps_config': 'adaptation',
    'extend_config': 'instruction',
    'ifeval_config': 'instruction',
    'instruct_config': '',
    'mmlu_config': 'evaluation',
    'sched_config': 'instruction',
    'vllm_sps_config': 'serving',
}


class _LegacyScript:
    """Load the grouped module once, or supply its code to legacy python -m jobs."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        sys.modules[module.__name__] = import_module(self.target.name)

    def get_code(self, fullname):
        return self.target.loader.get_code(self.target.name)


class _LegacyScriptFinder:
    def find_spec(self, fullname, path=None, target=None):
        package, _, name = fullname.rpartition('.')
        if name in _MOVED_CONFIGS and package in (__name__, f'{__name__}.{_MOVED_CONFIGS[name]}'):
            config = find_spec(f'src.experiments.{name}')
            return spec_from_loader(fullname, _LegacyScript(config), origin=config.origin)
        if package != __name__:
            return None
        for group in _GROUPS:
            source = _ROOT / group / f'{name}.py'
            if source.is_file() and not source.is_symlink():
                target = spec_from_file_location(f'{package}.{group}.{name}', source)
                return spec_from_loader(fullname, _LegacyScript(target), origin=str(source))
        return None


# Ordinary packages and the original top-level scripts take precedence.
sys.meta_path.append(_LegacyScriptFinder())
