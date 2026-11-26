"""Typed settings shared by family definitions and their worker processes."""

from copy import copy
import json
import os
from pathlib import Path


def configure(schema, module=None):
    overrides = json.loads(os.environ.get("SPS_FAMILY_ARGUMENTS", "{}"))
    values = overrides.get(module or schema.__module__, {})
    defaults = schema() if isinstance(schema, type) else schema
    if not values:
        return defaults
    kwargs = {
        key: restore(value, getattr(defaults, key)) for key, value in values.items()
    }
    if isinstance(schema, type):
        return schema(**kwargs)
    result = copy(defaults)
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


def restore(value, default):
    if isinstance(default, Path):
        return Path(value)
    if isinstance(default, (tuple, list, set, frozenset)):
        template = list(default)
        items = (
            restore(v, template[i % len(template)]) if template else v
            for i, v in enumerate(value)
        )
        return type(default)(items)
    if isinstance(default, dict):
        sample = next(iter(default), None)
        return {
            type(sample)(k) if sample is not None else k: restore(v, default.get(k))
            for k, v in value.items()
        }
    return value


def export_fields(config):
    """Preserve field imports used by existing workers and checkpoint readers."""

    def lookup(name):
        if name.isupper():
            return getattr(config, name)
        raise AttributeError(name)

    return lookup


def settings(source):
    config = getattr(source, "family_config", None)
    values = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    values.update((k, v) for k, v in vars(source).items() if k.isupper())
    return values
