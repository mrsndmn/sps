"""Typed settings shared by family definitions and their worker processes."""

import json
import os
from copy import copy
from pathlib import Path


def configure(schema, module=None):
    overrides = json.loads(os.environ.get("SPS_FAMILY_ARGUMENTS", "{}"))
    values = {
        k: v
        for k, v in overrides.get(module or schema.__module__, {}).items()
        if k.isupper()
    }
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


_definition_tables = {}


def definition_rows(module, name):
    return rows(
        _definition_tables.get((module.__name__, name), getattr(module, name, []))
    )


def rows(table):
    return table.values() if isinstance(table, dict) else table


def configure_runs(module, table_name, table):
    """Reconstruct the explicitly supplied rows before workers import their aliases."""
    supplied = json.loads(os.environ.get("SPS_FAMILY_ARGUMENTS", "{}")).get(module, {})
    overrides = supplied.get(table_name, {})
    _definition_tables[module, table_name] = table
    if not overrides:
        return table
    result = table.copy()
    keys = list(result) if isinstance(result, dict) else range(len(result))
    for index, key in enumerate(keys):
        if values := overrides.get(str(index)):
            row = result[key]
            arguments = {
                name: restore(value, getattr(row, name))
                for name, value in values.items()
            }
            result[key] = type(row)(**arguments)
    return result
