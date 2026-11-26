"""Inspect a family's declared defaults for the experiment catalog."""

from dataclasses import fields, is_dataclass


def settings(source):
    config = getattr(source, "family_config", source)
    if is_dataclass(config):
        return {
            f.name: getattr(config, f.name) for f in fields(config) if f.name.isupper()
        }
    return {k: v for k, v in vars(source).items() if k.isupper()}
