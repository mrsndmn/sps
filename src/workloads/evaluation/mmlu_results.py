"""Durable sample stream recovery (stdlib-only, usable in CPU CI)."""
import json


def generation_protocol(config):
    return dict(dataset=config.DATASET, revision=config.REVISION,
                seed=config.SEED, max_new_tokens=config.MAX_NEW_TOKENS)


def matches_manifest(manifest, experiment, limit):
    from experiments.mmlu import FamilyConfig, family_config

    saved = manifest.get("experiment")
    # Early result manifests predate explicit generation settings.
    protocol = manifest.get("protocol", generation_protocol(FamilyConfig()))
    return (isinstance(saved, dict) and "limit" in manifest and manifest["limit"] == limit
            and all(saved.get(key) == value for key, value in vars(experiment).items())
            and protocol == generation_protocol(family_config))


def recover_samples(path):
    saved = []
    if not path.exists():
        return saved
    with path.open('rb+') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                # Treat a missing newline as an uncommitted row, even if JSON is valid.
                stream.truncate(start)
                break
            try:
                saved.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError('Corrupt committed sample row') from error
    return saved
