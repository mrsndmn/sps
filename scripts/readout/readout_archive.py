"""Keep the pre-refactor identities of resumable readout evaluations."""

import hashlib
import json

# Fingerprints cover the migrated evaluator and model sources. The historical
# form is accepted only with the same workload, weights and reviewed code.
_MIGRATED = {
    "gsm8k": {
        "0c0c5c913fa160b54c3631c60112118d2725c274a0a089ac95ceff251831aa53",
        "8b04836255f3ec2ec6699be8e32e0ef7281ff341d42df64547ecfc083e0cc7b2",
    },
    "humaneval": {
        "8271263538eef06dbff7b7a19f808189193c0146478b4ea278aaf332f0b6b48e",
        "362831dcffbf63a5156afd1369265767f878c39cfd704fa7037ac20a5a4b627b",
    },
    "mmlu": {
        "b77fdc99ccfbeb0d956792a3052c50cbb54e05295c9eb3afdd834ba4dc22597c",
        "af09887ba15f1a146047b3448c6107e3883dc15574731c294e0609596c8a8f84",
    },
}
# Source checksums from the readout workers at 620a1f1.
_HISTORICAL_LOADER = "f860d958d0487a7fdb5de369da0d0d02549316ab8445dd021110035b36a73977"
_HISTORICAL_WORKERS = {
    "gsm8k": "536bfe5f220b298f7ed31c19ff88baae2acf79d695a51eabad3485e959f4942b",
    "humaneval": "76240fd720ee0a56f71fe7ff83b9e21fdd9833ea506527b9e3e1d8226e6f6dff",
    "mmlu": "db30d1b819b62e8160b38668044a01c39710ae5ff76f46eb5ce9f5573eb389ed",
}


def code_fingerprint(current):
    if "serving" in current:
        serving = {key: value for key, value in current["serving"].items()
                   if key not in ("arm", "checkpoint")}
        sources = (current["code"], serving)
    else:
        sources = (current["code"], current["model_sources"])
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def compatible_identity(saved, current, marker, worker):
    if saved == current:
        return True
    training = marker.get("design_signature")
    if not training or code_fingerprint(current) not in _MIGRATED[worker]:
        return False
    historical = dict(current)
    if worker == "mmlu":
        serving = dict(historical["serving"])
        del serving["model_sources"]
        serving.update(
            training=training,
            graphs="43414e5b262421b5f319f7188f3ea8b254e2f609d8a408189838c868c48df294",
            evaluator="27a19ce43e8fc12c7c0e9f71d6479b17069cba59edf2f9778faef29ab2aacd15",
            loader=_HISTORICAL_LOADER,
            analysis=dict(serving["analysis"],
                          draft_sweep="1ee0fe13dba07265cac760e7091bf71c0722e68ed802ef60bced782a404938c8"),
        )
        historical["serving"] = serving
        historical["code"] = _HISTORICAL_WORKERS[worker]
    else:
        del historical["model_sources"]
        historical["code"] = dict(historical["code"], real_readout_eval=_HISTORICAL_LOADER)
        historical["code"]["real_readout_" + worker] = _HISTORICAL_WORKERS[worker]
        if worker == "humaneval":
            historical["training"] = training
    return saved == historical
