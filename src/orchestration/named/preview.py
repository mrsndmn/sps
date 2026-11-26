"""Show the payload produced by the selected native launcher."""

import json
import os
import subprocess
import sys


def preview(plan):
    env = dict(os.environ, SPS_NAMED_PREVIEW="1")
    if plan.output_dir is not None:
        env["SPS_SUBMIT_OUTPUT"] = str(plan.output_dir)
    text = subprocess.check_output(
        [sys.executable, plan.script, *plan.args, "--dry"], env=env, text=True
    )
    result = json.loads(text)
    print(json.dumps(dict(settings=plan.settings, **result), default=str, indent=2))
    return 0
