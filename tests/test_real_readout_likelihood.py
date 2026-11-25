from copy import deepcopy

import pytest

from scripts.real_readout_likelihood import compare


def test_choice_change_invalidates_matching_question_hashes():
    sample = dict(doc_id=0, doc_hash="doc", prompt_hash="prompt", target_hash="gold",
                  arguments=[["Question?", "red"], ["Question?", "blue"]])
    parent = {"samples": {"arc_easy": [sample]}}
    current = deepcopy(parent)
    current["samples"]["arc_easy"][0]["arguments"][1][1] = "yellow"
    with pytest.raises(ValueError, match="choices or requests"):
        compare("arc_easy", current, parent)
