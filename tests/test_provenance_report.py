import json
import pytest
from scripts import provenance_report as r


def test_paired_interaction_zero_and_known_effect():
    assert r.interaction([0]*10)==dict(delta_pp=0.,ci95_pp=[0.,0.],n=10)
    assert r.interaction([1]*10)['delta_pp']==100.
    with pytest.raises(ValueError):r.interaction([3])


def test_control_rejects_mismatched_completion(tmp_path,monkeypatch):
    exp=r.evaluation.experiment_configs[0]
    monkeypatch.setattr(r.evaluation,'output_dir',lambda _:tmp_path)
    (tmp_path/'eval').mkdir()
    ident={'experiment':exp.__dict__}
    (tmp_path/'evaluated.json').write_text(json.dumps(dict(identity=ident,passed=True)))
    (tmp_path/'eval/manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):r.load_control(exp)
