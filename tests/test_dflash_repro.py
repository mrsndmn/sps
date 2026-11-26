from dataclasses import replace

import pytest
import torch
from torch import nn

from experiments.dflash_repro_train import HardDFlash

from workloads.distillation import dflash_repro as jobs, dflash_repro_train as training

(BASELINE,) = HardDFlash.experiment_configs()


def test_epoch_shuffling_and_anchor_sampling():
    assert training.total_steps(93_216, BASELINE) == 11_652
    order = list(range(24))
    assert set(training.epoch_order(order, 0, BASELINE)) == set(order)
    assert training.epoch_order(order, 0, BASELINE) != training.epoch_order(
        order, 1, BASELINE
    )

    mask = [False] * 12 + [True] * 700
    first = training.sample_anchors(mask, 0, 37, BASELINE)
    assert first == training.sample_anchors(mask, 0, 37, BASELINE)
    assert first != training.sample_anchors(mask, 1, 37, BASELINE)
    assert len(set(first)) == BASELINE.max_anchors
    assert all(i >= 12 and i + BASELINE.proposals < len(mask) for i in first)


def test_block_visibility_does_not_reveal_supervised_future_or_other_blocks():
    starts = torch.tensor([2, 5])
    ctx = torch.tensor(12)
    for q, correct in [(0, [0, 1, *range(12, 28)]), (31, [*range(5), *range(28, 44)])]:
        keys = training.block_visibility(
            starts, ctx, torch.tensor(q), torch.arange(44), BASELINE
        )
        assert keys.nonzero().flatten().tolist() == correct


def test_checkpoint_restores_optimizer_cursor_and_schedule(tmp_path):
    def create():
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        return model, optimizer, schedule

    def update(model, optimizer, schedule):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).square().sum().backward()
        optimizer.step()
        schedule.step()

    original = create()
    update(*original)
    folder = tmp_path / "step-00001"
    training.save_checkpoint(folder, *original, 1, "protocol")
    restored = create()
    assert training.load_checkpoint(folder, *restored, "protocol") == 1
    update(*original)
    update(*restored)
    for expected, actual in zip(
        original[0].parameters(), restored[0].parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert original[2].state_dict() == restored[2].state_dict()


def test_worker_payload_requests_matching_training_topology_and_single_eval_gpu(
    tmp_path, monkeypatch
):
    for phase in ("probe", "train"):
        body = jobs.payload(
            BASELINE,
            phase,
            None,
            tmp_path,
            "scripts/distillation/dflash_repro.py",
            jobs.worker_arguments(training, BASELINE, phase, None),
        )
        assert body["instance_type"] == "a100.4gpu"
        assert "--nproc-per-node=4" in body["script"]
        assert body["processes_per_worker"] == 1
    body = jobs.payload(
        BASELINE,
        "eval",
        10000,
        tmp_path,
        "scripts/distillation/dflash_repro.py",
        jobs.worker_arguments(training, BASELINE, "eval", 10000),
    )
    assert body["instance_type"] == "a100.1gpu"
    assert "step-10000" in body["script"]
    assert "torch.distributed" not in body["script"]


def test_milestone_evaluation_waits_for_published_training(tmp_path, monkeypatch):
    import pytest
    from orchestration.named import publication

    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    exp = replace(BASELINE, output_dir=tmp_path / "experiment", selection_file=selection)
    monkeypatch.setattr(publication, "COMMON_ROOT", tmp_path)
    weights = training.checkpoint_path(5000, exp.output("train")) / "weights.safetensors"
    weights.parent.mkdir(parents=True)
    weights.touch()
    calls = []
    monkeypatch.setattr(
        jobs.launch.SubmissionBatch, "submit", lambda *a, **kw: calls.append(a)
    )
    arguments = jobs.worker_arguments(training, exp, "eval", 5000)
    monkeypatch.setitem(jobs.evaluation.ARCHIVED_INPUTS, "archive", jobs.evaluation.settings_digest(
        {"settings_sha256": arguments.training_settings_sha256, **training.input_provenance(arguments)}))
    with pytest.raises(RuntimeError, match="Complete DFlash training"):
        jobs.submit_phase(exp, "eval", 5000, {}, "test", arguments)
    jobs.marker(exp, "train").write_text('{"design":"archive","checkpoints":[5000]}')
    torch.save(dict(step=5000, design="other"), weights.parent / "resume.pt")
    with pytest.raises(ValueError, match="published training run"):
        jobs.submit_phase(exp, "eval", 5000, {}, "test", arguments)
    assert not calls
    torch.save(dict(step=5000, design="archive"), weights.parent / "resume.pt")
    jobs.submit_phase(exp, "eval", 5000, {}, "test", arguments)
    assert len(calls) == 1
    exp.output("eval", 5000).mkdir()
    jobs.submit_phase(exp, "eval", 5000, {}, "test", arguments)
    assert len(calls) == 1


def test_checkpoint_inputs_detect_reordered_selection(tmp_path, monkeypatch):
    from workloads.distillation import dflash_repro_eval as evaluation

    selection = tmp_path / "selection.json"
    exp = replace(BASELINE, selection_file=selection)
    selection.write_text('{"train":[1,2],"val":[3]}')
    inputs = training.training_inputs(jobs.worker_arguments(training, exp, "train", None))
    training.launch.store_json(tmp_path / "inputs.json", inputs)
    selection.write_text('{"train":[2,1],"val":[3]}')
    with pytest.raises(ValueError, match="checkpoint training inputs"):

        evaluation.evaluate(
            replace(
                jobs.worker_arguments(training, exp, "eval", 5000),
                checkpoint=tmp_path,
            )
        )


def test_archived_checkpoint_must_match_published_run_and_step(tmp_path, monkeypatch):
    from workloads.distillation import dflash_repro_eval as evaluation

    checkpoint = tmp_path / BASELINE.name / "step-05000"
    checkpoint.mkdir(parents=True)
    args = replace(jobs.worker_arguments(training, BASELINE, "eval", 5000), checkpoint=checkpoint)
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    args = replace(args, selection_file=selection)
    monkeypatch.setitem(evaluation.ARCHIVED_INPUTS, "archived", evaluation.settings_digest(
        {"settings_sha256": args.training_settings_sha256, **training.input_provenance(args)}))
    training.launch.store_json(checkpoint.parent / "trained.json", dict(design="archived", checkpoints=[5000]))
    torch.save(dict(step=5000, design="archived"), checkpoint / "resume.pt")
    evaluation.validate_checkpoint(args)
    for changed in (replace(BASELINE, lr=0.3), replace(BASELINE, corpus_revision="different")):
        digest = jobs.worker_arguments(training, changed, "eval", 5000).training_settings_sha256
        with pytest.raises(ValueError, match="training inputs"):
            evaluation.validate_checkpoint(replace(args, training_settings_sha256=digest))
    for step, design in ((10000, "archived"), (5000, "another")):
        torch.save(dict(step=step, design=design), checkpoint / "resume.pt")
        with pytest.raises(ValueError, match="published training run"):
            evaluation.validate_checkpoint(args)

def test_archived_probe_is_reported_and_skipped_before_submission(tmp_path, monkeypatch):

    design = next(iter(jobs.evaluation.ARCHIVED_INPUTS))
    exp = replace(BASELINE, output_dir=tmp_path / design[:16])
    archive = exp.output_dir / "prepared.json"
    training.launch.store_json(archive, {"design": design})
    assert jobs.marker(exp, "probe") == archive
    jobs.submit_phase(exp, "probe", None, {}, "unused", None)
    import shlex
    args = jobs.worker_arguments(training, exp, "probe", None)
    plan = jobs.payload(exp, "probe", None, tmp_path, exp.script, args)
    command = shlex.split(plan["script"])
    monkeypatch.setattr(training, "train", lambda **kwargs: pytest.fail("reran a completed probe"))
    jobs.worker(training, None, "probe", command[command.index("--worker") + 2 :])
    assert not exp.output("probe").exists()
