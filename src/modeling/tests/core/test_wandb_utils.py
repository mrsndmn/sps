import os

from omegaconf import OmegaConf

from training.wandb_utils import (
    default_wandb_dir_from_repo_config,
    prepare_wandb_dir,
    prepare_wandb_dir_from_config,
    resolve_wandb_dir,
)


def test_prepare_wandb_dir_from_config_uses_configured_scratch_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    cfg = OmegaConf.create(
        {
            "system": {"data_root": str(tmp_path / "scratch_project")},
            "logging": {"wandb_dir": "${system.data_root}/wandb"},
        }
    )

    resolved = prepare_wandb_dir_from_config(cfg)

    expected = tmp_path / "scratch_project" / "wandb"
    assert resolved == str(expected)
    assert expected.is_dir()
    assert os.environ["WANDB_DIR"] == str(expected)


def test_resolve_wandb_dir_falls_back_to_env(tmp_path, monkeypatch):
    env_dir = tmp_path / "env_wandb"
    monkeypatch.setenv("WANDB_DIR", str(env_dir))

    resolved = resolve_wandb_dir(create=True)

    assert resolved == str(env_dir)
    assert env_dir.is_dir()


def test_prepare_wandb_dir_does_not_overwrite_existing_env(tmp_path, monkeypatch):
    env_dir = tmp_path / "existing_env"
    configured_dir = tmp_path / "configured"
    monkeypatch.setenv("WANDB_DIR", str(env_dir))

    resolved = prepare_wandb_dir(wandb_dir=str(configured_dir))

    assert resolved == str(configured_dir)
    assert configured_dir.is_dir()
    assert os.environ["WANDB_DIR"] == str(env_dir)


def test_default_wandb_dir_honors_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "logs"))
    assert default_wandb_dir_from_repo_config(tmp_path) == str(tmp_path / "logs")
