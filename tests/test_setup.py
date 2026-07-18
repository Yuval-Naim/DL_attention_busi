"""CP0 smoke tests: the package and its config import cleanly (no torch needed)."""

import busi
from busi.config import Config, DEFAULT, MODEL_CHOICES


def test_package_imports():
    assert busi.__version__


def test_config_defaults_valid():
    cfg = Config()
    assert cfg.model_name in MODEL_CHOICES
    assert cfg.n_classes == 2
    assert cfg.experiment_name == "attention_unet_dice_ce_seed42"


def test_config_override_and_validation():
    cfg = Config(model_name="cbam_unet", seed=1)
    assert cfg.experiment_name == "cbam_unet_dice_ce_seed1"
    # invalid model name must raise
    import pytest
    with pytest.raises(ValueError):
        Config(model_name="not_a_model")


def test_config_roundtrip(tmp_path):
    cfg = Config(model_name="scse_unet", epochs=3)
    p = tmp_path / "cfg.json"
    cfg.save(str(p))
    assert Config.load(str(p)).model_name == "scse_unet"
