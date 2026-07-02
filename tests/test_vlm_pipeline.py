import pytest
from unittest.mock import MagicMock, patch
from src.models import get_vlm
from src.utils.metrics import calculate_metrics


def test_get_vlm_unknown_model(config):
    config['model']['name'] = 'nonexistent_model'
    with pytest.raises(ValueError, match="Unknown model"):
        get_vlm(config)


def test_get_vlm_returns_correct_type(config):
    from src.models.qwen2_5_vl import Qwen25VLModel
    config['model']['name'] = 'qwen2_5_vl'
    vlm = get_vlm(config)
    assert isinstance(vlm, Qwen25VLModel)


def test_model_id_contains_all_components(config):
    config['model']['name'] = 'qwen2_5_vl'
    config['model']['hf_model_id'] = 'Qwen/Qwen2.5-VL-7B-Instruct'
    config['model']['strategy'] = 'zero_shot'
    config['model']['load_in_8bit'] = True
    config['evaluation']['num_samples'] = 100
    vlm = get_vlm(config)
    model_id = vlm.get_model_id()
    assert model_id.startswith("VLM_")
    assert "qwen2_5_vl" in model_id
    assert "100samples" in model_id
    assert "zero_shot" in model_id
    assert "8bit" in model_id


def test_model_id_changes_with_num_samples(config):
    config['model']['name'] = 'llava'
    config['evaluation']['num_samples'] = 50
    vlm_50 = get_vlm(config)
    config['evaluation']['num_samples'] = 300
    vlm_300 = get_vlm(config)
    assert "50samples" in vlm_50.get_model_id()
    assert "300samples" in vlm_300.get_model_id()
    assert vlm_50.get_model_id() != vlm_300.get_model_id()


def test_model_id_changes_with_strategy(config):
    config['model']['name'] = 'llava'
    config['model']['strategy'] = 'zero_shot'
    vlm_zs = get_vlm(config)
    config['model']['strategy'] = 'few_shot'
    vlm_fs = get_vlm(config)
    assert "zero_shot" in vlm_zs.get_model_id()
    assert "few_shot" in vlm_fs.get_model_id()


def test_metrics_returns_all_keys():
    refs = ["a dog is running in the park", "a cat is sitting on a mat"]
    hyps = ["dog runs park", "cat sits mat"]
    metrics = calculate_metrics(refs, hyps)
    for key in ["test_bleu1", "test_bleu2", "test_bleu3", "test_bleu4",
                "test_meteor", "test_rougeL", "test_bertscore_f1"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_metrics_perfect_match():
    caption = "a dog is running in the park"
    metrics = calculate_metrics([caption], [caption])
    assert metrics["test_bleu1"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["test_bertscore_f1"] > 0.99
