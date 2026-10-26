"""Software checks. Artificial labels are never a model benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from sklearn.metrics import f1_score
import torch

from aimed.model import CLASSES, checkpoint_files, load_ensemble, predict_probabilities, predicted_labels, preprocess
from examples.make_demo_data import make_data
from media.classification.callbacks import BatchAccumulatedMetricsCallback
from media.dataset.base.dataloader import BaseDataloader
from media.dataset.base.dataset import BaseDataset
from media.dataset.build_csv import build_csv
from media.dataset.get_labels import add_classes
from media.dataset.validation.stratified import stratified_group_k_fold


def test_preprocessing_is_deterministic_grayscale():
    image = Image.new("RGB", (512, 256), (230, 30, 80))
    first, second = preprocess(image), preprocess(image)
    assert first.shape == (3, 384, 384)
    assert torch.equal(first, second)
    restored = first.numpy() * np.array([0.229, 0.224, 0.225])[:, None, None]
    restored += np.array([0.485, 0.456, 0.406])[:, None, None]
    np.testing.assert_allclose(restored[0], restored[1], atol=1e-6)
    np.testing.assert_allclose(restored[1], restored[2], atol=1e-6)


def test_ensemble_averages_probabilities_and_threshold_is_strict():
    class Constant(torch.nn.Module):
        def __init__(self, probability):
            super().__init__()
            self.value = torch.nn.Parameter(torch.logit(torch.tensor([probability] * 11)))

        def forward(self, images):
            return self.value.unsqueeze(0).expand(len(images), -1)

    values = predict_probabilities([Constant(0.1), Constant(0.7)], Image.new("RGB", (32, 32)))
    np.testing.assert_allclose(values, 0.4, atol=1e-6)
    assert predicted_labels([0.5] * 11) == []
    assert predicted_labels([0.6] + [0.2] * 10) == ["NORM"]


def test_missing_and_duplicate_checkpoints_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_weights"):
        checkpoint_files(tmp_path)
    (tmp_path / "model").mkdir()
    (tmp_path / "model_fold0.ckpt").touch()
    (tmp_path / "model" / "model_fold0.ckpt").touch()
    with pytest.raises(ValueError, match="Duplicate"):
        checkpoint_files(tmp_path)


def test_incompatible_checkpoint_is_rejected(tmp_path):
    torch.save({"state_dict": {"model.wrong": torch.zeros(1)}}, tmp_path / "model_fold0.ckpt")
    with pytest.raises(RuntimeError, match="state_dict"):
        load_ensemble(tmp_path, "cpu")


def test_all_renderings_of_a_record_stay_in_one_fold(tmp_path):
    make_data(tmp_path, 20)
    for header in tmp_path.glob("*.hea"):
        name = header.stem
        Image.new("RGB", (32, 32)).save(tmp_path / f"{name}-1.png")
        header.write_text(header.read_text().replace(f"{name}-0.png", f"{name}-0.png, {name}-1.png"))
    table, _ = add_classes(build_csv(str(tmp_path)), classes=CLASSES)
    seen = set()
    for fold in stratified_group_k_fold(table, CLASSES, "header_base", n_splits=5):
        train_groups = set(table.iloc[fold["train"]].header_base)
        valid_groups = set(table.iloc[fold["valid"]].header_base)
        assert not train_groups.intersection(valid_groups)
        assert not seen.intersection(valid_groups)
        seen.update(valid_groups)
    assert len(seen) == 20


def test_zero_workers_remains_zero():
    loader = BaseDataloader(BaseDataset, dataframe=None, num_workers=0)
    assert loader.num_workers == 0


def test_callback_reports_multilabel_macro_f1():
    class Recorder:
        def log(self, name, value, **kwargs):
            self.result = value

    actual = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 0]])
    predicted = torch.tensor([[1, 0], [0, 1], [0, 1], [0, 0]])
    callback = BatchAccumulatedMetricsCallback({"f1": (f1_score, {"average": "macro"})})
    recorder = Recorder()
    callback.on_validation_epoch_start(None, recorder)
    callback.on_validation_batch_end(None, recorder, {"actual_labels": actual, "pred_labels": predicted}, None, 0)
    callback.on_validation_epoch_end(None, recorder)
    assert recorder.result == pytest.approx(f1_score(actual, predicted, average="macro"))


def test_train_save_reload_and_challenge_inference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data, output = tmp_path / "data", tmp_path / "run"
    make_data(data, 20)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    command = [sys.executable, "train.py", "-d", str(data), "-m", str(output),
               "--init", "none", "--fold", "0", "--epochs", "1", "--size", "64",
               "--batch-size", "2", "--num-workers", "0", "--device", "cpu", "--fast-dev-run", "--quiet"]
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "model" / "model_fold0.ckpt").is_file()
    assert json.loads((output / "run_config.json").read_text())["classes"] == CLASSES
    assert (output / "predictions" / "validation_fold0.csv").is_file()
    metrics = json.loads((output / "metrics.json").read_text())["0"]
    assert len(metrics["per_class_f1"]) == 11
    from team_code import load_models, run_models
    digitizer, models = load_models(output, False)
    assert digitizer is None
    assert len(models) == 1 and not models[0].training
    signal, labels = run_models(str(data / "demo000"), digitizer, models, False)
    assert signal is None and isinstance(labels, list)
    rerun = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=60)
    assert rerun.returncode != 0 and "already contains" in rerun.stderr
