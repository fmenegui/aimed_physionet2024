"""Regression checks for tutorial paths beyond the initial training smoke test."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import download_weights
import prepare_image_data
import prepare_ptbxl_data


@pytest.fixture
def metadata_case(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "00001_hr.hea").write_text(
        "00001_hr 1 500 5000\n00001_hr.dat 16 1000/mV 16 0 0 0 0 I\n"
        "# Existing note: preserve me\n")
    ptb = tmp_path / "ptb.csv"
    pd.DataFrame([{"ecg_id": 1, "recording_date": "1984-11-09 09:17:34",
                   "age": 50, "sex": 0, "height": 170, "weight": 70,
                   "scp_codes": "{'MI': 100.0}"}]).to_csv(ptb, index=False)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame([{"code": "MI", "diagnostic": 1, "diagnostic_class": "MI"}]).set_index("code").to_csv(mapping)
    sl_mapping = tmp_path / "sl_mapping.csv"
    pd.DataFrame([{"StatementNumber": 821, "Acronym": "AMI"}]).to_csv(sl_mapping, index=False)
    return SimpleNamespace(input_folder=str(source), output_folder=str(tmp_path / "prepared"),
                           ptbxl_database_file=str(ptb), ptbxl_mapping_file=str(mapping),
                           sl_mapping_file=str(sl_mapping), sl_database_file=str(tmp_path / "sl.csv"))


@pytest.mark.parametrize("statements,label", [
    ("['AMI']", "Acute MI"), ("[]", "Old MI"), (None, "Old MI"), ("['XAMI']", "Old MI")])
def test_prepare_real_format_labels_and_preserve_comments(metadata_case, statements, label):
    args = metadata_case
    pd.DataFrame([{"ecg_id": 1, "statements": statements}]).to_csv(args.sl_database_file, index=False)
    prepare_ptbxl_data.run(args)
    output = Path(args.output_folder, "00001_hr.hea").read_text()
    assert f"# Labels: {label}\n" in output
    assert "# Existing note: preserve me\n# Age:" in output
    # Repeating the documented conversion does not duplicate labels/demographics.
    args.input_folder = args.output_folder
    prepare_ptbxl_data.run(args)
    repeated = Path(args.output_folder, "00001_hr.hea").read_text()
    assert repeated.count("# Age:") == 1 and repeated.count("# Labels:") == 1


def test_empty_header_directory_has_an_actionable_error(metadata_case):
    args = metadata_case
    Path(args.input_folder, "00001_hr.hea").unlink()
    pd.DataFrame([{"ecg_id": 1, "statements": "[]"}]).to_csv(args.sl_database_file, index=False)
    with pytest.raises(FileNotFoundError, match="No WFDB headers"):
        prepare_ptbxl_data.run(args)


def test_image_registration_rejects_missing_generated_images(metadata_case):
    args = SimpleNamespace(input_folder=metadata_case.input_folder,
                           output_folder=metadata_case.output_folder)
    with pytest.raises(FileNotFoundError, match="No generated images"):
        prepare_image_data.run(args)


def test_imagenet_dependency_bridge_is_available():
    from timm.models._hub import has_hf_hub
    assert has_hf_hub(necessary=True)


def test_standard_wrapper_can_disable_worker_processes(monkeypatch):
    from train import get_parser
    monkeypatch.setenv("AIMED_NUM_WORKERS", "0")
    args = get_parser().parse_args(["-d", "data", "-m", "run"])
    assert args.num_workers == 0


def test_downloader_integrity_cache_and_license_notices(tmp_path, monkeypatch):
    data = b"checkpoint test bytes"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"ensemble": {
        "repository": "fixture/test", "revision": "pinned-revision", "directory": "unused",
        "files": [{"name": "model_fold0.ckpt", "size": len(data),
                   "sha256": hashlib.sha256(data).hexdigest()}]}}}))
    monkeypatch.setattr(download_weights, "MANIFEST", manifest)
    calls = []

    def response(url, timeout):
        calls.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr(download_weights, "urlopen", response)
    output = tmp_path / "weights"
    download_weights.download("ensemble", output)
    download_weights.download("ensemble", output)
    assert len(calls) == 1 and "pinned-revision" in calls[0]
    assert (output / "LICENSE.txt").is_file()
    assert (output / "MODEL_LICENSE.md").is_file()
    assert "Required Notice:" in (output / "NOTICE").read_text()
    (output / "model_fold0.ckpt").write_bytes(b"corrupt")
    download_weights.download("ensemble", output)
    assert len(calls) == 2 and (output / "model_fold0.ckpt").read_bytes() == data


def test_failed_weight_download_is_not_accepted(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"ensemble": {
        "repository": "fixture/test", "revision": "pinned-revision", "directory": "unused",
        "files": [{"name": "model_fold0.ckpt", "size": 5, "sha256": "0" * 64}]}}}))
    monkeypatch.setattr(download_weights, "MANIFEST", manifest)
    monkeypatch.setattr(download_weights, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    output = tmp_path / "weights"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_weights.download("ensemble", output)
    assert not (output / "model_fold0.ckpt").exists()
    assert not list(output.glob("*.part"))
