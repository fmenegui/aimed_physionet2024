"""Checkpoint-compatible ConvNeXt construction and deterministic inference."""

from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import timm
import torch

CLASSES = "NORM,Acute MI,Old MI,STTC,CD,HYP,PAC,PVC,AFIB/AFL,TACHY,BRADY".split(",")
MODEL_NAME = "convnext_tiny_in22k"
IMAGE_SIZE = 384
THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_WEIGHTS = [2.29125499, 104.30143541, 4.10527307, 4.16408787,
                 4.45059208, 8.22914307, 54.77135678, 19.07174103,
                 13.88471338, 25.14302191, 34.22135008]


def create_model(num_classes=11, imagenet=False):
    model = timm.create_model(MODEL_NAME, pretrained=imagenet, num_classes=num_classes)
    # Preserve the unused layer also saved by the original submission.
    model.classifier = torch.nn.Linear(model.get_classifier().in_features, num_classes)
    return model


def read_model_state(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state):
        return {key.removeprefix("model."): value for key, value in state.items()
                if key.startswith("model.")}
    return state


def training_model(initialization="ecg", checkpoint_path="pretrained_model/model_fold0.ckpt"):
    if initialization == "ecg":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing ECG pretraining checkpoint: {checkpoint_path}. "
                                    "Run python download_weights.py --kind pretrained, "
                                    "or choose --init imagenet / --init none.")
        model = create_model(num_classes=13)
        model.load_state_dict(read_model_state(checkpoint_path), strict=True)
        model.reset_classifier(len(CLASSES))
        model.classifier = torch.nn.Linear(model.get_classifier().in_features, len(CLASSES))
        return model
    if initialization not in ("imagenet", "none"):
        raise ValueError(f"Unknown initialization: {initialization}")
    return create_model(imagenet=initialization == "imagenet")


def make_transforms(training=False):
    augmentations = [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.1),
    ] if training else []
    # Preserve training's original ToGray probability. Make evaluation deterministic.
    return A.Compose(augmentations + [A.ToGray(p=0.5 if training else 1.0),
                                     A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def preprocess(image, size=IMAGE_SIZE):
    image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return make_transforms()(image=np.array(image))["image"]


def checkpoint_files(folder):
    folder = Path(folder)
    files = sorted(set(folder.glob("model*.ckpt")) | set((folder / "model").glob("*.ckpt")))
    if not files:
        raise FileNotFoundError(f"No classification checkpoints in {folder}. "
                                "Run python download_weights.py --kind ensemble first.")
    if len({path.name for path in files}) != len(files):
        raise ValueError("Duplicate checkpoint names in model folder and its model/ subfolder.")
    return files


def load_ensemble(folder, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    models = []
    for filename in checkpoint_files(folder):
        model = create_model()
        model.load_state_dict(read_model_state(filename), strict=True)
        models.append(model.to(device).eval())
    return models


def predict_probabilities(models, image):
    if not models:
        raise ValueError("At least one model is required.")
    tensor = preprocess(image).unsqueeze(0).to(next(models[0].parameters()).device)
    with torch.inference_mode():
        probabilities = [torch.sigmoid(model(tensor)).cpu() for model in models]
    return torch.stack(probabilities).mean(dim=0).squeeze(0).numpy()


def predicted_labels(probabilities):
    return [label for label, value in zip(CLASSES, probabilities) if value > THRESHOLD]
