"""PhysioNet 2024 entry points for the AIMED classification method."""
from aimed.model import load_ensemble, predict_probabilities, predicted_labels
from helper_code import load_images

def train_models(data_folder, model_folder, verbose):
    from train import get_parser, run
    args = get_parser().parse_args(["-d", data_folder, "-m", model_folder])
    args.quiet = not verbose
    run(args)

def load_models(model_folder, verbose):
    models = load_ensemble(model_folder)
    if verbose:
        print(f"Loaded {len(models)} classification models")
    return None, models

def run_models(record, digitization_model, classification_model, verbose):
    return None, run_dx_model(classification_model, record, None, verbose)

def run_dx_model(dx_model, record, signal=None, verbose=False):
    images = load_images(record)
    if not images:
        raise ValueError(f"No images in header for {record}")
    # The submission used images[0], even with multiple pages.
    return predicted_labels(predict_probabilities(dx_model, images[0]))

def train_dx_model(data_folder, model_folder, verbose=False):
    return train_models(data_folder, model_folder, verbose)
