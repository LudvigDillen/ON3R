from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = Path("/home2/lu2277di/data")
TRAINING_PATH = DATA_PATH / "on3r/outputs/training/"  # training checkpoints
EVAL_PATH = DATA_PATH / "on3r/outputs/results/"  # evaluation results
DATASET = "megadepth"  # default dataset for evaluation


def set_dataset(dataset_name):
    global DATASET
    if dataset_name not in ["megadepth", "scannet1500", "cambridge"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    DATASET = dataset_name
    print(f"Dataset set to: {DATASET}")