import joblib
import os

def load_model(model_path: str):
    """
    Loads a trained model from the specified file path using joblib.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)