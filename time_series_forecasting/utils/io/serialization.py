import pickle
import json
from typing import Any

def save_pickle(data: Any, path: str) -> None:
    """
    Save data to a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: str) -> Any:
    """
    Load data from a pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(data: Any, path: str) -> None:
    """
    Save data to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path: str) -> Any:
    """
    Load data from a JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)