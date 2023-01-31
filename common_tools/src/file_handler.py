import json


def load_json_file(file_path: str) -> dict:
    """
    Load a json file.

    Args:
        file_path (str): Path to the json file.

    Returns:
        dict: Dictionary containing the json file content.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

