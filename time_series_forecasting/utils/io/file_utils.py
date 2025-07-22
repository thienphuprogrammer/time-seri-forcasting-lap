import os

def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists.
    """
    os.makedirs(path, exist_ok=True)

def list_files(path: str) -> list:
    """
    List all files in a directory.
    """
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def get_file_info(path: str) -> dict:
    """
    Get information about a file.
    """
    return {
        'name': os.path.basename(path),
        'size': os.path.getsize(path),
        'modified': os.path.getmtime(path)
    }