from yaml import safe_load, YAMLError
import pathlib

def load_config(path: str) -> dict:
    path = pathlib.Path(path) 

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open() as f:
        try:
            config = safe_load(f)
            return config
        except YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {exc}")