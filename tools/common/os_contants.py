from pathlib import Path

# Get the root path where the code file is located
ROOT_PATH = Path(__file__).parent.parent.parent

# Get the path to the result
RESULT_PATH = str(ROOT_PATH / "result")
STATIC_PATH = str(ROOT_PATH / "static")
