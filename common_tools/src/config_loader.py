import os
from dotenv import load_dotenv


# Load .env file
# After this, you can access to environment like os.environ.get("VARIABLE")
def load_env_file(env_file_path: str) -> None:
    """
    Try to load environment variables from file path
    :param env_file_path: Path to the environment variable file
    """
    if os.path.isfile(env_file_path):
        load_dotenv(dotenv_path=env_file_path)

