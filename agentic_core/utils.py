from pathlib import Path
from dotenv import load_dotenv

def init_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    return load_dotenv(dotenv_path=env_path) 
