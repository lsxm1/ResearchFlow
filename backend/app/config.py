"""Local-first configuration. Model keys never go to the browser."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
DATA = Path(os.getenv('RF_DATA_DIR', 'data'))
if not DATA.is_absolute():
    DATA = ROOT / DATA
DATA.mkdir(parents=True, exist_ok=True)
UPLOADS = DATA / 'uploads'
UPLOADS.mkdir(exist_ok=True)
DB = DATA / 'researchflow.sqlite3'
MAX_UPLOAD = int(os.getenv('RF_MAX_UPLOAD_MB', '50')) * 1024 * 1024


def model_config() -> dict:
    return {
        'mode': os.getenv('RF_MODEL_MODE', 'disabled'),
        'base_url': os.getenv('RF_MODEL_BASE_URL', '').rstrip('/'),
        'model': os.getenv('RF_MODEL_NAME', ''),
        'key': os.getenv('RF_MODEL_API_KEY', ''),
    }
