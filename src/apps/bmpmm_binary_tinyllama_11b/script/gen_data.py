import os
import runpy
from pathlib import Path

os.environ["BMPMM_MODEL_FILTER"] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
runpy.run_path(str(Path(__file__).resolve().parents[2] / "bmpmm_binary" / "script" / "gen_data.py"), run_name="__main__")
