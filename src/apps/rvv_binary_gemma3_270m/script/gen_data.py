import os
import runpy
from pathlib import Path

os.environ["BMPMM_MODEL_FILTER"] = 'google/gemma-3-270m'
runpy.run_path(str(Path(__file__).resolve().parents[2] / "bmpmm_binary" / "script" / "gen_data.py"), run_name="__main__")
