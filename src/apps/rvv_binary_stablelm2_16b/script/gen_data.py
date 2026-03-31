import os
import runpy
from pathlib import Path

os.environ["BMPMM_MODEL_FILTER"] = 'stabilityai/stablelm-2-1_6b'
runpy.run_path(str(Path(__file__).resolve().parents[2] / "bmpmm_binary" / "script" / "gen_data.py"), run_name="__main__")
