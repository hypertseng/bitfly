import os
import runpy
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "common"
sys_path = str(COMMON)
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)

from bmpmm_case_selection import infer_model_filter_from_app_name

os.environ["BMPMM_MODEL_FILTER"] = infer_model_filter_from_app_name(Path(__file__).resolve().parents[1].name)
runpy.run_path(str(Path(__file__).resolve().parents[2] / "rvv_INT4" / "script" / "gen_data.py"), run_name="__main__")
