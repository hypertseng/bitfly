from pathlib import Path
import sys

COMMON = Path(__file__).resolve().parents[2] / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from bmpmm_gen_common import generate_lowp_dataset

print(generate_lowp_dataset(2, model_filter='Qwen/Qwen2.5-1.5B'), end="")
