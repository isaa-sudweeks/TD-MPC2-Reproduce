from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent / "TD-MPC2"
MODULE_PATH = PROJECT_ROOT / "optuna_runner.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = spec_from_file_location("tdmpc2_optuna_runner", MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load Optuna runner module from {MODULE_PATH}")

_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

OptunaWorker = _module.OptunaWorker
run_multirun = _module.run_multirun

