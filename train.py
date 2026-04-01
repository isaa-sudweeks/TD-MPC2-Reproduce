from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT / "TD-MPC2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "-optuna" in sys.argv or "--optuna" in sys.argv:
    from optuna_runner import run_multirun

    forwarded = [
        arg for arg in sys.argv[1:]
        if arg not in {"-optuna", "--optuna"}
    ]
    run_multirun(forwarded)
else:
    runpy.run_path(str(PROJECT_ROOT / "train.py"), run_name="__main__")
