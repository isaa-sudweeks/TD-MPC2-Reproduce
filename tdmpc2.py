from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "TD-MPC2" / "tdmpc2.py"

with open(_MODULE_PATH, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(_MODULE_PATH), "exec"), globals(), globals())
