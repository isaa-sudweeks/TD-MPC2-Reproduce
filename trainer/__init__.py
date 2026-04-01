from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "TD-MPC2" / "trainer"
__path__ = [str(_PACKAGE_ROOT)]
__file__ = str(_PACKAGE_ROOT / "__init__.py")

with open(__file__, "r", encoding="utf-8") as f:
    exec(compile(f.read(), __file__, "exec"), globals(), globals())
