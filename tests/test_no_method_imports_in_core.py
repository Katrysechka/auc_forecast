import ast
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "core"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


def test_core_never_imports_methods():
    bad = []
    for f in CORE_DIR.rglob("*.py"):
        for mod in _imported_modules(f):
            if mod.startswith("methods") or mod == "methods":
                bad.append(f"{f.relative_to(CORE_DIR.parent)} imports {mod}")
    assert not bad, "core/ must not import from methods/:\n" + "\n".join(bad)
