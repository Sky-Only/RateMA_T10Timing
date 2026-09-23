"""包结构与分层约束测试。"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import ratema

PKG = Path(ratema.__file__).parent

#: L0 算法层：不允许依赖任何其他内部模块
LEAF_MODULES = ["io_utils", "indicators", "backtest", "metrics", "charts"]

#: 允许的内部依赖方向（谁 -> 可以 import 谁）
ALLOWED_DEPS = {
    "io_utils": set(),
    "indicators": set(),
    "backtest": set(),
    "metrics": set(),
    "charts": set(),
    "pipeline": {"io_utils", "indicators", "backtest", "metrics"},
    "composite": {"io_utils", "indicators", "backtest", "metrics"},
    "daily": {"io_utils", "indicators"},
    "journal": {"metrics"},
    "render": {"indicators", "pipeline", "composite"},
    "writers": {"io_utils", "pipeline", "render"},
    "parser": {
        "backtest",
        "composite",
        "daily",
        "indicators",
    },
    "cli": {"commands", "commands._shared", "parser"},
    "commands._shared": {"backtest", "indicators", "io_utils", "pipeline"},
    "commands.convert": {"io_utils", "commands._shared"},
    "commands.backtest": {
        "charts",
        "io_utils",
        "parser",
        "pipeline",
        "render",
        "writers",
        "commands._shared",
    },
    "commands.signal": {"daily", "io_utils", "commands._shared"},
    "commands.composite": {
        "charts",
        "composite",
        "io_utils",
        "pipeline",
        "render",
        "writers",
        "commands._shared",
    },
    "commands.journal": {
        "composite",
        "io_utils",
        "journal",
        "pipeline",
        "commands._shared",
    },
    "commands.sweep": {
        "charts",
        "composite",
        "indicators",
        "io_utils",
        "parser",
        "pipeline",
        "writers",
        "commands._shared",
    },
    "commands.all": {
        "io_utils",
        "commands.backtest",
        "commands.composite",
        "commands.convert",
        "commands.journal",
        "commands.sweep",
    },
    "__init__": {
        "io_utils",
        "indicators",
        "backtest",
        "metrics",
        "pipeline",
        "composite",
        "daily",
        "journal",
        "charts",
    },
}


def _module_path(module: str) -> Path:
    """把 ``commands.sweep`` 这样的名字映射到实际文件。"""
    return PKG.joinpath(*module.split(".")).with_suffix(".py")


def _internal_imports(module: str) -> set[str]:
    """用 AST 提取内部依赖（含函数内延迟导入），并解析相对层级。

    ``from ._shared import``  在 ``commands.sweep`` 里应解析为 ``commands._shared``；
    ``from ..io_utils import`` 解析为 ``io_utils``。
    """
    path = _module_path(module)
    pkg_parts = module.split(".")[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level >= 1 and node.module:
            up = node.level - 1
            base = pkg_parts[: len(pkg_parts) - up] if up <= len(pkg_parts) else []
            found.add(".".join([*base, node.module]) if base else node.module)
    return found


@pytest.mark.parametrize("module", sorted(ALLOWED_DEPS))
def test_module_dependencies_follow_layering(module: str):
    """依赖必须单向、无环，且不越层。"""
    deps = _internal_imports(module)
    allowed = ALLOWED_DEPS[module]
    illegal = deps - allowed
    assert not illegal, (
        f"{module}.py 引入了未授权的内部依赖 {sorted(illegal)}；允许的是 {sorted(allowed)}"
    )


@pytest.mark.parametrize("module", LEAF_MODULES)
def test_leaf_modules_have_no_internal_dependencies(module: str):
    """L0 算法层必须零内部依赖，才能被独立单测和交叉验证复用。"""
    deps = _internal_imports(module)
    assert deps == set(), f"{module}.py 应无内部依赖，实际 {sorted(deps)}"


def test_no_circular_dependencies():
    """整图无环。"""
    graph = {m: _internal_imports(m) for m in ALLOWED_DEPS}
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str, path: list[str]) -> None:
        if node in done:
            return
        assert node not in visiting, f"检测到循环依赖：{' -> '.join(path + [node])}"
        visiting.add(node)
        for nxt in graph.get(node, set()):
            walk(nxt, path + [node])
        visiting.discard(node)
        done.add(node)

    for node in graph:
        walk(node, [])


def test_every_module_is_importable():
    """每个模块都必须能被独立导入（防止重构后残留悬空引用）。"""
    import importlib

    for module in ALLOWED_DEPS:
        if module == "__init__":
            continue
        importlib.import_module(f"ratema.{module}")


def test_importing_package_does_not_load_matplotlib():
    """`import ratema` 不应触发 matplotlib（它会在模块级设置全局后端）。

    这是 charts 必须惰性导入的原因。用子进程验证，避免被其他测试污染。
    """
    code = (
        "import sys, ratema;"
        "assert 'matplotlib' not in sys.modules, "
        "'导入 ratema 不应加载 matplotlib';"
        "assert 'matplotlib.pyplot' not in sys.modules;"
        "print(ratema.__version__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ratema.__version__


def test_lazy_attribute_still_works():
    """惰性属性必须能正常取到，且取用后才加载 matplotlib。"""
    from ratema import make_all_charts

    assert callable(make_all_charts)
    # 取过一次后应缓存进模块命名空间
    assert "make_all_charts" in vars(ratema)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = ratema.definitely_not_a_real_name


def test_all_exports_are_resolvable():
    """__all__ 里的每个名字都必须真的能取到。"""
    for name in ratema.__all__:
        assert getattr(ratema, name) is not None, f"__all__ 中的 {name} 无法解析"
