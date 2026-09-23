"""命令行入口（薄壳）。

真正的实现在 :mod:`ratema.commands`，参数定义在 :mod:`ratema.parser`。
"""

from __future__ import annotations

from .commands import COMMANDS
from .commands._shared import setup_console
from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    # 提前设置控制台编码，保证 --help 的中文也正常显示
    setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(COMMANDS[args.command](args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
