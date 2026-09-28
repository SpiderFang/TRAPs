"""`python -m traps` 的薄入口；完整 parser 與錯誤處理位於 :mod:`traps.cli`。"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - subprocess 測試驗證真正 CLI
    raise SystemExit(main())

