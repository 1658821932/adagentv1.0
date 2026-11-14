"""预热检索环境。

只做 import + 打开一次索引，把 chromadb 等依赖的冷加载（约 18s）和索引文件
读进操作系统页缓存，随后退出。之后任何新起的 runner 子进程 import 只需约 1s。

由 agent/prewarm.py 在任务开始时以独立子进程点火；任何异常都不影响业务流程。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    try:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass

        from .config import SearchConfig
        from ..core.vendor.sentrysearch.store import SentryStore

        db_path = SearchConfig().runtime_home / ".sentrysearch" / "db"
        if db_path.is_dir():
            SentryStore(db_path=db_path, backend="qwen_api").collection.count()

        try:
            from ..core.vendor.sentrysearch.embedder import get_embedder

            get_embedder("qwen_api")
        except Exception:
            pass
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
