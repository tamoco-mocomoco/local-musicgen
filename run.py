#!/usr/bin/env python3
"""音楽生成サーバーの起動スクリプト (Python 標準ライブラリのみ)。

ACE-Step 1.5 の仮想環境 (ACESTEP_HOME/.venv) の Python で uvicorn を起動する。
設定は環境変数か、このフォルダの .env (KEY=VALUE) で渡す。最低限 ACESTEP_HOME が必要。

  python3 run.py              # 既定 127.0.0.1:8100
  python3 run.py --reload     # 引数はそのまま uvicorn に渡る
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_env(ROOT / ".env")
    home = Path(os.environ.get("ACESTEP_HOME", "")).expanduser()
    py = home / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not (home / "acestep").is_dir() or not py.exists():
        print(f"ACESTEP_HOME ({home or '未設定'}) に ACE-Step 1.5 のチェックアウトと .venv が見つかりません。"
              "README の「準備」を参照し、.env に ACESTEP_HOME=... を書いてください。", file=sys.stderr)
        return 1
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8100")
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print(f"local-musicgen: http://{host}:{port}  (ACE-Step: {home})")
    cmd = [str(py), "-m", "uvicorn", "server.app:app", "--host", host, "--port", port, *sys.argv[1:]]
    try:
        return subprocess.call(cmd, cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
