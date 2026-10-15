"""Development launcher: ``python run.py`` (from the ``backend`` directory).

For production use ``uvicorn app.main:app --host 0.0.0.0 --port 8000``.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("ANATO3D_HOST", "127.0.0.1")
    port = int(os.environ.get("ANATO3D_PORT", "8000"))
    reload = os.environ.get("ANATO3D_RELOAD", "0") == "1"
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
