"""
ui-service entry point (Project LEDGER).

Run from the ui-service/ folder with:
    python -m app.main

(kept inside app/, same convention as orchestrator-api's app/main.py)

Reads its configuration (orchestrator URL, port, etc.) from app/config.py,
which is itself overridable via a .env file — see .env.example.
"""
from app.config import settings
from app.ui import LedgerUIApp


def main() -> None:
    app = LedgerUIApp(settings)
    blocks = app.build()
    try:
        blocks.launch(server_name=settings.HOST, server_port=settings.PORT, share=settings.SHARE)
    finally:
        app.client.close()


if __name__ == "__main__":
    main()
