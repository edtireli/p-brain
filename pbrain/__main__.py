"""``python -m pbrain`` → delegates to the CLI dispatcher."""

from pbrain.cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
