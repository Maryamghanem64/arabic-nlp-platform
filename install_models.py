from __future__ import annotations

import sys


def main() -> int:
    print("Installing downloadable NLP models...")
    try:
        import stanza
    except Exception as exc:
        print(f"[WARN] stanza is not installed: {exc}")
        print("Install requirements first: pip install -r requirements.txt")
        return 1

    try:
        stanza.download("ar", verbose=True)
        print("[OK] Stanza Arabic models installed.")
    except Exception as exc:
        print(f"[WARN] Could not download Stanza Arabic models: {exc}")
        return 1

    print("[INFO] CAMeL models are loaded by camel-tools when available.")
    print("[INFO] Farasa may require Java and local Farasa binaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
