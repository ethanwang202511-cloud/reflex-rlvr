"""Modal smoketest entry-point.

Forwards to ``reflex_rlvr.modal_app.smoketest``. Run with:

    modal run scripts/modal_smoketest.py

CPU only; ≈ $0.001 in Modal credits.
"""

from reflex_rlvr.modal_app.smoketest import main, smoketest  # noqa: F401

if __name__ == "__main__":
    main()
