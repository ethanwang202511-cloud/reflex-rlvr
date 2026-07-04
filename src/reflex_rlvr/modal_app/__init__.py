"""Modal-app glue: image definitions, volume / secret wiring,
verifier-worker pool, mining + RL + translator entry points.

CPU-only smoketest is the only function in this package that can be
launched without explicit approval. Every GPU entry point asks
the runtime config first; no implicit GPU dispatch.
"""
